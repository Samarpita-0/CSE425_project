
"""
Tag pipeline for MusicCaps' aspect_list.
  parse_aspect_list  -> ['pop','melancholic'] from "['pop','melancholic']"
  build_tag_vocab    -> top-K tags from TRAIN split only
  compute_tag_stats  -> label distribution diagnostics
  categorize_tags    -> structural regex + embedding centroid categorization
"""
import ast, json, re, functools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Parsing
def parse_aspect_list(raw):
    if not isinstance(raw, str):
        return []
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return [str(x).strip().lower() for x in items if str(x).strip()]


# Vocabulary
def build_tag_vocab(df, top_k=60, min_count=5):
    counter = Counter()
    for raw in df["aspect_list"]:
        counter.update(parse_aspect_list(raw))
    return [t for t, c in counter.most_common() if c >= min_count][:top_k]


def save_vocab(vocab, path):
    Path(path).write_text(json.dumps(vocab, indent=2))


def load_vocab(path):
    return json.loads(Path(path).read_text())


def compute_tag_stats(df, vocab):
    all_counts = Counter()
    multihot = []
    for raw in df["aspect_list"]:
        tags = parse_aspect_list(raw)
        all_counts.update(tags)
        multihot.append([1 if t in tags else 0 for t in vocab])
    mh = torch.tensor(multihot, dtype=torch.float32)
    vocab_counts = [(t, all_counts.get(t, 0)) for t in vocab]
    return {
        "vocab_size": len(vocab),
        "total_unique_tags_in_data": len(all_counts),
        "most_common_overall": all_counts.most_common(10),
        "least_common_in_vocab": sorted(vocab_counts, key=lambda x: x[1])[:10],
        "avg_labels_per_sample": mh.sum(dim=1).mean().item(),
        "max_labels_per_sample": int(mh.sum(dim=1).max().item()),
        "samples_with_zero_labels": int((mh.sum(dim=1) == 0).sum().item()),
        "sparsity": 1 - (mh.sum().item() / (len(df) * len(vocab))),
    }


# Categorization
# Layer 1: structural regex rules (order matters -- first match wins)
_STRUCTURAL_RULES = [
    ("vocal",      re.compile(r"\b(vocal|voice|singer|singing|choir|humming|instrumental)\b")),
    ("tempo",      re.compile(r"(\btempo\b|uptempo|\bmid-tempo\b|\bmidtempo\b|\bslow\b|\bfast\b)")),
    ("production", re.compile(r"\b(quality|recording|mono|stereo|noisy|reverb|distortion|mix|muddy|lo-fi|hifi|live|performance)\b")),
    ("instrument", re.compile(r"\b(piano|guitar|bass|drum|snare|kick|hat|synth|organ|violin|"
                              r"sax|trumpet|flute|cello|string|percussion|clap|keyboard|"
                              r"bell|harp|accordion|banjo|mandolin|clarinet|trombone)\b")),
    ("structure",  re.compile(r"\b(intro|outro|verse|chorus|bridge|riff|groove|beat|rhythm|"
                              r"melody|melodic|harmonic|chord|progression)\b")),
]

# Layer 2: seed phrases for embedding-based nearest-centroid fallback
_CATEGORY_SEEDS = {
    "genre":      ["pop music", "rock song", "jazz tune", "electronic dance track",
                   "folk ballad", "hip hop beat", "classical piece"],
    "mood":       ["happy and joyful", "sad and melancholic", "energetic and intense",
                   "calm and relaxing", "aggressive and angry", "romantic and sentimental",
                   "dark and moody", "uplifting and hopeful"],
    "instrument": ["solo piano", "electric guitar riff", "acoustic guitar",
                   "synthesizer pad", "live drums", "bass line"],
    "vocal":      ["male singer", "female vocalist", "instrumental no vocals",
                   "choir singing", "spoken word"],
    "tempo":      ["slow tempo", "fast tempo", "moderate tempo", "uptempo dance"],
    "production": ["lo-fi recording", "clean studio mix", "live performance",
                   "amateur recording", "noisy audio"],
    "structure":  ["has a catchy chorus", "extended instrumental intro",
                   "repeating chord progression", "drop and build"],
}

_FALLBACK = "other"


@functools.lru_cache(maxsize=1)
def _get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@functools.lru_cache(maxsize=1)
def _get_centroids():
    enc = _get_encoder()
    centroids = {}
    for cat, seeds in _CATEGORY_SEEDS.items():
        embs = enc.encode(seeds, normalize_embeddings=True)
        c = embs.mean(axis=0)
        centroids[cat] = c / (np.linalg.norm(c) + 1e-9)
    return centroids


def _structural_category(tag):
    for cat, pat in _STRUCTURAL_RULES:
        if pat.search(tag):
            return cat
    return None


def _semantic_category(tag, min_similarity=0.25):
    enc = _get_encoder()
    v = enc.encode([tag], normalize_embeddings=True)[0]
    best_cat, best_sim = None, -1.0
    for cat, c in _get_centroids().items():
        sim = float(np.dot(v, c))
        if sim > best_sim:
            best_sim, best_cat = sim, cat
    return best_cat if best_sim >= min_similarity else None


def categorize_tag(tag):
    """Assign one tag -> (category, method). NOT recursive."""
    cat = _structural_category(tag)
    if cat is not None:
        return cat, "structural"
    cat = _semantic_category(tag)
    if cat is not None:
        return cat, "semantic"
    return _FALLBACK, "fallback"


def categorize_tags(vocab, verbose=True):
    """Assign every tag in vocab to a category. Returns {cat: [tags]}."""
    buckets = defaultdict(list)
    provenance = {}
    for tag in vocab:
        cat, how = categorize_tag(tag)
        buckets[cat].append(tag)
        provenance[tag] = how

    if verbose:
        print("\n[tag_pipeline] category assignment:")
        for cat, tags in sorted(buckets.items(), key=lambda x: -len(x[1])):
            how_counts = Counter(provenance[t] for t in tags)
            detail = ", ".join(f"{k}={v}" for k, v in how_counts.items())
            print(f"  {cat:12s} ({len(tags):2d})  [{detail}]")

    return dict(buckets)


def category_indices(vocab, categorized, category):
    s = set(categorized.get(category, []))
    return [i for i, t in enumerate(vocab) if t in s]


# Validation & splits
def validate_graphs(df, graph_dir):
    graph_dir = Path(graph_dir)
    missing = [y for y in df["ytid"] if not (graph_dir / f"{y}.pt").exists()]
    return {
        "total_samples": len(df),
        "missing_graphs": len(missing),
        "missing_percentage": 100 * len(missing) / len(df) if len(df) else 0,
        "sample_missing": missing[:5],
    }


def create_musiccaps_splits(df, output_dir, val_ratio=0.15):
    """Official split: test = is_audioset_eval==1; train/val carved from rest."""
    df = df.copy()
    df["is_audioset_eval"] = df["is_audioset_eval"].astype(int)

    test = df[df["is_audioset_eval"] == 1].reset_index(drop=True)
    train_val = df[df["is_audioset_eval"] == 0].reset_index(drop=True)

    if "author_id" in train_val.columns and train_val["author_id"].notna().any():
        from sklearn.model_selection import GroupShuffleSplit
        groups = train_val["author_id"].fillna("unknown").astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
        tr_idx, va_idx = next(splitter.split(train_val, groups=groups))
        train, val = train_val.iloc[tr_idx], train_val.iloc[va_idx]
    else:
        from sklearn.model_selection import train_test_split
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)} (official eval)")
    return train, val, test
