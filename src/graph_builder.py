
"""Build segment graphs from audio (chroma features + similarity edges)."""
import warnings
from pathlib import Path
import numpy as np
import torch
import librosa
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
warnings.filterwarnings("ignore")


class MusicGraphBuilder:
    def __init__(self, sample_rate=22050, segment_duration=5.0, hop_length=512,
                 n_chroma=12, feature_type="chroma", similarity_threshold=0.7):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.feature_type = feature_type
        self.similarity_threshold = similarity_threshold

    def _features(self, y):
        if self.feature_type == "chroma":
            f = librosa.feature.chroma_stft(
                y=y, sr=self.sample_rate, hop_length=self.hop_length,
                n_chroma=self.n_chroma)
        else:
            f = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=y, sr=self.sample_rate, hop_length=self.hop_length),
                ref=np.max)
        return f.T  # (T, D)

    def _segment(self, feats):
        fps = int(self.segment_duration * self.sample_rate / self.hop_length)
        n = max(1, feats.shape[0] // fps)
        segs = []
        for i in range(n):
            s, e = i * fps, min((i + 1) * fps, feats.shape[0])
            if e - s < fps // 2 and i > 0:
                break
            segs.append(feats[s:e].mean(0))
        return np.stack(segs, 0)

    def build(self, audio_path):
        try:
            y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            if len(y) < self.sample_rate:
                return None
            feats = self._features(y)
            seg = self._segment(feats)
            # Normalize
            seg = (seg - seg.mean(0, keepdims=True)) / (seg.std(0, keepdims=True) + 1e-6)
            x = torch.tensor(seg, dtype=torch.float32)
            n = x.shape[0]

            edges = []
            if n > 1:
                sim = cosine_similarity(seg)
                for i in range(n):
                    for j in range(i + 1, n):
                        if sim[i, j] > self.similarity_threshold:
                            edges += [(i, j), (j, i)]
                for i in range(n - 1):
                    edges += [(i, i + 1), (i + 1, i)]
            if not edges:
                edges = [(i, i) for i in range(n)]

            ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
            return Data(x=x, edge_index=ei, num_nodes=n)
        except Exception as e:
            print(f"[graph_builder] {audio_path}: {e}")
            return None


def build_all_graphs(audio_dir, output_dir, ytids, builder):
    audio_dir, output_dir = Path(audio_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    built = skipped = failed = 0
    for ytid in tqdm(ytids, desc="Building graphs"):
        out = output_dir / f"{ytid}.pt"
        if out.exists():
            skipped += 1; continue
        ap = audio_dir / f"{ytid}.wav"
        if not ap.exists():
            failed += 1; continue
        g = builder.build(str(ap))
        if g is None:
            failed += 1; continue
        torch.save(g, out)
        built += 1
    print(f"Built={built} Skipped={skipped} Failed={failed}")
    return built, skipped, failed
