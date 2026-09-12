
"""Dataset + collate for MusicCaps fusion training."""
import json, ast
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


def parse_aspect_list(raw):
    if not isinstance(raw, str): return []
    try: items = ast.literal_eval(raw)
    except (ValueError, SyntaxError): return []
    return [str(x).strip().lower() for x in items if str(x).strip()]


class MusicCapsFusionDataset(Dataset):
    def __init__(self, split_csv, vocab_path, graph_dir, tokenizer, max_length=128):
        self.df = pd.read_csv(split_csv).reset_index(drop=True)
        self.vocab = json.loads(Path(vocab_path).read_text())
        self.t2i = {t: i for i, t in enumerate(self.vocab)}
        self.graph_dir = Path(graph_dir)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.df)

    def _multihot(self, raw):
        y = torch.zeros(len(self.vocab))
        for t in parse_aspect_list(raw):
            if t in self.t2i: y[self.t2i[t]] = 1.0
        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ytid = row["ytid"]
        gp = self.graph_dir / f"{ytid}.pt"
        if gp.exists():
            g = torch.load(gp, weights_only=False)
        else:
            g = Data(x=torch.zeros(1, 12),
                     edge_index=torch.tensor([[0],[0]], dtype=torch.long),
                     num_nodes=1)
        cap = row.get("caption", "")
        if not isinstance(cap, str) or not cap.strip():
            cap = "[PAD]"
        enc = self.tok(cap, truncation=True, padding="max_length",
                       max_length=self.max_length, return_tensors="pt")
        return {
            "graph": g,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self._multihot(row["aspect_list"]),
            "ytid": ytid,
            "caption": cap,
        }


def collate_fn(batch):
    gs = Batch.from_data_list([b["graph"] for b in batch])
    return {
        "graph_x": gs.x,
        "edge_index": gs.edge_index,
        "graph_batch": gs.batch,
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "ytids": [b["ytid"] for b in batch],
        "captions": [b["caption"] for b in batch],
    }
