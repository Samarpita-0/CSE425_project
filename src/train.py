
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from sklearn.metrics import f1_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*No positive class.*")

sys.path.append("/content/CSE425_project/src")
from data_loader import MusicCapsFusionDataset, collate_fn
from fusion_model import FusionModel
from tag_pipeline import categorize_tags, category_indices


def compute_metrics(T, P, cat_idx, thr=0.5):
    b = (P > thr).astype(int)
    m = {
        "macro_f1": f1_score(T, b, average="macro", zero_division=0),
        "micro_f1": f1_score(T, b, average="micro", zero_division=0),
        "auc_pr":   average_precision_score(T, P, average="macro")
                    if T.sum() > 0 else 0.0,
    }
    for cat, idxs in cat_idx.items():
        if not idxs: continue
        ct, cb = T[:, idxs], b[:, idxs]
        if ct.sum() > 0:
            m[f"{cat}_f1"] = f1_score(ct, cb, average="macro", zero_division=0)
    return m


def best_threshold(T, P, grid=np.arange(0.05, 0.55, 0.05)):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        b = (P > t).astype(int)
        if b.sum() == 0: continue
        f1 = f1_score(T, b, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def run_epoch(model, loader, opt, crit, dev, train, cat_idx):
    model.train() if train else model.eval()
    losses, T, P = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, leave=False, desc="train" if train else "eval")
        for b in pbar:
            gx = b["graph_x"].to(dev); ei = b["edge_index"].to(dev)
            gb = b["graph_batch"].to(dev)
            ii = b["input_ids"].to(dev); am = b["attention_mask"].to(dev)
            y  = b["labels"].to(dev)
            out = model(gx, ei, gb, ii, am)
            logits = out["tag_logits"]
            loss = crit(logits, y)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())
            T.append(y.detach().cpu().numpy())
            P.append(torch.sigmoid(logits).detach().cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    T, P = np.vstack(T), np.vstack(P)
    m = {"loss": float(np.mean(losses))}
    m.update(compute_metrics(T, P, cat_idx))
    m["T"] = T
    m["P"] = P
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_method", default="cross_attention",
                    choices=["concat","cross_attention","bert_only","gnn_only"])
    ap.add_argument("--gnn_type", default="graphsage", choices=["graphsage","gat"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--freeze_bert", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    ROOT = Path("/content/CSE425_project")
    DATA = ROOT / "data/task3_ready"
    GRAPHS = ROOT / "data/processed/graphs"
    tag = f"{args.fusion_method}_{args.gnn_type}"
    SAVE = ROOT / "models" / tag; SAVE.mkdir(parents=True, exist_ok=True)
    LOG  = ROOT / "runs" / tag

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    vocab = json.loads((DATA / "tag_vocab.json").read_text())
    cats = categorize_tags(vocab, verbose=False)
    cat_idx = {c: category_indices(vocab, cats, c) for c in cats}
    print(f"Vocab={len(vocab)}")

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = MusicCapsFusionDataset(DATA/"train.csv", DATA/"tag_vocab.json", GRAPHS, tok)
    val_ds   = MusicCapsFusionDataset(DATA/"val.csv",   DATA/"tag_vocab.json", GRAPHS, tok)
    test_ds  = MusicCapsFusionDataset(DATA/"test.csv",  DATA/"tag_vocab.json", GRAPHS, tok)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=args.num_workers)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=args.num_workers)
    test_ld  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=args.num_workers)

    model = FusionModel(
        gnn_type=args.gnn_type, gnn_input_dim=12,
        gnn_hidden_dim=128, gnn_output_dim=128, gnn_num_layers=2,
        fusion_method=args.fusion_method, num_tags=len(vocab),
        num_heads=4, dropout=0.3, freeze_bert=args.freeze_bert,
    ).to(dev)

    # ---- Split LR ----
    bert_params = [p for n, p in model.named_parameters()
                   if n.startswith("bert.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("bert.") and p.requires_grad]
    print(f"Trainable BERT params: {sum(p.numel() for p in bert_params):,}")
    print(f"Trainable head params: {sum(p.numel() for p in head_params):,}")

    opt = optim.AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10},
    ], weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                               factor=0.5, patience=3)

    # ---- Capped pos_weight ----
    all_y = np.stack([train_ds[i]["labels"].numpy() for i in range(len(train_ds))])
    pos = all_y.sum(axis=0); neg = len(all_y) - pos
    pos_weight = torch.tensor(
        np.clip(neg / (pos + 1e-6), 1.0, 10.0), dtype=torch.float32
    ).to(dev)
    print(f"pos_weight: min={pos_weight.min():.2f} "
          f"max={pos_weight.max():.2f} mean={pos_weight.mean():.2f}")
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    writer = SummaryWriter(str(LOG))
    best_val_loss = float("inf")
    best_ep, best_thresh = 0, 0.5

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_ld, opt, crit, dev, True, cat_idx)
        va = run_epoch(model, val_ld,   opt, crit, dev, False, cat_idx)
        t_opt, f1_opt = best_threshold(va["T"], va["P"])

        print(f"\nEpoch {ep}/{args.epochs}")
        print(f"  train loss={tr['loss']:.4f} mF1={tr['macro_f1']:.4f} "
              f"AUC-PR={tr['auc_pr']:.4f}")
        print(f"  val   loss={va['loss']:.4f} mF1={va['macro_f1']:.4f} "
              f"| @thr={t_opt:.2f} mF1={f1_opt:.4f} AUC-PR={va['auc_pr']:.4f}")

        for k, v in tr.items():
            if isinstance(v, (int, float)): writer.add_scalar(f"train/{k}", v, ep)
        for k, v in va.items():
            if isinstance(v, (int, float)): writer.add_scalar(f"val/{k}", v, ep)

        sch.step(va["loss"])

        if va["loss"] < best_val_loss:
            best_val_loss, best_ep, best_thresh = va["loss"], ep, t_opt
            torch.save({"model": model.state_dict(), "vocab": vocab,
                        "args": vars(args), "epoch": ep,
                        "val_loss": best_val_loss, "threshold": best_thresh},
                       SAVE / "best.pt")
            print(f"  ✓ saved (val_loss={best_val_loss:.4f}, thr={t_opt:.2f})")

    print(f"\nBest val_loss={best_val_loss:.4f} @ ep{best_ep} | thr={best_thresh:.2f}")

    # ---- Test with best checkpoint + tuned threshold ----
    ck = torch.load(SAVE / "best.pt", weights_only=False)
    model.load_state_dict(ck["model"])
    best_thresh = ck["threshold"]
    te = run_epoch(model, test_ld, opt, crit, dev, False, cat_idx)
    b = (te["P"] > best_thresh).astype(int)

    te_final = {
        "threshold": float(best_thresh),
        "epoch": int(best_ep),
        "macro_f1": f1_score(te["T"], b, average="macro", zero_division=0),
        "micro_f1": f1_score(te["T"], b, average="micro", zero_division=0),
        "auc_pr":   average_precision_score(te["T"], te["P"], average="macro"),
    }
    for cat, idxs in cat_idx.items():
        if not idxs: continue
        ct, cb = te["T"][:, idxs], b[:, idxs]
        if ct.sum() > 0:
            te_final[f"{cat}_f1"] = f1_score(ct, cb, average="macro", zero_division=0)

    print("\n=== Test (best checkpoint + tuned threshold) ===")
    for k, v in te_final.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    (SAVE / "test_metrics.json").write_text(json.dumps(te_final, indent=2))
    writer.close()


if __name__ == "__main__":
    main()
