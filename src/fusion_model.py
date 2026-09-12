
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import GraphSAGE, GATConv, global_mean_pool


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid, out, layers, dropout, heads=4):
        super().__init__()
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(layers):
            in_ch = in_dim if i == 0 else hid
            out_ch = hid if i < layers - 1 else out
            h = heads if i < layers - 1 else 1
            self.convs.append(GATConv(in_ch, out_ch // h, heads=h,
                                      concat=(h > 1), dropout=dropout))
            if i < layers - 1:
                self.bns.append(nn.BatchNorm1d(out_ch))

    def forward(self, x, ei):
        for i, c in enumerate(self.convs):
            x = c(x, ei)
            if i < len(self.convs) - 1:
                x = self.dropout(F.relu(self.bns[i](x)))
        return x


class FusionModel(nn.Module):
    def __init__(self, gnn_type="graphsage", gnn_input_dim=12,
                 gnn_hidden_dim=128, gnn_output_dim=128, gnn_num_layers=2,
                 bert_model_name="bert-base-uncased",
                 fusion_method="cross_attention", num_tags=60,
                 num_heads=4, dropout=0.3, freeze_bert=False):
        super().__init__()
        self.fusion_method = fusion_method

        if gnn_type == "graphsage":
            self.gnn = GraphSAGE(
                in_channels=gnn_input_dim,
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_output_dim,
                num_layers=gnn_num_layers,
                dropout=dropout,
                norm=nn.BatchNorm1d(gnn_hidden_dim),
            )
        elif gnn_type == "gat":
            self.gnn = GATEncoder(gnn_input_dim, gnn_hidden_dim,
                                  gnn_output_dim, gnn_num_layers, dropout)
        else:
            raise ValueError(gnn_type)

        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        bert_dim = self.bert.config.hidden_size

        if fusion_method == "concat":
            self.text_proj = nn.Linear(bert_dim, gnn_output_dim)
            self.fuse = nn.Sequential(
                nn.Linear(gnn_output_dim * 2, gnn_output_dim),
                nn.ReLU(), nn.Dropout(dropout))
            head_in = gnn_output_dim
        elif fusion_method == "cross_attention":
            self.q_proj = nn.Linear(gnn_output_dim, gnn_output_dim)
            self.kv_proj = nn.Linear(bert_dim, gnn_output_dim)
            self.attn = nn.MultiheadAttention(gnn_output_dim, num_heads,
                                              dropout=dropout, batch_first=True)
            head_in = gnn_output_dim
        elif fusion_method == "bert_only":
            self.text_proj = nn.Linear(bert_dim, gnn_output_dim)
            head_in = gnn_output_dim
        elif fusion_method == "gnn_only":
            head_in = gnn_output_dim
        else:
            raise ValueError(fusion_method)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_in, num_tags),
        )

    def forward(self, graph_x, edge_index, graph_batch,
                input_ids, attention_mask, return_embeddings=False):
        g = None
        if self.fusion_method != "bert_only":
            node = self.gnn(graph_x, edge_index)
            g = global_mean_pool(node, graph_batch)

        t_cls = t_seq = None
        if self.fusion_method != "gnn_only":
            bo = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Masked mean pooling over token embeddings
            mask = attention_mask.unsqueeze(-1).float()
            t_cls = (bo.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            t_seq = bo.last_hidden_state

        if self.fusion_method == "concat":
            fused = self.fuse(torch.cat([g, self.text_proj(t_cls)], 1))
        elif self.fusion_method == "cross_attention":
            kv = self.kv_proj(t_seq)
            q = self.q_proj(g).unsqueeze(1)
            a, _ = self.attn(q, kv, kv, need_weights=False)
            fused = a.squeeze(1)
        elif self.fusion_method == "bert_only":
            fused = self.text_proj(t_cls)
        elif self.fusion_method == "gnn_only":
            fused = g

        fused = self.dropout(fused)
        out = {"tag_logits": self.head(fused)}
        if return_embeddings:
            out["embeddings"] = fused
        return out
