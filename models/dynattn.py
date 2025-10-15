import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class DynamicPathCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_paths=4, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_paths = num_paths
        self.top_k = top_k

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj_list = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_paths)])
        self.v_proj_list = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_paths)])

        # 动态路径打分器：Q → path_scores
        self.path_score_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_paths)
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, source_features):
        """
        Q: [B, L_q, D]
        source_features: List of [B, L_k_i, D] for i in 1..N
        """
        B, L_q, D = Q.shape
        Q_proj = self.q_proj(Q)  # [B, L_q, D]

        # 每个路径的 attention 输出
        outputs = []
        for i in range(self.num_paths):
            K = self.k_proj_list[i](source_features[i])  # [B, L_k, D]
            V = self.v_proj_list[i](source_features[i])  # [B, L_k, D]
            attn_logits = torch.matmul(Q_proj, K.transpose(-2, -1)) / math.sqrt(D)
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_out = torch.matmul(attn_weights, V)  # [B, L_q, D]
            outputs.append(attn_out)

        # 路径权重计算（动态选择）
        path_logits = self.path_score_mlp(Q_proj.mean(dim=1))  # [B, N]
        path_scores = F.softmax(path_logits, dim=-1)  # [B, N]

        # 只选择 top-k 路径（稀疏选择）
        topk_values, topk_indices = torch.topk(path_scores, self.top_k, dim=-1)  # [B, k]
        mask = torch.zeros_like(path_scores)
        mask.scatter_(1, topk_indices, 1.0)
        sparse_weights = path_scores * mask  # [B, N]
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=1, keepdim=True) + 1e-8)

        # 融合多个输出
        outputs_stack = torch.stack(outputs, dim=1)  # [B, N, L_q, D]
        weights_expanded = sparse_weights.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        out = (outputs_stack * weights_expanded).sum(dim=1)  # [B, L_q, D]

        return self.out_proj(out)  # [B, L_q, D]

# # 假设主干输出 Q 和多个模型输出 K/V
# Q = torch.randn(4, 32, 256)
# feat1 = torch.randn(4, 64, 256)
# feat2 = torch.randn(4, 49, 256)
# feat3 = torch.randn(4, 100, 256)
# feat4 = torch.randn(4, 36, 256)
#
# model = DynamicPathCrossAttention(embed_dim=256, num_paths=4, top_k=2)
# out = model(Q, [feat1, feat2, feat3, feat4])  # [4, 32, 256]

