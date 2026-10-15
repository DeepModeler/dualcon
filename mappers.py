import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional, Union



class MemoryBank:
    def __init__(self, device, feature_dim, bank_size=1000):
        self.bank_qi = torch.zeros(bank_size, feature_dim, device=device)
        self.bank_a = torch.zeros(bank_size, feature_dim, device=device)
        self.ptr = 0
        self.size = 0
        self.bank_size = bank_size

    @torch.no_grad()
    def update(self, new_qi, new_a):
        B = new_qi.size(0)
        assert new_a.size(0) == B, "size mismatch"

        if self.ptr + B > self.bank_size:
            overflow = self.ptr + B - self.bank_size
            self.bank_qi[self.ptr:] = new_qi[:B - overflow]
            self.bank_a[self.ptr:] = new_a[:B - overflow]
            self.bank_qi[:overflow] = new_qi[B - overflow:]
            self.bank_a[:overflow] = new_a[B - overflow:]
            self.ptr = overflow
        else:
            self.bank_qi[self.ptr:self.ptr + B] = new_qi
            self.bank_a[self.ptr:self.ptr + B] = new_a
            self.ptr = (self.ptr + B) % self.bank_size

        self.size = min(self.size + B, self.bank_size)

    def get(self):
        return self.bank_qi[:self.size], self.bank_a[:self.size]



# ---------- Prototype-level Memory Bank ----------
class PrototypeBank:
    def __init__(self, n_proto, dim, device, bank_size, momentum=0.9):
        self.protos_qi = torch.zeros(n_proto, dim, device=device)
        self.protos_a = torch.zeros(n_proto, dim, device=device)
        self.counts = torch.zeros(n_proto, device=device)
        self.m = momentum
        self.bank_size = bank_size
        self.memory_filled = False

    @torch.no_grad()
    def update(self, proto_ids, feats_qi, feats_a, current_size):
        if current_size >= self.bank_size:
            self.memory_filled = True

        for pid, f_qi, f_a in zip(proto_ids.tolist(), feats_qi, feats_a):
            if not self.memory_filled:
                self.protos_qi[pid] = (self.protos_qi[pid] * self.counts[pid] + f_qi) / (self.counts[pid] + 1)
                self.protos_a[pid] = (self.protos_a[pid] * self.counts[pid] + f_a) / (self.counts[pid] + 1)
            else:  # moving average
                self.protos_qi[pid] = F.normalize(self.m * self.protos_qi[pid] + (1 - self.m) * f_qi, dim=-1)
                self.protos_a[pid] = F.normalize(self.m * self.protos_a[pid] + (1 - self.m) * f_a, dim=-1)
            self.counts[pid] += 1

    def get(self, ids=None):
        if ids is None:
            return self.protos_qi, self.protos_a
        return self.protos_qi[ids], self.protos_a[ids]


# ---------- Sample-level Memory Bank ----------
class MemoryBank_n:
    def __init__(self, bank_size, dim, device):
        self.bank_qi = torch.zeros(bank_size, dim, device=device)
        self.bank_a = torch.zeros(bank_size, dim, device=device)
        self.bank_label = torch.full((bank_size,), -1, device=device, dtype=torch.long)
        self.ptr = 0
        self.size = 0
        self.bank_size = bank_size

    @torch.no_grad()
    def enqueue(self, qifeats, afeats, labels):
        B = qifeats.size(0)
        if B > self.bank_size:
            qifeats, afeats, labels = qifeats[:self.bank_size], afeats[:self.bank_size], labels[:self.bank_size]
            B = self.bank_size

        end = self.ptr + B
        if end <= self.bank_size:
            self.bank_qi[self.ptr:end] = qifeats
            self.bank_a[self.ptr:end] = afeats
            self.bank_label[self.ptr:end] = labels
        else:
            overflow = end - self.bank_size
            self.bank_qi[self.ptr:] = qifeats[:B-overflow]
            self.bank_a[self.ptr:] = afeats[:B-overflow]
            self.bank_label[self.ptr:] = labels[:B-overflow]
            self.bank_qi[:overflow] = qifeats[B-overflow:]
            self.bank_a[:overflow] = afeats[B-overflow:]
            self.bank_label[:overflow] = labels[B-overflow:]
        self.ptr = (self.ptr + B) % self.bank_size
        self.size = min(self.size + B, self.bank_size)

    def get(self):
        return self.bank_qi[:self.size], self.bank_a[:self.size], self.bank_label[:self.size]


class DualMemoryBank:
    def __init__(self, bank_size_open, bank_size_close,
                 feat_dim, n_proto_open, n_proto_close,
                 global2local_open, global2local_close,
                 device):

        # sample-level MemoryBank
        self.open_bank = MemoryBank_n(bank_size_open, feat_dim, device)
        self.close_bank = MemoryBank_n(bank_size_close, feat_dim, device)

        # proto-level MemoryBank
        self.proto_open = PrototypeBank(n_proto_open, feat_dim, device, bank_size_open)
        self.proto_close = PrototypeBank(n_proto_close, feat_dim, device, bank_size_close)

        self.global2local_open = global2local_open
        self.global2local_close = global2local_close

    @torch.no_grad()
    def update_samples(self, qi, a, labels, is_close):
        if is_close == 0:
            self.open_bank.enqueue(qi, a, labels)
        elif is_close == 1:
            self.close_bank.enqueue(qi, a, labels)

    @torch.no_grad()
    def update_prototypes(self, qi, a, proto_ids_global, is_close):
        if is_close == 0:
            proto_ids = torch.tensor([self.global2local_open[pid.item()] for pid in proto_ids_global],
                                     device=qi.device, dtype=torch.long)
            current_size = self.open_bank.size
            self.proto_open.update(proto_ids, qi, a, current_size)

        elif is_close == 1:
            proto_ids = torch.tensor([self.global2local_close[pid.item()] for pid in proto_ids_global],
                                     device=qi.device, dtype=torch.long)
            current_size = self.close_bank.size
            self.proto_close.update(proto_ids, qi, a, current_size)

    def get_samples(self, is_close):
        if is_close == 0:
            return self.open_bank.get()
        elif is_close == 1:
            return self.close_bank.get()

    def get_prototypes(self, is_close):
        if is_close == 0:
            return self.proto_open.get()
        elif is_close == 1:
            return self.proto_close.get()



class QueryAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, multimodal_sequence):
        B = multimodal_sequence.size(0)
        query = self.query_token.expand(B, -1, -1)
        attn_output, _ = self.attn(query, multimodal_sequence, multimodal_sequence)
        output = self.norm(attn_output.squeeze(1))
        return output


class ProjectionHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d)
        )
    def forward(self, x):
        return self.mlp(x)
        #return F.normalize(self.mlp(x), dim=-1)


class T2I_Attention(nn.Module):
    def __init__(self, im_dim, emb_dim, att_dropout=0.1, bias=True):
        super(T2I_Attention, self).__init__()
        self.inlinear = nn.Linear(im_dim, emb_dim, bias=bias)
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(att_dropout)
        self.act = F.gelu   #F.relu  nn.Tanh()
        self.outlinear = nn.Linear(emb_dim, emb_dim)
        self.scale = emb_dim ** -0.5

    def forward(self, emb_txt, emb_Im, pad_mask=None):
        emb_im = self.inlinear(emb_Im)
        emb_im = self.act(emb_im)
        assert emb_im.size(-1) == emb_txt.size(-1)

        Q = self.Wq(emb_txt)
        K = self.Wk(emb_im)
        V = self.Wv(emb_im)

        #att_weights = torch.einsum('bid,bjd -> bij', Q, K) # [batch_size, seq_len_a, seq_len_b]
        att_weights = torch.matmul(Q, K.transpose(-2, -1))
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, float('-inf'))

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = self.dropout(att_weights)

        # out = torch.einsum('bij, bjd -> bid', att_weights, V)
        out = torch.matmul(att_weights, V)
        out = self.act(out)
        out = self.outlinear(out)
        return out



class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True):
        super(MLP, self).__init__()
        act = nn.GELU() #nn.Tanh
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(nn.Dropout(p=0.5))
                layers.append(act)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Linear_proj(nn.Module):
    def __init__(self, input_hidden_size, hidden_size, mlp_depth=2):
        super(Linear_proj, self).__init__()
        modules = [nn.Linear(input_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.projector = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)