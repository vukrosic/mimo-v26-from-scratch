"""A tiny language model in the shape of MiMo: every layer is either global attention or sliding-window attention,
and the feed-forward part is either one dense network or a set of sparse experts."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH, HEADS = 64, 4


def allowed(length, window=None):
    """A table: may token i look at token j? Global: every earlier token. Window: only the last `window` tokens."""
    i, j = torch.arange(length)[:, None], torch.arange(length)[None, :]
    ok = j <= i
    if window:
        ok = ok & (j > i - window)
    return ok


class Experts(nn.Module):
    """Sparse feed-forward. A router scores every expert for a token; only the best `k` run. MiMo: 256 and 8."""

    def __init__(self, n=16, k=2, hidden=32):
        super().__init__()
        self.n, self.k = n, k
        self.router = nn.Linear(WIDTH, n, bias=False)
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(WIDTH, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, WIDTH, bias=False))
            for _ in range(n))

    def forward(self, x):
        score = torch.sigmoid(self.router(x))                    # one score per expert, for every token
        best, self.chosen = score.topk(self.k, dim=-1)           # keep the k best experts
        weight = best / best.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, net in enumerate(self.nets):
            mine = self.chosen == e                              # which tokens picked expert e
            rows = mine.any(-1)
            if rows.any():                                       # the expert runs only on those tokens
                out[rows] += net(x[rows]) * (weight * mine).sum(-1)[rows][:, None]
        return out


class Block(nn.Module):
    def __init__(self, kind, window, experts, causal=True):
        super().__init__()
        self.window, self.causal = (window if kind == 'W' else None), causal
        self.n1, self.n2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.qkv, self.proj = nn.Linear(WIDTH, 3 * WIDTH), nn.Linear(WIDTH, WIDTH)
        self.ff = Experts() if experts else nn.Sequential(
            nn.Linear(WIDTH, 4 * WIDTH), nn.SiLU(), nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = (t.view(B, L, HEADS, -1).transpose(1, 2) for t in self.qkv(self.n1(x)).chunk(3, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed(L, self.window) if self.causal else None)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, WIDTH))
        return x + self.ff(self.n2(x))


class TinyLM(nn.Module):
    """pattern: one letter per layer, G = global attention, W = window attention. MiMo's first block is G and dense."""

    def __init__(self, vocab, length, pattern='GGGG', window=8, experts=False):
        super().__init__()
        self.pattern, self.window = pattern, window
        self.tok, self.pos = nn.Embedding(vocab, WIDTH), nn.Embedding(length, WIDTH)
        self.blocks = nn.ModuleList(Block(kind, window, experts and i > 0) for i, kind in enumerate(pattern))
        self.norm, self.out = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, vocab)

    def forward(self, seq, hidden=False):
        """seq: rows of tokens. Returns a score for every possible next token, at every position."""
        x = self.tok(seq) + self.pos(torch.arange(seq.shape[1]))
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return (self.out(x), x) if hidden else self.out(x)


def pairs(pattern, length, window):
    """How many (token, earlier token) pairs each layer compares. This is the cost of attention."""
    return [int(allowed(length, window if kind == 'W' else None).sum()) for kind in pattern]


def fit(model, batches, steps, lr=2e-3):
    """Plain next-token training: read a batch of lines, predict every next token, nudge the weights."""
    opt = torch.optim.AdamW(model.parameters(), lr)
    for step in range(steps):
        seq = batches(step)
        loss = F.cross_entropy(model(seq[:, :-1]).flatten(0, 1), seq[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
