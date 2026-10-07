"""A toy agent task. The agent stands at 0 on a number line and must stop exactly on a target.
A test checks only one thing: did it stop on the target? It does not look at how many moves it took."""
import random

import torch
import torch.nn.functional as F

from model import TinyLM, fit

TARGETS = list(range(10, 46))              # 36 possible targets: 10 .. 45
MOVES = [1, 2, 3, 5, -1, -2]
T0, A0, DONE, PAD = 0, 36, 42, 43          # tokens 0-35 targets, 36-41 moves (+1 +2 +3 +5 -1 -2), 42 "done", 43 padding
VOCAB, MAXMOVES = 44, 20
LENGTH = 1 + MAXMOVES
WORDS = [f'go{t}' for t in TARGETS] + ['+1', '+2', '+3', '+5', '-1', '-2', 'done', '.']
BACK = torch.tensor([m < 0 for m in MOVES] + [False, False])    # a backwards move is the "defect" the grader looks for

FEWEST = [0]                               # FEWEST[t] = the fewest moves that can reach t
for t in range(1, TARGETS[-1] + 1):
    FEWEST.append(1 + min(FEWEST[t - m] for m in MOVES if 0 < m <= t))


def least(target):
    return FEWEST[target]


def demo(target, rng):
    """One messy example: wander with random moves until the target is hit, then say done."""
    while True:
        pos, seq = 0, []
        while pos != target and len(seq) < MAXMOVES:
            a = rng.choices(range(6), [0.2, 0.2, 0.2, 0.15, 0.15, 0.1])[0]
            if pos + MOVES[a] >= 0:
                seq.append(A0 + a)
                pos += MOVES[a]
        if pos == target and len(seq) < MAXMOVES:
            return [T0 + target - TARGETS[0]] + seq + [DONE] + [PAD] * (MAXMOVES - len(seq) - 1)


def start_model(seed=0, examples=4000, steps=300):
    """The starting agent: it has copied messy examples for a short time. It passes sometimes, and wastes moves."""
    rng = random.Random(seed)
    data = torch.tensor([demo(rng.choice(TARGETS), rng) for _ in range(examples)])
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TinyLM(VOCAB, LENGTH, 'GG')
    return fit(model, lambda step: data[torch.randint(examples, (64,), generator=gen)], steps, lr=1e-3)


@torch.no_grad()
def rollouts(model, targets, gen):
    """Let the agent act once for every target in the list. Returns the token rows and what happened."""
    n = len(targets)
    seq = (targets - TARGETS[0] + T0)[:, None]
    pos, moves, back = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.bool)
    done = torch.zeros(n, dtype=torch.bool)
    step = torch.tensor(MOVES + [0, 0])
    for _ in range(MAXMOVES):
        logits = model(seq)[:, -1]
        logits[:, :A0], logits[:, PAD] = -1e9, -1e9              # the agent may only write a move or "done"
        a = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)[:, 0]
        a = torch.where(done, torch.tensor(PAD), a)
        live = ~done & (a != DONE)
        kind = (a - A0).clamp(0, 7)
        pos = pos + torch.where(live, step[kind], torch.tensor(0))
        moves, back = moves + live, back | (live & BACK[kind])
        done = done | (a == DONE)
        seq = torch.cat([seq, a[:, None]], 1)
    passed = done & (pos == targets)                             # the test: stopped, and on the target
    return seq, {'passed': passed, 'moves': moves, 'back': back, 'wasted': moves - torch.tensor(FEWEST)[targets]}


def token_logp(model, seq):
    """The log-chance the agent gave to each token it wrote, and which places it wrote (not padding)."""
    logits = model(seq[:, :-1])
    logits[:, :, :A0], logits[:, :, PAD] = -1e9, -1e9
    logp = F.log_softmax(logits, -1)
    wrote = seq[:, 1:] != PAD
    took = logp.gather(-1, seq[:, 1:].clamp(max=DONE)[..., None])[..., 0]
    entropy = -(logp.exp() * logp.clamp(min=-50)).sum(-1)
    return took, wrote, entropy


def check(model, seed=99, each=100):
    """Measure the agent on 100 tries of every target (3,600 tries)."""
    targets = torch.tensor(TARGETS).repeat_interleave(each)
    seq, info = rollouts(model, targets, torch.Generator().manual_seed(seed))
    with torch.no_grad():
        _, wrote, entropy = token_logp(model, seq)
    ok = info['passed']
    return {'pass': float(ok.float().mean()), 'moves': float(info['moves'][ok].float().mean()),
            'wasted': float(info['wasted'][ok].float().mean()), 'back': float(info['back'][ok].float().mean()),
            'clean': float((ok & (info['wasted'] == 0)).float().mean()),   # share of ALL tries that pass with no wasted move
            'entropy': float(entropy[wrote].mean()), 'length': float(wrote.sum(1).float().mean())}


def show(row):
    return ' '.join(WORDS[t] for t in row if t != PAD)
