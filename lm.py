"""Shared by steps 1 to 5: train a model on the practice text and measure it."""
import json
import os
import time

import torch

from model import TinyLM, fit, pairs
from task import LENGTH, VOCAB, lines

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS, BATCH = 1500, 64


def train(pattern, experts=False, seed=0, window=8):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TinyLM(VOCAB, LENGTH, pattern, window, experts)
    t = time.time()
    fit(model, lambda step: lines(BATCH, gen), STEPS)
    return model, time.time() - t


@torch.no_grad()
def measure(model, n=2000, seed=1):
    """Two scores on fresh lines: the digits (a nearby pattern) and the final name (written 62 tokens after it was last seen)."""
    seq = lines(n, torch.Generator().manual_seed(seed))
    guess = model(seq[:, :-1]).argmax(-1)
    right = guess == seq[:, 1:]
    return {'digits': float(right[:, 3:61].float().mean()), 'name': float(right[:, -1].float().mean())}


def report(title, pattern, model, secs, window=8):
    m, p = measure(model), pairs(pattern, LENGTH, window)
    print(title)
    print(f'  layers            {" ".join(pattern)}')
    print(f'  pairs per layer   {" ".join(str(x) for x in p)}   total {sum(p):,}')
    print(f'  digits right      {m["digits"]:.1%}')
    print(f'  final name right  {m["name"]:.1%}   (guessing gives 12.5%)')
    return {'pattern': pattern, 'pairs': p, 'total_pairs': sum(p), **m, 'seconds': secs}


def save(name, data):
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    with open(os.path.join(HERE, 'runs', name + '.json'), 'w') as f:
        json.dump(data, f, indent=1)


def load(name):
    with open(os.path.join(HERE, 'runs', name + '.json')) as f:
        return json.load(f)
