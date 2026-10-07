"""Step 5. Draft and verify. A small drafter writes 7 tokens in one pass. The big model checks all 7 in one pass and
keeps the ones it would have written itself. The text is the same as writing one token at a time, with fewer passes."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lm import save
from model import WIDTH, Block, TinyLM
from task import LENGTH, VOCAB, lines

BLOCK = 8                                    # 1 token already written (the anchor) + 7 drafted
MASK = VOCAB                                 # the "blank" token the drafter fills in


class Drafter(nn.Module):
    """One layer. It is not causal: all 8 slots see each other, so all 7 blanks are filled in one pass."""

    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB + 1, WIDTH), nn.Embedding(LENGTH + BLOCK, WIDTH)
        self.read = nn.Linear(WIDTH, WIDTH)  # reads the big model's hidden state
        self.block = Block('G', None, False, causal=False)
        self.norm, self.out = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, VOCAB)

    def forward(self, hidden, anchor, where):
        """hidden: the big model's state just before the anchor. anchor: the last written token. where: its position."""
        slots = torch.cat([anchor[:, None], torch.full((len(anchor), BLOCK - 1), MASK)], 1)
        x = self.tok(slots) + self.pos(where[:, None] + torch.arange(BLOCK)) + self.read(hidden)[:, None]
        return self.out(self.norm(self.block(x)))[:, 1:]          # a guess for each of the 7 blanks


def train_drafter(big, steps=1500, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 5)
    drafter = Drafter()
    opt = torch.optim.AdamW(drafter.parameters(), 2e-3)
    for _ in range(steps):
        seq = lines(64, gen)
        with torch.no_grad():
            hidden = big(seq[:, :-1], hidden=True)[1]             # the big model is frozen
        i = torch.randint(LENGTH - 1, (len(seq), 8), generator=gen)            # 8 random places per line
        rows = torch.arange(len(seq))[:, None].expand_as(i)
        want = F.pad(seq, (0, BLOCK), value=-100)                 # past the end of the line there is nothing to learn
        want = torch.stack([want[rows, i + 2 + j] for j in range(BLOCK - 1)], -1)
        guess = drafter(hidden[rows, i].flatten(0, 1), seq[rows, i + 1].flatten(), (i + 1).flatten())
        loss = F.cross_entropy(guess.flatten(0, 1), want.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return drafter


@torch.no_grad()
def one_at_a_time(big, start, total=LENGTH):
    seq, passes = start.clone(), 0
    while len(seq) < total:
        seq = torch.cat([seq, big(seq[None])[0, -1].argmax()[None]])
        passes += 1
    return seq, passes


@torch.no_grad()
def draft_and_verify(big, drafter, start, total=LENGTH):
    score, hidden = big(start[None], hidden=True)                 # pass 1: the first new token
    seq, state, passes, kept = torch.cat([start, score[0, -1].argmax()[None]]), hidden[0, -1], 1, []
    while len(seq) < total:
        room = min(BLOCK - 1, total - len(seq))
        draft = drafter(state[None], seq[-1:], torch.tensor([len(seq) - 1]))[0].argmax(-1)[:room]
        score, hidden = big(torch.cat([seq, draft])[None], hidden=True)       # one pass checks every drafted token
        passes += 1
        own = score[0, len(seq) - 1:].argmax(-1)                  # what the big model would write at each place
        n = 0
        while n < room and draft[n] == own[n]:
            n += 1                                                # keep drafted tokens while they agree
        kept.append(n)
        state = hidden[0, len(seq) - 1 + n]
        seq = torch.cat([seq, draft[:n], own[n:n + 1]])[:total]   # then one token from the big model itself
    return seq, passes, kept


if __name__ == '__main__':
    big = TinyLM(VOCAB, LENGTH, 'GWWG')
    big.load_state_dict(torch.load('runs/GWWG.pt'))
    big.eval()
    torch.manual_seed(0)
    arms = {'drafter with random weights': Drafter(), 'trained drafter': train_drafter(big)}
    tests = lines(200, torch.Generator().manual_seed(2))
    out = {'tokens': LENGTH - 3}
    base = [one_at_a_time(big, line[:3]) for line in tests]
    print(f'write {LENGTH - 3} tokens after a 3-token start, 200 lines')
    print(f'  one token at a time           {sum(p for _, p in base) / 200:5.1f} passes of the big model')
    for name, drafter in arms.items():
        runs = [draft_and_verify(big, drafter, line[:3]) for line in tests]
        same = sum(bool((a == b).all()) for (a, _), (b, _, _) in zip(base, runs))
        passes = sum(p for _, p, _ in runs) / 200
        kept = [k for _, _, ks in runs for k in ks]
        print(f'  {name:29s} {passes:5.1f} passes   same text in {same} of 200   '
              f'drafted tokens kept {sum(kept) / len(kept):.2f} of 7 per pass')
        out['trained' if name.startswith('trained') else 'untrained'] = {'passes': passes, 'same': same, 'kept': sum(kept) / len(kept)}
    seq, passes, kept = draft_and_verify(big, arms['trained drafter'], tests[0][:3])
    print(f'  first line, drafted tokens kept in each pass: {kept}   ({passes} passes)')
    small = sum(p.numel() for p in arms['trained drafter'].parameters())
    large = sum(p.numel() for p in big.parameters())
    print(f'  weights: big model {large:,}, drafter {small:,}')
    out.update(first_kept=kept, first_passes=passes, big_weights=large, drafter_weights=small,
               base=sum(p for _, p in base) / 200)
    save('step5', out)
