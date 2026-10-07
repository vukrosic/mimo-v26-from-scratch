"""Quick checks of the parts. Run after run_all.sh (two checks load the trained models). Takes a few seconds."""
import os

import torch

from errand import TARGETS, least, rollouts, start_model
from grpo import CAP, G, binary, downweight, factors, redistribute
from model import TinyLM, allowed, pairs
from task import LENGTH, VOCAB, lines

# the task: a line is 64 tokens and ends with the name it started with
seq = lines(50, torch.Generator().manual_seed(0))
assert seq.shape == (50, LENGTH) and (seq[:, 0] == seq[:, -1]).all()

# the masks: a token never sees the future, and a window layer sees only the last 8
full, win = allowed(LENGTH), allowed(LENGTH, 8)
assert int(full.sum()) == LENGTH * (LENGTH + 1) // 2 == 2080
assert int(win.sum()) == 484 and not win[20, 12] and win[20, 13] and not win[20, 21]
assert pairs('GWWG', LENGTH, 8) == [2080, 484, 484, 2080]

# the errand: fewest moves with +1 +2 +3 +5
assert [least(t) for t in (12, 27, 41)] == [3, 6, 9]

# the three ways to hand out credit, on 8 random groups
torch.manual_seed(0)
agent = TinyLM(44, 21, 'GG')
targets = torch.tensor(TARGETS)[:8].repeat_interleave(G)
_, info = rollouts(agent, targets, torch.Generator().manual_seed(0))
info = {k: v.view(8, G) for k, v in info.items()}
R = info['passed'].float()
R[:, 0], R[:, 1] = 1, 0                                    # make sure every group has a pass and a fail
A, D, H, f = binary(R), downweight(R, info), redistribute(R, info), factors(R, info)
ok = R == 1
assert A.sum(1).abs().max() < 1e-5                          # pass/fail credit sums to zero in every group
assert H.sum(1).abs().max() < 1e-5                          # so does shrink + hand back
assert (D[ok] <= A[ok] + 1e-6).all() and (D[~ok] == A[~ok]).all()   # shrinking only ever removes credit from passes
assert ((f >= 0.2) & (f <= 1)).all()
scale = (A * ok).sum(1) / (f * A * ok).sum(1)
free = scale <= CAP                                         # where the cap is not hit, passes keep their total credit
assert torch.allclose((H * ok).sum(1)[free], (A * ok).sum(1)[free], atol=1e-5)

# draft and verify must give exactly the text the big model writes alone, even with an untrained drafter
if os.path.exists('runs/GWWG.pt'):
    from step5_draft import Drafter, draft_and_verify, one_at_a_time
    big = TinyLM(VOCAB, LENGTH, 'GWWG')
    big.load_state_dict(torch.load('runs/GWWG.pt'))
    big.eval()
    torch.manual_seed(3)
    for line in lines(20, torch.Generator().manual_seed(7)):
        assert (one_at_a_time(big, line[:3])[0] == draft_and_verify(big, Drafter(), line[:3])[0]).all()

print('all checks pass')
