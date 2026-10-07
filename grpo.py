"""GRPO, stripped down. Run the agent 16 times on the same task, compare each try with the group average,
make better-than-average tries more likely. The same loop serves steps 6 and 7: only `shape` changes."""
import torch

from errand import TARGETS, check, rollouts, token_logp

Q, G = 64, 16                              # 64 tasks per step, 16 tries each.  MiMo-V2.6: 1,568 tasks x 16 tries


def binary(R, info=None):
    """Plain GRPO with a pass/fail test. Advantage = my reward - the group's average reward."""
    return R - R.mean(1, keepdim=True)


def train(model, shape=binary, iters=300, lr=3e-4, seed=0, every=10, log=None):
    gen = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.Adam(model.parameters(), lr)
    hist = []
    for it in range(iters + 1):
        if it % every == 0:
            row = {'iter': it, **check(model)}
            if hist and 'credit' in hist[-1]:
                row.update(credit=hist[-1]['credit'], loss=hist[-1]['loss'], mixed=hist[-1]['mixed'])
            hist.append(row)
            if log:
                log(row)
        if it == iters:
            break
        targets = torch.tensor(TARGETS)[torch.randint(len(TARGETS), (Q,), generator=gen)].repeat_interleave(G)
        seq, info = rollouts(model, targets, gen)
        R = info['passed'].float().view(Q, G)
        mixed = (R.mean(1) > 0) & (R.mean(1) < 1)              # keep groups with at least one pass and one fail
        adv = shape(R, {k: v.view(Q, G) for k, v in info.items()}) * mixed[:, None]
        took, wrote, _ = token_logp(model, seq)
        wrote = wrote & mixed.repeat_interleave(G)[:, None]
        loss = -(adv.flatten()[:, None] * took * wrote).sum() / wrote.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist[-1].setdefault('mixes', []).append(float(mixed.float().mean()))
        hist[-1].setdefault('credits', []).append(float(adv.sum(1)[mixed].mean()) if mixed.any() else 0.0)
        hist[-1].setdefault('losses', []).append(float(loss.detach()))
        hist[-1]['credit'] = sum(hist[-1]['credits']) / len(hist[-1]['credits'])
        hist[-1]['loss'] = sum(hist[-1]['losses']) / len(hist[-1]['losses'])
        hist[-1]['mixed'] = sum(hist[-1]['mixes']) / len(hist[-1]['mixes'])    # share of groups with a pass and a fail
    for h in hist:
        h.pop('credits', None), h.pop('losses', None), h.pop('mixes', None)
    return hist


def grade(info):
    """The toy grader. It looks at every passing try and gives it a tier: 1 = tidy, 2 = wasteful, 3 = has a defect.
    Xiaomi's grader is a language model that reads all 16 solutions and scores five criteria."""
    tier = torch.where(info['back'], 3, torch.where(info['wasted'] <= 1, 1, 2))
    return tier, info['moves']                                   # inside a tier, fewer moves ranks higher


F_RUNNER, F_MAX, F_MIN, F_LOW, CAP = 0.9, 0.85, 0.4, 0.2, 1.5   # the values in Xiaomi's paper


def factors(R, info):
    """A number between 0 and 1 for every passing try: how much of its credit it keeps."""
    tier, moves = grade(info)
    f = torch.ones_like(R)
    for g in range(len(R)):
        ok = R[g] == 1
        for t in (1, 2):
            ranks = sorted(set(moves[g][ok & (tier[g] == t)].tolist()))          # tied tries share a rank
            for k, m in enumerate(ranks):
                here = ok & (tier[g] == t) & (moves[g] == m)
                if t == 1:
                    f[g][here] = 1.0 if k == 0 else F_RUNNER
                else:
                    f[g][here] = F_MAX if len(ranks) == 1 else F_MAX - (F_MAX - F_MIN) * k / (len(ranks) - 1)
        f[g][ok & (tier[g] == 3)] = F_LOW
    return f


def downweight(R, info):
    """Shrink the credit of worse passing tries. Failing tries are untouched, so credit is removed and not given back."""
    A, f = binary(R), factors(R, info)
    return torch.where(R == 1, f * A, A)


def redistribute(R, info):
    """Shrink, then scale all passing tries back up so that they share the same total credit as before."""
    A, f = binary(R), factors(R, info)
    ok = R == 1
    total = (A * ok).sum(1, keepdim=True)                        # the credit passing tries had with pass/fail
    lam = total / (f * A * ok).sum(1, keepdim=True).clamp(min=1e-9)
    B = torch.where(ok, lam.clamp(max=CAP) * f * A, A)           # the scale-up is capped at 1.5
    return B - B.mean(1, keepdim=True)                           # if the cap was hit, re-center the group on zero
