"""Step 7. The new part of MiMo-V2.6's training. A grader ranks the passing tries, good ones keep their credit,
worse ones lose some, and the lost credit is handed back to the passing group. Three ways to train, same start.
Nine training runs, three at a time: about 12 minutes on a laptop."""
from multiprocessing import Pool

import torch

from errand import LENGTH, VOCAB, least, rollouts, show
from grpo import CAP, G, binary, downweight, factors, grade, redistribute, train
from lm import save
from model import TinyLM

ARMS = {'pass/fail only': binary, 'shrink only': downweight, 'shrink + hand back': redistribute}
SEEDS, EVERY = 3, 25
KEYS = ('pass', 'moves', 'wasted', 'back', 'clean', 'entropy', 'length')


def start():
    model = TinyLM(VOCAB, LENGTH, 'GG')
    model.load_state_dict(torch.load('runs/start.pt'))
    return model


def run(job):
    name, seed = job
    torch.set_num_threads(2)
    return name, seed, train(start(), shape=ARMS[name], seed=seed, every=EVERY)


if __name__ == '__main__':
    # one group of 16 tries, worked by hand (target 11, picked because its passing tries cover all three tiers)
    target = 11
    seq, info = rollouts(start(), torch.full((G,), target), torch.Generator().manual_seed(18))
    info = {k: v[None] for k, v in info.items()}
    R = info['passed'].float()
    A, f, tier = binary(R), factors(R, info), grade(info)[0]
    down, full = downweight(R, info), redistribute(R, info)
    ok = R == 1
    lam = float((A * ok).sum() / (f * A * ok).sum())
    order = sorted(range(G), key=lambda i: (-R[0, i], tier[0, i], info['moves'][0, i]))
    print(f'one group: target {target}, fewest moves possible {least(target)}, {int(ok.sum())} of {G} tries pass')
    print('  try                                             test   tier  keeps   pass/fail   shrink only   shrink + hand back')
    group = []
    for i in order:
        p = bool(R[0, i])
        print(f'  {show(seq[i].tolist()[1:]):47s} {"pass" if p else "fail"}   {int(tier[0, i]) if p else "-":>3}   '
              f'{f"{f[0, i]:.2f}" if p else "  - "}   {A[0, i]:+8.4f}   {down[0, i]:+10.4f}   {full[0, i]:+14.4f}')
        group.append({'moves': show(seq[i].tolist()[1:]), 'pass': p, 'tier': int(tier[0, i]), 'f': float(f[0, i]),
                      'A': float(A[0, i]), 'down': float(down[0, i]), 'full': float(full[0, i])})
    print(f'  sum of the column                                                    {A.sum():+8.4f}   {down.sum():+10.4f}   {full.sum():+14.4f}')
    print(f'  credit of the passing tries                                          {(A * ok).sum():+8.4f}   {(down * ok).sum():+10.4f}   {(full * ok).sum():+14.4f}')
    print(f'  hand-back scale = {(A * ok).sum():.4f} / {(f * A * ok).sum():.4f} = {lam:.3f}   (the cap is {CAP})')
    out = {'group': group, 'target': target, 'least': least(target), 'passes': int(ok.sum()), 'scale': lam,
           'sums': [float(A.sum()), float(down.sum()), float(full.sum())],
           'pass_credit': [float((A * ok).sum()), float((down * ok).sum()), float((full * ok).sum())], 'arms': {}}

    with Pool(3) as pool:
        done = pool.map(run, [(n, s) for n in ARMS for s in range(SEEDS)])
    print(f'\n300 training steps from the same starting agent, {SEEDS} runs each')
    print('                              pass    moves   wasted   stepped back   pass + no waste   entropy   moves written')
    for name in ARMS:
        runs = [h for n, s, h in done if n == name]
        mean = [{'iter': runs[0][i]['iter'], **{k: sum(r[i][k] for r in runs) / SEEDS for k in KEYS}} for i in range(len(runs[0]))]
        for s, r in enumerate(runs):
            e = r[-1]
            print(f'  {name:19s} run {s + 1}  {e["pass"]:6.1%}  {e["moves"]:6.2f}  {e["wasted"]:7.2f}  {e["back"]:11.1%}  '
                  f'{e["clean"]:15.1%}  {e["entropy"]:8.3f}  {e["length"]:10.2f}')
        e = mean[-1]
        print(f'  {name:19s} mean   {e["pass"]:6.1%}  {e["moves"]:6.2f}  {e["wasted"]:7.2f}  {e["back"]:11.1%}  '
              f'{e["clean"]:15.1%}  {e["entropy"]:8.3f}  {e["length"]:10.2f}')
        out['arms'][name] = {'mean': mean, 'runs': runs}
    print(f'\npass rate and wasted moves along the way (mean of {SEEDS} runs):')
    print('  step   ' + '   '.join(f'{n:>22s}' for n in ARMS))
    for i in range(len(out['arms']['pass/fail only']['mean'])):
        row = [out['arms'][n]['mean'][i] for n in ARMS]
        print(f'  {int(row[0]["iter"]):4d}   ' + '   '.join(f'{r["pass"]:10.1%} {r["wasted"]:10.2f} ' for r in row))
    save('step7', out)
