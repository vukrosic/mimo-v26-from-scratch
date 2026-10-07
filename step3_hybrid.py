"""Step 3. Mix the two: most layers use the cheap window, a few layers are global. This is MiMo's attention."""
import torch

from lm import report, save, train

out = {}
for pattern in ('GWWW', 'GWWG'):
    model, secs = train(pattern)
    out[pattern] = report(f'pattern {pattern}', pattern, model, secs)
    torch.save(model.state_dict(), f'runs/{pattern}.pt')


def cost(n, w, glob, win):
    g = n * (n + 1) // 2                      # a global layer: token i looks at i + 1 tokens
    s = n * w - w * (w - 1) // 2              # a window layer: at most w tokens each
    return g, s, glob * g + win * s


n, w = 1_000_000, 128
g, s, mimo = cost(n, w, 9, 39)
_, _, full = cost(n, w, 48, 0)
print(f'\nMiMo-V2.6-Flash at {n:,} tokens, window {w}:')
print(f'  one global layer   {g:,} pairs')
print(f'  one window layer   {s:,} pairs   ({g / s:,.0f} times fewer)')
print(f'  48 global layers   {full:,}')
print(f'  9 global + 39 win  {mimo:,}   ({full / mimo:.2f} times fewer)')
print(f'  tokens a layer must keep in memory: global {n:,}, window {w}')
out['mimo'] = {'n': n, 'window': w, 'global_layer': g, 'window_layer': s, 'all_global': full, 'hybrid': mimo,
               'layer_ratio': g / s, 'model_ratio': full / mimo}
save('step3', out)
