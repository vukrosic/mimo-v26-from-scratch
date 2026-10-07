"""Step 4. Sparse experts: store 16 small networks per layer, run only the best 2 for each token."""
import torch

from lm import report, save, train
from model import Experts
from task import LENGTH, lines

model, secs = train('GWWG', experts=True)
out = report('pattern GWWG with sparse experts (16 stored, 2 used per token)', 'GWWG', model, secs)
torch.save(model.state_dict(), 'runs/experts.pt')

block = model.blocks[1]
one = sum(p.numel() for p in block.ff.nets[0].parameters())
stored, used = one * block.ff.n, one * block.ff.k
dense = sum(p.numel() for p in model.blocks[0].ff.parameters())
print(f'  one expert                {one:,} weights')
print(f'  stored in a layer         {stored:,}   (16 experts)')
print(f'  used for one token        {used:,}   (2 experts, {used / stored:.1%})')
print(f'  the dense layer it replaces has {dense:,} weights, all used for every token')

with torch.no_grad():
    seq = lines(500, torch.Generator().manual_seed(1))
    model(seq[:, :-1])
    chosen = block.ff.chosen                                   # (lines, positions, 2)
    share = torch.bincount(chosen.flatten(), minlength=16).float() / chosen.numel()
    digit = seq[:, :-1][:, 2:62]
    top = [int(torch.bincount(chosen[:, 2:62][digit == d].flatten(), minlength=16).argmax()) for d in range(10)]
print('  share of tokens sent to each expert: ' + ' '.join(f'{x:.0%}' for x in share.tolist()))
print('  busiest expert for digits 0..9:      ' + ' '.join(str(t) for t in top))

h, m, layers, n, k = 4096, 2048, 47, 256, 8
exp = 3 * h * m                                                # assumed: a gated feed-forward with 3 matrices
print(f'\nMiMo-V2.6-Flash, from its config (hidden {h}, expert width {m}, {n} experts, {k} used, {layers} sparse layers):')
print(f'  one expert       {exp:,} weights (if it has 3 matrices)')
print(f'  all experts      {exp * n * layers / 1e9:.1f} billion of the 309 billion')
print(f'  used per token   {exp * k * layers / 1e9:.1f} billion of the 15 billion active')
out.update(one=one, stored=stored, used=used, dense=dense, share=share.tolist(), top=top,
           mimo_expert=exp, mimo_all=exp * n * layers, mimo_used=exp * k * layers)
save('step4', out)
