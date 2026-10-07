"""Step 1. A tiny GPT where every layer is global: each token looks at every earlier token."""
import torch

from lm import report, save, train
from task import lines, show

print('three lines of practice text:')
for line in lines(3, torch.Generator().manual_seed(0)):
    print(' ', show(line.tolist()))
model, secs = train('GGGG')
save('step1', report('\nglobal attention in all 4 layers', 'GGGG', model, secs))
torch.save(model.state_dict(), 'runs/full.pt')
