"""Step 6. Train the agent with GRPO and a pass/fail test. It learns to pass. Nothing tells it to be tidy."""
import random

import torch

from errand import check, demo, least, show, start_model
from grpo import train
from lm import save

rng = random.Random(0)
print('messy examples the agent starts from:')
for t in (12, 27, 41):
    print(f'   target {t} (fewest moves possible: {least(t)}):', show(demo(t, rng)[1:]))
model = start_model()
torch.save(model.state_dict(), 'runs/start.pt')
print('\nstep   pass   moves when it passes   wasted moves   stepped back   entropy   moves written   mixed groups')
line = lambda r: print(f'{r["iter"]:4d}  {r["pass"]:6.1%}  {r["moves"]:10.2f}  {r["wasted"]:18.2f}  {r["back"]:12.1%}  '
                       f'{r["entropy"]:8.3f}  {r["length"]:10.2f}  {r.get("mixed", 0):12.1%}')
hist = train(model, log=line, every=25)
save('step6', hist)
