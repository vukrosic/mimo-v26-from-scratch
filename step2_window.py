"""Step 2. Sliding-window attention: each token looks only at the last 8 tokens. Cheap, but short-sighted."""
from lm import report, save, train

model, secs = train('WWWW')
save('step2', report('window attention (8 tokens) in all 4 layers', 'WWWW', model, secs))
print('  4 layers of 8 tokens reach 4 x 7 = 28 tokens back. The name is 62 tokens before the question mark.')
