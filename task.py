"""The practice text. One line is 64 tokens: a name, a step, 60 digits that count up by that step, a question mark,
and the name again. The digits can be guessed from the last few tokens. The first name is 62 tokens before the question mark."""
import torch

NAMES, STEPS, DIGITS = 8, 3, 60
NAME0, STEP0, ASK = 10, 18, 21            # tokens 0-9 are digits, 10-17 names, 18-20 steps (+1, +2, +3), 21 is "?"
VOCAB, LENGTH = 22, 64
WORDS = [str(d) for d in range(10)] + list('ABCDEFGH') + ['+1', '+2', '+3', '?']


def lines(n, gen):
    """n random lines as one table of tokens."""
    name = torch.randint(NAMES, (n, 1), generator=gen)
    step = torch.randint(STEPS, (n, 1), generator=gen)
    first = torch.randint(10, (n, 1), generator=gen)
    digits = (first + torch.arange(DIGITS) * (step + 1)) % 10
    return torch.cat([name + NAME0, step + STEP0, digits, torch.full((n, 1), ASK), name + NAME0], 1)


def show(line):
    return ' '.join(WORDS[t] for t in line)


if __name__ == '__main__':
    for line in lines(3, torch.Generator().manual_seed(0)):
        print(show(line.tolist()))
