# MiMo-V2.6 from scratch

Seven short PyTorch programs (about 700 lines with the shared files and the checks) that build, at toy size, the
ideas inside Xiaomi's MiMo-V2.6 language model and the way it was trained. They produce every toy number in the
video "Xiaomi MiMo-V2.6 From Scratch".

This is not MiMo and it reproduces none of Xiaomi's results. The real model has 309 billion weights. The largest
model here has 207 thousand and trains on a laptop CPU.

## Run it

```bash
pip install torch
bash run_all.sh
```

Python 3.9+. `demo-output.txt` is one saved run. Steps 1 to 6 take a minute or two each. Step 7 trains nine
agents and takes about 15 minutes.

## The seven steps

| file | idea | result of the saved run |
| --- | --- | --- |
| `step1_full.py` | a 4-layer language model where every token looks at every earlier token (global attention) | digits 100 %, name 100 %, 8,320 token pairs per line |
| `step2_window.py` | every token looks only at its last 8 tokens (sliding-window attention) | 1,936 pairs; digits 100 %, name 12.3 % (guessing is 12.5 %) |
| `step3_hybrid.py` | first layer global, the rest windows, as in MiMo | `GWWW`: 3,532 pairs, name 100 %. `GWWG`: 5,128 pairs, name 100 % |
| `step4_experts.py` | sparse mixture of experts: store 16 small networks per layer, run the best 2 per token | 65,536 weights stored, 8,192 used per token; accuracy unchanged |
| `step5_draft.py` | draft and verify: a small drafter writes 7 tokens in one pass, the big model checks them in one pass | 61 tokens in 9.0 passes of the big model, not 61; the same text in 200 of 200 lines |
| `step6_grpo.py` | GRPO with a pass/fail test: 16 tries per task, credit = reward − group mean | pass rate 17.0 % → 90.8 %; wasted moves stay at 4.3 |
| `step7_redistribute.py` | Xiaomi's method: grade the passing tries, shrink the credit of sloppy ones, hand the removed credit back to the passing tries | wasted moves 4.31 → 0.06 at about the same pass rate |

Shared files: `task.py` (the practice text), `model.py` (the model, the attention masks, the experts), `lm.py`
(training and measuring), `errand.py` (the agent's task), `grpo.py` (the three credit rules), `tests.py` (checks).

## The practice text

One line is 64 tokens: a name (one of 8 letters), a step (+1, +2 or +3), 60 digits that count up by that step,
a question mark, and the name again.

```
E +1 7 8 9 0 1 2 3 ... 5 6 ? E
```

The digits need only the last two tokens. The final name needs the first token, 62 places before the question
mark. Four window layers of 8 tokens reach 4 × 7 = 28 tokens back, so a windows-only model cannot know the name.
One global layer fixes it.

## The errand

The agent starts at 0 and gets a target between 10 and 45. It writes moves (+1, +2, +3, +5, −1, −2) and then
`done`. It passes if it says `done` exactly on the target. A wasted move is one more than the fewest possible.

Step 7 trains the same starting agent with three credit rules, three runs each. Means after 300 steps:

| rule | pass rate | wasted moves when it passes | tries that pass with no wasted move |
| --- | --- | --- | --- |
| pass/fail only | 91.1 % | 4.31 | 0.0 % |
| shrink only | 93.5 % | 0.45 | 62.7 % |
| shrink + hand back | 89.9 % | 0.06 | 84.8 % |

Three runs are too few to separate "shrink only" from "shrink + hand back", and the pass rates do not differ.
What the runs do show: a pass/fail test alone never removed the wasted moves, and grading the passing tries did.

## The exercise from the end of the video

In `step2_window.py`, change the window from 8 to 16:

```python
model, secs = train('WWWW', window=16)
```

With four layers, how far back can a token see now? Is that enough to reach the name, 62 tokens before the
question mark? Run it. Then add a fifth layer: `train('WWWWW', window=16)`.

<details>
<summary>My result (run yours first)</summary>

A window of 16 lets one layer look 15 tokens back. Four layers reach 4 × 15 = 60 tokens. The name is 62 back,
so it is still out of reach: name 12.3 %, the same as guessing (12.5 %).

Five layers reach 5 × 15 = 75 tokens, which is enough on paper. After 1,500 training steps the name was still
at 12.3 %. Reach is necessary, but it is not sufficient: the name has to be passed along five times in a row,
and this small model did not learn to do that in the steps I gave it. One global layer (`step3_hybrid.py`)
solves it directly.

</details>

## What is different from the real thing

- Xiaomi's grader is a language-model agent that reads 16 code solutions and scores five criteria. The grader
  here is three lines of arithmetic (`grade` in `grpo.py`). The tier factors and the 1.5 cap are the paper's.
- In Xiaomi's ablation, shrinking without handing back made training unstable (entropy 0.359 → 0.905, rollouts
  47.1k → 114.1k tokens). My toy stayed stable with shrinking only, so it does not show that failure.
- The toy drafter is one layer. Its saving in passes says nothing about MiMo's speed.
- Not built: the image, audio and video encoders, the training before RL, distillation, the tokenizer.

## Sources

- Model card and `config.json` of `XiaomiMiMo/MiMo-V2.6-Flash-RL` on Hugging Face (MIT license), read 2026-10-07.
- arXiv 2609.32577, "Groupwise Agentic Grading and Advantage Redistribution for Code Agent RL", 26 Sep 2026.
- I did not read Xiaomi's separate technical report.
