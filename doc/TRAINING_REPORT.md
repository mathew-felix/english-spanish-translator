# Training Report

Updated: 2026-04-09

## Overview

This report documents the completed full training run captured in `output.txt`.

The run covered:

- OPUS dataset download
- preprocessing and split generation
- 30-epoch training
- W&B tracking
- full held-out evaluation with `sacrebleu`

## Hardware And Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Reported VRAM | 97,887 MiB |
| Torch | `2.10.0+cu128` |
| CUDA available | `True` |

## Training Configuration

| Hyperparameter | Value |
| --- | --- |
| Epochs | 30 |
| Batch size | 640 |
| Max sequence length | 60 |
| Learning rate | `4.5e-4` |
| Patience | 5 |
| Warmup steps | 4000 |
| Beam width | 4 |
| BLEU eval batches | 5 |
| OpenSubtitles max rows | 2,000,000 |
| Num workers | 4 |

## Dataset Pipeline Result

The completed preprocessing run kept:

| Corpus | Kept pairs |
| --- | --- |
| Europarl | 1,940,734 |
| News-Commentary | 46,904 |
| TED2020 | 403,752 |
| OpenSubtitles | 2,000,000 |
| Total | 4,391,390 |

Final split used for training:

- train: `3,512,826`
- test: `878,564`

## Training Progress Summary

| Checkpoint | Train loss | Val loss | BLEU |
| --- | --- | --- | --- |
| Epoch 1 | `26.5318` | `4.2375` | `0.0831` |
| Epoch 15 | `2.7339` | `2.6032` | `0.1693` |
| Epoch 21 | `2.6248` | `2.5211` | `0.2579` |
| Epoch 25 | `2.6153` | `2.5099` | `0.2781` |
| Epoch 29 | `2.6080` | `2.5055` | `0.3167` |
| Epoch 30 | `2.6065` | `2.5060` | `0.2786` |

Important scheduler event:

- learning rate was halved at epoch 15 from `0.000450` to `0.000225`

Best checkpoint during training:

- epoch 29
- validation loss: `2.5055`

## Final Evaluation Result

The full held-out evaluation completed successfully.

Final result:

- `Average BLEU Score (sacrebleu): 31.41`
- `Final BLEU: 0.31406433640102166`
- evaluation wall time: `2:33:37`

This is the strongest quantitative result from the completed run.

## W&B Tracking

W&B logging worked successfully for this run.

Links:

- Project: https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator
- Run: https://wandb.ai/relixmatrix-texas-state-university/english-spanish-translator/runs/acxn0hti

W&B run name:

- `silvery-galaxy-1`

## Qualitative Samples From The Run

Late-epoch training samples showed the model learning common translation patterns:

- `How are you? -> ¿Cómo estás?`
- `Where is the hospital? -> ¿Dónde está el hospital?`
- `I need help with my homework. -> Necesito ayuda con mis deberes.`

These samples were observed multiple times during later epochs.

## Important Clarification

The last three example lines at the bottom of `output.txt` were identified by the user as stale output from a previous run.

They are not used in this report.

That means the verified result for this run is based on:

- the 30-epoch training logs
- the W&B run summary
- the final full evaluation block

## Local Reload Verification

After the Colab export artifacts were copied back into the project paths, the checkpoint was re-verified locally on 2026-04-09.

What was confirmed:

- `source/inference.py` can translate with the epoch 29 checkpoint
- `serve.py` returns a working `/health` and `/translate` response
- `source/Evaluate.py` now reloads the saved checkpoint config before building the evaluation model, so the trained `max_seq_length = 60` checkpoint loads correctly

## What This Run Proves

This run confirms that the current project can:

- download and preprocess the full OPUS corpus mix
- train the custom Transformer at large batch size on a high-memory GPU
- complete all 30 epochs without divergence
- log metrics to W&B
- produce a full held-out `sacreBLEU` result above 30

## What It Does Not Prove

This run does not by itself prove that the model is production-grade for arbitrary user prompts.

What it does prove is:

- the training system works
- the model learns useful translation behavior
- the repo has a credible end-to-end training result

## Bottom Line

The latest verified run is a successful large-scale training run on the new OPUS pipeline.

The headline result is:

> `31.41 sacreBLEU` on the held-out test split after a 30-epoch run with batch size `640` on an RTX PRO 6000 Blackwell GPU
