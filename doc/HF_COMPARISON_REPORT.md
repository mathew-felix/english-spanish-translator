# Hugging Face Comparison Report

Updated: 2026-04-09

## Overview

This report documents the comparison between:

- the custom English-to-Spanish Transformer trained in this repository
- the pretrained Hugging Face baseline `Helsinki-NLP/opus-mt-en-es`

The original comparison used dataset-derived rows. That turned out to be a weak benchmark for two reasons:

- the early slice was too domain-biased
- some corpus reference pairs were noisy or misaligned

The preferred benchmark is now a hand-written 50-row file with clean references:

- `finetune/manual_comparison_test_set.csv`

## Files

Implemented comparison files:

- `finetune/baseline_hf.py`
- `finetune/manual_comparison_test_set.csv`
- `finetune/custom_model_results_manual.json`
- `finetune/baseline_results_manual.json`

## Manual Benchmark Setup

Command used:

```bash
venv/bin/python finetune/baseline_hf.py \
  --csv-path finetune/manual_comparison_test_set.csv \
  --limit 50 \
  --custom-output custom_model_results_manual.json \
  --baseline-output baseline_results_manual.json
```

Models compared:

- Custom model: local checkpoint loaded through `source/inference.py`
- Baseline: `Helsinki-NLP/opus-mt-en-es` via MarianMT

Manual 50-row domain mix:

| Domain | Rows |
| --- | --- |
| Daily | 5 |
| Travel | 5 |
| Health | 5 |
| Work | 5 |
| Education | 5 |
| Emergency | 5 |
| Shopping | 5 |
| Social | 5 |
| Technology | 5 |
| Home | 5 |
| Total | 50 |

Selection rules:

- all English/Spanish pairs written by hand for this benchmark
- mixed everyday domains instead of one corpus style
- same exact source sentences used for both models
- references intended to be clean and semantically aligned

## Summary Results

| Item | Custom Transformer | MarianMT Baseline |
| --- | --- | --- |
| Result file | `finetune/custom_model_results_manual.json` | `finetune/baseline_results_manual.json` |
| Sentences compared | `50` | `50` |
| Average latency | `6518.67 ms` | `470.43 ms` |
| Exact string match to reference | `11 / 50` | `20 / 50` |

Additional comparison fact:

- both models produced the same output string on `21 / 50` rows

Important note:

- exact string match is not a strong translation metric by itself
- many valid translations are paraphrases rather than literal matches

## What The Manual Comparison Showed

### 1. The manual set is a cleaner benchmark

The dataset-derived comparisons were useful exploration, but they introduced noise from source corpora and benchmark selection.

The manual set is better because:

- each English/Spanish pair was authored directly for this benchmark
- the references are not inherited from noisy corpus alignments
- the prompts cover everyday domains you would actually demo
- both models face the same clean inputs

### 2. MarianMT is stronger on the manual set

On the manual benchmark, MarianMT behaves like the stronger baseline you would expect:

- better lexical choice on technology and shopping phrases
- fewer obviously wrong verb choices
- more stable wording on household and practical prompts

### 3. The custom model still proves the architecture works

The custom model still produced grammatically valid Spanish on many rows.

That means:

- the training pipeline is real
- the architecture learned meaningful translation behavior
- the project is not relying on placeholder outputs

### 4. On this measured CPU run, MarianMT was faster

Measured average latency:

- custom model: `6518.67 ms`
- MarianMT: `470.43 ms`

So the defensible statement is:

- in this local CPU comparison, MarianMT was faster

Not defensible:

- claiming the custom model is faster based on the measured manual benchmark

### 5. The quality gap is still best explained by data scale and baseline maturity

The comparison supports this interpretation:

- MarianMT benefits from large-scale pretrained translation data
- MarianMT also benefits from a mature pretrained optimization path
- the custom Transformer still demonstrates valid encoder-decoder learning
- the main gap is data scale and model maturity, not the impossibility of the architecture

## Side-By-Side Examples

These are real outputs from the manual comparison JSON files.

| Domain | English | Reference | Custom Transformer | MarianMT |
| --- | --- | --- | --- | --- |
| Daily | Good morning, did you sleep well? | Buenos días, ¿dormiste bien? | Buenos días, ¿durmió bien? | Buenos días, ¿durmieron bien? |
| Travel | Where can I buy a train ticket to Madrid? | ¿Dónde puedo comprar un billete de tren a Madrid? | ¿Dónde puedo comprar un billete de tren a Madrid? | ¿Dónde puedo comprar un billete de tren a Madrid? |
| Health | I have had a headache since early this morning. | Tengo dolor de cabeza desde temprano esta mañana. | Tengo dolor de cabeza desde esta mañana. | He tenido dolor de cabeza desde temprano esta mañana. |
| Education | The professor explained the problem step by step. | El profesor explicó el problema paso a paso. | El profesor explicó el problema paso a paso. | El profesor explicó el problema paso a paso. |
| Emergency | We need an ambulance right away. | Necesitamos una ambulancia de inmediato. | Necesitamos una ambulancia enseguida. | Necesitamos una ambulancia de inmediato. |
| Shopping | Can I pay by card, or do you only accept cash? | ¿Puedo pagar con tarjeta o solo aceptan efectivo? | ¿Puedo pagar con tarjeta o sólo aceptar dinero? | ¿Puedo pagar con tarjeta, o solo aceptas efectivo? |
| Social | We laughed so hard that we cried. | Nos reímos tanto que lloramos. | Nos reímos tanto que lloramos. | Nos reímos tanto que lloramos. |
| Technology | Did you remember to back up the files? | ¿Te acordaste de hacer una copia de seguridad de los archivos? | ¿Recuerdas retrasar los archivos? | ¿Te acordaste de hacer copias de seguridad de los archivos? |
| Technology | I need to reset my password again. | Necesito restablecer mi contraseña otra vez. | Necesito reanudar mi contraseña otra vez. | Necesito restablecer mi contraseña de nuevo. |
| Home | The washing machine stopped working this morning. | La lavadora dejó de funcionar esta mañana. | La lavadora dejó de trabajar esta mañana. | La lavadora dejó de funcionar esta mañana. |

## Honest Interpretation

The strongest honest interview takeaway is:

> On a clean hand-written 50-sentence benchmark, MarianMT is the stronger and faster baseline, while the custom Transformer still produces valid Spanish often enough to prove the architecture and training pipeline work.

What this supports:

- you understand the difference between training from scratch and transfer learning
- you know how benchmark selection can distort conclusions
- you can explain why pretrained data scale matters
- you can evaluate your own system without overselling it

What it does not support:

- claiming the baseline is weak
- claiming the custom model beats MarianMT
- claiming the custom model is faster on the measured benchmark

## Bottom Line

The manual comparison is the one you should use in interviews and documentation.

It shows:

- the custom model is real and functional
- MarianMT is the stronger pretrained baseline
- the earlier dataset-derived comparisons were weaker benchmarks
- your evaluation story improved once the benchmark design improved
