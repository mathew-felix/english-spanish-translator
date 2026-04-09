# Model Spot Check Report

## Scope

This report documents a manual translation spot check using the exported Colab run artifacts in `english_spanish_translator_20260409_155517/`.

Artifacts used:

- `english_spanish_translator_20260409_155517/best_model.pth`
- `english_spanish_translator_20260409_155517/tokenizer/`
- `english_spanish_translator_20260409_155517/test.csv`

Checkpoint metadata:

- Epoch: `29`
- Validation loss: `2.5055`
- Max sequence length: `60`
- Batch size: `640`
- Learning rate: `4.5e-4`

## Test Method

The model was loaded locally on CPU with the exported tokenizer and saved checkpoint config.

Two kinds of checks were run:

- user-style prompts to judge practical translation quality
- held-out `test.csv` rows to compare predictions against real references

## User Prompt Results

| Input | Output | Verdict |
|---|---|---|
| `How are you?` | `¿Cómo estás?` | Proper |
| `Where is the nearest hospital?` | `¿Dónde está el hospital más cercano?` | Proper |
| `I need help with my homework.` | `Necesito ayuda con mis deberes.` | Proper |
| `Please speak more slowly.` | `Por favor, hable más despacio.` | Proper |
| `My phone is not working.` | `Mi teléfono no funciona.` | Proper |

## Held-Out Test Row Results

### Row 0

Source:

`I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period.`

Reference:

`Declaro reanudado el período de sesiones del Parlamento Europeo, interrumpido el viernes 17 de diciembre pasado, y reitero a Sus Señorías mi deseo de que hayan tenido unas buenas vacaciones.`

Prediction after fallback:

`Declaro reanudado el período de sesiones del Parlamento Europeo, interrumpido el viernes 17 de diciembre de 1999, y quisiera desearle de nuevo un feliz Año Nuevo con la esperanza de que disfrutara de un agradable período festivo.`

Verdict:

Proper paraphrase. Longer than the reference, but meaning is preserved.

### Row 1

Source:

`Madam President, on a point of order.`

Reference:

`Señora Presidenta, una cuestión de procedimiento.`

Prediction:

`Señora Presidenta, una cuestión de orden.`

Verdict:

Proper. Slight wording difference, same meaning.

### Row 2

Source:

`One of the people assassinated very recently in Sri Lanka was Mr Kumar Ponnambalam, who had visited the European Parliament just a few months ago.`

Reference:

`Una de las personas que recientemente han asesinado en Sri Lanka ha sido al Sr. Kumar Ponnambalam, quien hace pocos meses visitó el Parlamento Europeo.`

Prediction:

`Una de las personas asesinadas recientemente en Sri Lanka fue el Sr. Kumar Ponnambalam, que había visitado el Parlamento Europeo hace unos meses.`

Verdict:

Proper paraphrase. Natural Spanish and correct meaning.

## Inference Edge Case Found

One real inference bug showed up during the spot check:

- the API-style decoding path uses `beam_width=4`
- on the long Row 0 sentence, `beam_width=4` returned an empty translation
- the same sentence with `beam_width=2` returned a valid translation

This means the problem was not that the model failed to learn the sentence. The failure was in decoding behavior for a specific long input.

## Fix Applied

`source/inference.py` now retries with `beam_width=2` when `beam_width=4` returns no tokens.

This is a minimal inference-layer fix:

- short prompts still use the existing wide-beam path when it works
- long edge cases no longer collapse to empty output
- the model weights and training pipeline were not changed

## Overall Verdict

The exported model is proper enough for demo use.

What it does well:

- short user prompts
- common travel/help phrases
- medium-length formal sentences
- Europarl-style held-out paraphrases

What is still true:

- it is not production-grade general translation
- decoding can still be sensitive on long inputs
- evaluation should still combine BLEU with human spot checks

## Conclusion

The Colab-trained exported checkpoint is legitimate and usable.

The translation quality is clearly real, not random or broken. The main issue found in this spot check was an inference decoding edge case, not a failed model.
