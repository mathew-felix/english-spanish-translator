# Reproducing Results

This project reports a custom Transformer run with a final test sacreBLEU of
31.41 on the English-Spanish corpus mix documented in `doc/PROJECT_REPORT.md`.

## Environment

- Python: 3.12
- Primary training hardware used for the reported run: NVIDIA RTX PRO 6000
  Blackwell Server Edition
- Random seed: `42` in `source/Config.py`
- W&B project: `english-spanish-translator`
- Model release tag for inference artifacts: `eng-sp-tranlate`

## Steps

1. Install dependencies.

   ```bash
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

2. Download the OPUS corpora.

   ```bash
   .venv\Scripts\python -m src.run --step download
   ```

3. Build the merged dataset and train/test split.

   ```bash
   .venv\Scripts\python -m src.run --step preprocess
   ```

4. Train the Transformer.

   ```bash
   set WANDB_MODE=offline
   .venv\Scripts\python -m src.run --step train
   ```

5. Evaluate the best checkpoint.

   ```bash
   .venv\Scripts\python -m src.run --step evaluate
   ```

## Expected Outputs

- `data/english_spanish.csv`: merged bilingual corpus.
- `data/train.csv` and `data/test.csv`: deterministic split generated with seed
  `42`.
- `best_model.pth`: best validation checkpoint.
- `data/tokenizer/`: tokenizer saved by training.
- Expected reported benchmark: `31.41` sacreBLEU on the full held-out test set.

## Notes

- The raw and processed data files are intentionally ignored by git because they
  are large generated artifacts.
- The Colab notebook is kept with outputs cleared. Re-run it in Colab if you need
  notebook-based training evidence.
- If online W&B logging is needed, set `WANDB_API_KEY` outside the repository.
