import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from source.Config import Config
from source.DatasetTranslation import TranslationDataset
from source.Model import Transformer

try:
    import sacrebleu as _sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    HAS_SACREBLEU = False
    print("Warning: sacrebleu not installed — falling back to nltk corpus_bleu.")


def generate_translations(transformer, encoder_inputs, config, max_seq_length=40):
    """
    Batch greedy/beam generation for evaluation.
    Returns list of decoded token ID lists, one per sentence in the batch.
    """
    device = encoder_inputs.device
    batch_size = encoder_inputs.size(0)
    decoded_tokens = torch.full(
        (batch_size, max_seq_length), config.pad_token_id, dtype=torch.long
    ).to(device)
    decoded_tokens[:, 0] = config.sos_token_id

    with torch.no_grad():
        for t in range(1, max_seq_length):
            outputs = transformer(encoder_inputs, decoded_tokens[:, :t])
            next_tokens = torch.argmax(outputs[:, -1, :], dim=-1)
            decoded_tokens[:, t] = next_tokens
            if torch.all(next_tokens == config.eos_token_id):
                break

    return decoded_tokens.cpu().numpy()


def decode_sentences(token_indices, tokenizer, special_token_ids):
    """
    FIX Bug #2: use tokenizer.decode() instead of vocab inversion.
    Eliminates WordPiece ##subword fragments from hypothesis text.
    special_token_ids: set of IDs to strip before decoding.
    """
    sentences = []
    for tokens in token_indices:
        filtered = [int(t) for t in tokens if int(t) not in special_token_ids]
        sentence = tokenizer.decode(filtered, skip_special_tokens=True)
        sentences.append(sentence)
    return sentences


def evaluate_model(transformer, test_dataloader, config, tokenizer, max_seq_length=40):
    """
    Improvement #7: use sacrebleu corpus_bleu (proper corpus-level BLEU).
    Falls back to nltk corpus_bleu if sacrebleu is unavailable.
    """
    transformer.to(config.device)
    transformer.eval()

    special_ids = {config.pad_token_id, config.sos_token_id, config.eos_token_id}
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
            encoder_inputs, _, targets = [x.to(config.device) for x in batch]
            predicted_tokens = generate_translations(
                transformer, encoder_inputs, config, max_seq_length
            )
            hyps = decode_sentences(predicted_tokens, tokenizer, special_ids)
            refs = decode_sentences(targets.cpu().numpy(), tokenizer, special_ids)
            all_hypotheses.extend(hyps)
            all_references.extend(refs)

    if HAS_SACREBLEU:
        result = _sacrebleu.corpus_bleu(all_hypotheses, [all_references])
        average_bleu = result.score / 100.0  # sacrebleu returns 0-100
        print(f"Average BLEU Score (sacrebleu): {result.score:.2f}")
    else:
        tokenised_hyps = [h.split() for h in all_hypotheses]
        tokenised_refs = [[r.split()] for r in all_references]
        average_bleu = corpus_bleu(
            tokenised_refs, tokenised_hyps,
            smoothing_function=SmoothingFunction().method1
        )
        print(f"Average BLEU Score (nltk corpus_bleu): {average_bleu:.4f}")

    # Individual sentence BLEUs for histogram
    if HAS_SACREBLEU:
        indiv = [
            _sacrebleu.sentence_bleu(h, [r]).score / 100.0
            for h, r in zip(all_hypotheses, all_references)
        ]
    else:
        from nltk.translate.bleu_score import sentence_bleu
        indiv = [
            sentence_bleu([r.split()], h.split(),
                          smoothing_function=SmoothingFunction().method1)
            for h, r in zip(all_hypotheses, all_references)
        ]

    plt.figure(figsize=(10, 6))
    plt.hist(indiv, bins=50, color="skyblue", edgecolor="black")
    plt.title("Distribution of BLEU Scores Across Test Set")
    plt.xlabel("BLEU Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("bleu_score_distribution.png")
    plt.close()

    return average_bleu


def load_model(config, model_path="best_model.pth"):
    """
    Improvement #11: load rich checkpoint (supports both new dict format and
    legacy bare state_dict for backward compatibility).
    """
    transformer = Transformer(config)
    print(f"Loading model from '{model_path}'...")
    checkpoint = torch.load(model_path, map_location=config.device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        transformer.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Saved at epoch {checkpoint.get('epoch', '?')} "
              f"— val_loss={checkpoint.get('val_loss', '?'):.4f}")
    else:
        transformer.load_state_dict(checkpoint)  # legacy bare state_dict
    transformer.to(config.device)
    transformer.eval()
    return transformer


def _resolve_model_path(config):
    """
    Resolve the checkpoint path used for evaluation.
    Falls back to the weights directory when the root path is absent.
    """
    candidates = [
        config.model_save_path,
        os.path.join("weights", os.path.basename(config.model_save_path)),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Model checkpoint not found. Checked: {', '.join(candidates)}"
    )


def _apply_checkpoint_config(config, checkpoint):
    """
    Overlay saved checkpoint settings onto the runtime config.
    Device is recomputed locally instead of restored from the checkpoint.
    """
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    for key, value in checkpoint_config.items():
        if hasattr(config, key) and key != "device":
            setattr(config, key, value)
    return config


def evaluate():
    config = Config()
    model_path = _resolve_model_path(config)
    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    config = _apply_checkpoint_config(config, checkpoint)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.model_save_path = model_path

    print("Initialising tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path)

    config.pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
    config.unk_token_id = tokenizer.convert_tokens_to_ids(config.unk_token)
    config.sos_token_id = tokenizer.convert_tokens_to_ids(config.sos_token)
    config.eos_token_id  = tokenizer.convert_tokens_to_ids(config.eos_token)
    config.vocab_size    = len(tokenizer)

    # FIX Bug #3: closing ) was missing on TranslationDataset call
    test_dataset = TranslationDataset(
        csv_file=config.test_csv,
        tokenizer=tokenizer,
        sequence_length=config.max_seq_length,
        config=config,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = load_model(config, model_path=config.model_save_path)

    print("Evaluating model on test set...")
    avg_bleu = evaluate_model(
        model, test_dataloader, config, tokenizer,
        max_seq_length=config.max_seq_length,
    )
    print(f"Final Average BLEU: {avg_bleu:.4f}")
