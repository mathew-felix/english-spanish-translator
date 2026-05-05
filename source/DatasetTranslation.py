# FIX Bug #15: removed unused BertTokenizer import
import pandas as pd
import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, sequence_length, config):
        """
        Load translation pairs and verify required special tokens exist.
        FIX Bug #8: encoder uses SOS/EOS (not [CLS]/[SEP]) to match decoder convention.
        """
        self.data = pd.read_csv(csv_file).dropna()

        # Lowercase English for normalisation; leave Spanish casing intact
        # FIX Bug #4 (improvement): Spanish lowercasing removed — destroys accents
        self.data["English"] = self.data["English"].str.lower()

        self.tokenizer       = tokenizer
        self.sequence_length = sequence_length
        self.config          = config

        required_tokens = [config.pad_token, config.unk_token,
                           config.sos_token, config.eos_token]
        vocab = self.tokenizer.get_vocab()
        for tok in required_tokens:
            if tok not in vocab:
                raise ValueError(f"Required token '{tok}' not in tokenizer vocabulary.")

        self.pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
        self.unk_token_id = tokenizer.convert_tokens_to_ids(config.unk_token)
        self.sos_token_id = tokenizer.convert_tokens_to_ids(config.sos_token)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
          encoder_input : [SOS] + english_tokens + [EOS] + padding
          decoder_input : [SOS] + spanish_tokens + padding
          target        : spanish_tokens + [EOS] + padding
        FIX Bug #7: all tokenizer.encode() calls now have closing ).
        FIX Bug #8: encoder manually adds SOS/EOS (not bert [CLS]/[SEP]).
        """
        english = self.data.iloc[idx]["English"]
        spanish = self.data.iloc[idx]["Spanish"]

        # ── Encoder input: SOS + tokens + EOS ────────────────────────────────
        enc_tokens = self.tokenizer.encode(
            english,
            add_special_tokens=False,
            max_length=self.sequence_length - 2,  # reserve 2 slots for SOS + EOS
            truncation=True,
        )  # FIX Bug #7
        encoder_input = [self.sos_token_id] + enc_tokens + [self.eos_token_id]
        encoder_input += [self.pad_token_id] * (self.sequence_length - len(encoder_input))

        # ── Decoder input: SOS + spanish_tokens ──────────────────────────────
        dec_tokens = self.tokenizer.encode(
            spanish,
            add_special_tokens=False,
            max_length=self.sequence_length - 1,  # reserve 1 slot for SOS
            truncation=True,
        )  # FIX Bug #7
        decoder_input = [self.sos_token_id] + dec_tokens
        decoder_input += [self.pad_token_id] * (self.sequence_length - len(decoder_input))

        # ── Target: spanish_tokens + EOS ─────────────────────────────────────
        tgt_tokens = self.tokenizer.encode(
            spanish,
            add_special_tokens=False,
            max_length=self.sequence_length - 1,  # reserve 1 slot for EOS
            truncation=True,
        )  # FIX Bug #7
        target = tgt_tokens + [self.eos_token_id]
        target += [self.pad_token_id] * (self.sequence_length - len(target))

        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )  # FIX Bug #7: closing ) was missing
