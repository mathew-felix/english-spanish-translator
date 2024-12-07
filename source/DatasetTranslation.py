from transformers import BertTokenizer  # Replace with your specific tokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch


class TranslationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, sequence_length, config):
        """
        Initializes the dataset by loading data, preprocessing, and preparing token mappings.

        Parameters:
        - csv_file (str): Path to the CSV file containing translation pairs with 'English' and 'Spanish' columns.
        - tokenizer (PreTrainedTokenizer): Tokenizer used for encoding sentences.
        - sequence_length (int): Maximum sequence length for inputs and outputs.
        - config (Config): Configuration object with necessary parameters.
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data.dropna()  # Drop rows with NaN values
        self.data['English'] = self.data['English'].str.lower()
        self.data['Spanish'] = self.data['Spanish'].str.lower()

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.config = config

        # Define required tokens
        self.required_tokens = [config.pad_token, config.unk_token, config.sos_token, config.eos_token]

        # Verify that required tokens are in the tokenizer's vocabulary
        for token in self.required_tokens:
            if token not in self.tokenizer.get_vocab():
                raise ValueError(f"Required token '{token}' not found in tokenizer's vocabulary.")

        # Assign token IDs from the tokenizer
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(config.pad_token)
        self.unk_token_id = self.tokenizer.convert_tokens_to_ids(config.unk_token)
        self.sos_token_id = self.tokenizer.convert_tokens_to_ids(config.sos_token)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(config.eos_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized input and output sequences for a given index.

        Returns:
        - encoder_input (torch.Tensor): Token IDs for the English sentence.
        - decoder_input (torch.Tensor): Token IDs for the Spanish sentence with <SOS>.
        - target (torch.Tensor): Token IDs for the Spanish sentence with <END>.
        """
        english_sentence = self.data.iloc[idx]['English']
        spanish_sentence = self.data.iloc[idx]['Spanish']

        # Encode English sentence
        encoder_input = self.tokenizer.encode(
            english_sentence,
            add_special_tokens=True,
            max_length=self.sequence_length,
            padding='max_length',
            truncation=True
        )

        # Encode Spanish sentence for decoder input (prepend <SOS>)
        decoder_input = self.tokenizer.encode(
            spanish_sentence,
            add_special_tokens=False,
            max_length=self.sequence_length - 1,  # Reserve space for <SOS>
            truncation=True
        )
        decoder_input = [self.sos_token_id] + decoder_input
        decoder_input += [self.pad_token_id] * (self.sequence_length - len(decoder_input))

        # Encode Spanish sentence for target (append <END>)
        target = self.tokenizer.encode(
            spanish_sentence,
            add_special_tokens=False,
            max_length=self.sequence_length - 1,  # Reserve space for <END>
            truncation=True
        )
        target = target + [self.eos_token_id]
        target += [self.pad_token_id] * (self.sequence_length - len(target))

        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )