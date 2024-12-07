from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from source.Config import Config
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from source.Model import Transformer
from source.DatasetTranslation import TranslationDataset
from tqdm import tqdm
from transformers import BertTokenizer

def evaluate_model(transformer, test_dataloader, config, max_seq_length=40):
    """
    Evaluates the Transformer model using BLEU scores.
    Adds a progress bar using tqdm and returns BLEU scores for plotting.
    """
    device = config.device
    transformer.to(device)
    transformer.eval()

    bleu_scores = []

    with torch.no_grad():
        # Wrap the dataloader with tqdm for a progress bar
        for batch in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
            encoder_inputs, decoder_inputs, targets = [x.to(device) for x in batch]

            # Generate translations using greedy decoding
            predicted_tokens = generate_translations(transformer, encoder_inputs, config, max_seq_length)

            # Convert predicted tokens and targets back to sentences
            predicted_sentences = decode_sentences(predicted_tokens, config.spanish_index)
            reference_sentences = decode_sentences(targets.cpu().numpy(), config.spanish_index)

            # Compute BLEU score for each sentence in the batch
            for pred, ref in zip(predicted_sentences, reference_sentences):
                if len(ref) == 0:
                    bleu = 0
                else:
                    bleu = sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        smoothing_function=SmoothingFunction().method1
                    )
                bleu_scores.append(bleu)

    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    print(f"Average BLEU Score: {average_bleu:.4f}")

    # Plot BLEU score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(bleu_scores, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of BLEU Scores Across Test Set')
    plt.xlabel('BLEU Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('bleu_score_distribution.png')  # Save the plot as an image
    plt.close() 

    return average_bleu


def generate_translations(transformer, encoder_inputs, config, max_seq_length=40):
    """
    Generate translations using greedy decoding.
    """
    device = encoder_inputs.device
    batch_size = encoder_inputs.size(0)
    decoded_tokens = torch.full((batch_size, max_seq_length), config.pad_token_id, dtype=torch.long).to(device)

    # Start with <SOS> token
    decoded_tokens[:, 0] = config.sos_token_id

    for t in range(1, max_seq_length):
        # Prepare decoder inputs up to current time step
        current_decoder_input = decoded_tokens[:, :t]

        # Get model outputs
        outputs = transformer(encoder_inputs, current_decoder_input)
        next_token_logits = outputs[:, -1, :]  # Get logits for the last time step

        # Greedy decoding: select the token with highest probability
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        decoded_tokens[:, t] = next_tokens

        # Check if all sequences have generated <END> token
        if torch.all(next_tokens == config.eos_token_id):
            break

    return decoded_tokens.cpu().numpy()


def decode_sentences(token_indices, vocab):
    """
    Convert token indices back to sentences.
    """
    index_to_vocab = {idx: word for word, idx in vocab.items()}
    sentences = []

    for tokens in token_indices:
        words = []
        for token in tokens:
            if token == vocab.get("<END>", 0):
                break  # Stop at <END> token
            if token not in [vocab.get("<PAD>", 0), vocab.get("<SOS>", 0), vocab.get("<END>", 0)]:
                word = index_to_vocab.get(token, "<UNK>")
                words.append(word)
        sentence = " ".join(words)
        sentences.append(sentence)

    return sentences


def load_model(config, model_path="best_model.pth"):
    """
    Load the trained model from a checkpoint.
    """
    # Initialize the model with config
    transformer = Transformer(config)

    # Load the saved state_dict
    print(f"Loading model from '{model_path}'...")
    transformer.load_state_dict(torch.load(model_path, map_location=config.device))
    transformer.to(config.device)
    transformer.eval()  # Set the model to evaluation mode
    return transformer


def evaluate():

    # Initialize configuration
    config = Config()

    # Initialize tokenizer (must match the one used during training)
    print("Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path)  # Path where tokenizer was saved

    # Assign token IDs to the Config object
    config.pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
    config.unk_token_id = tokenizer.convert_tokens_to_ids(config.unk_token)
    config.sos_token_id = tokenizer.convert_tokens_to_ids(config.sos_token)
    config.eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)

    # Update vocab_size based on tokenizer
    config.vocab_size = len(tokenizer)

    # Assign vocabularies
    config.english_index = tokenizer.get_vocab()  # Assuming single tokenizer for both languages
    config.spanish_index = tokenizer.get_vocab()  # Modify if separate tokenizers are used

    # Initialize Test Dataset correctly
    test_csv = "./data/test.csv"
    test_dataset = TranslationDataset(
        csv_file=test_csv,
        tokenizer=tokenizer,
        sequence_length=config.max_seq_length,
        config=config
    )

    # Prepare Test Dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load Trained Model
    model = load_model(config, model_path=config.model_save_path)

    # Evaluate Model
    print("Evaluating model on test set...")
    average_bleu = evaluate_model(model, test_dataloader, config, max_seq_length=config.max_seq_length)
    print(f"Average BLEU Score: {average_bleu:.4f}")