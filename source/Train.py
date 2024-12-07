import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from source.Model import Transformer
from transformers import BertTokenizer  # Replace with your specific tokenizer
from source.DatasetTranslation import TranslationDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from source.Config import Config

def plot_losses(train_losses, val_losses, epoch, save_path="loss_plot.png"):
    """
    Plots training and validation losses up to the current epoch and saves the plot to a file.

    Parameters:
    - train_losses (list): List of training losses per epoch.
    - val_losses (list): List of validation losses per epoch.
    - epoch (int): Current epoch number.
    - save_path (str): Path where the plot will be saved (default: "loss_plot.png").
    """
    plt.figure(figsize=(10, 5))
    plt.title(f"Epoch {epoch} Loss")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')  # Save the plot
    plt.close()  # Close the figure to free memory

def train_model(transformer, train_dataloader, val_dataloader, config):
    """
    Train the Transformer model with loss visualization and early stopping.

    Parameters:
    - transformer (nn.Module): The Transformer model to train.
    - train_dataloader (DataLoader): DataLoader for the training dataset.
    - val_dataloader (DataLoader): DataLoader for the validation dataset.
    - config (Config): Configuration object with necessary parameters.
    """
    transformer.to(config.device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id, label_smoothing=0.1)

    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses = []
    val_losses = []

    writer = SummaryWriter('runs/translation_experiment')

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        transformer.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
            encoder_inputs, decoder_inputs, targets = [x.to(config.device) for x in batch]

            optimizer.zero_grad()
            outputs = transformer(encoder_inputs, decoder_inputs)  # [batch_size, seq_length, vocab_size]

            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_length, vocab_size]
            targets = targets.view(-1)  # [batch_size * seq_length]

            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=config.clip_norm)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)

        # Validation Phase
        transformer.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}", unit="batch"):
                encoder_inputs, decoder_inputs, targets = [x.to(config.device) for x in batch]
                outputs = transformer(encoder_inputs, decoder_inputs)  # [batch_size, seq_length, vocab_size]

                outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_length, vocab_size]
                targets = targets.view(-1)  # [batch_size * seq_length]

                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)

        # Update scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(transformer.state_dict(), config.model_save_path)
            print(f"Model saved at epoch {epoch + 1} with Validation Loss: {avg_val_loss:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        # Early Stopping
        if no_improve_epochs >= config.patience:
            print(f"Early stopping triggered after {config.patience} epochs with no improvement.")
            break

    # Plot losses
    plot_losses(train_losses, val_losses, epoch + 1)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch + 1)

    writer.close()

def Train():
    config = Config()

    # Create tokenizer directory if it doesn't exist
    os.makedirs(config.tokenizer_path, exist_ok=True)

    # Initialize tokenizer (e.g., BertTokenizer)
    print("Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Replace with your specific tokenizer if needed

    # Add special tokens if they are not present
    special_tokens = {
        'additional_special_tokens': [config.pad_token, config.unk_token, config.sos_token, config.eos_token]
    }
    tokenizer.add_special_tokens(special_tokens)

    # Update config.vocab_size after adding new tokens
    config.vocab_size = len(tokenizer)

    # Assign token IDs to the Config object
    config.pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
    config.unk_token_id = tokenizer.convert_tokens_to_ids(config.unk_token)
    config.sos_token_id = tokenizer.convert_tokens_to_ids(config.sos_token)
    config.eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)

    # Save the tokenizer for future use
    tokenizer.save_pretrained(config.tokenizer_path)
    print(f"Tokenizer saved to {config.tokenizer_path}")

    # Initialize datasets
    print("Initializing datasets...")
    train_dataset_full = TranslationDataset(
        csv_file=config.train_csv,
        tokenizer=tokenizer,
        sequence_length=config.max_seq_length,
        config=config
    )

    # Split dataset into training and validation sets (e.g., 90% train, 10% val)
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_full, [train_size, val_size])

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Initialize dataloaders
    print("Initializing dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    print("Initializing Transformer model...")
    transformer = Transformer(config)

    # Move model to device
    transformer.to(config.device)

    # Train the model
    print("Starting training...")
    train_model(transformer, train_dataloader, val_dataloader, config)

    print("Training complete.")

    # Load the best model for inference
    print("Loading the best model for inference...")
    transformer.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
    transformer.to(config.device)
    transformer.eval()

    # Example translation
    example_sentence = "How are you?"
    print(f"\nTranslating: {example_sentence}")
    encoder_input = tokenizer.encode(
        example_sentence,
        add_special_tokens=True,
        max_length=config.max_seq_length,
        padding='max_length',
        truncation=True
    )
    encoder_input = torch.tensor([encoder_input], dtype=torch.long).to(config.device)

    generated_ids = transformer.generate(encoder_input, config)
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Translated Sentence: {translation}")