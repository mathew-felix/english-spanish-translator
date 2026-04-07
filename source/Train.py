import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from source.Model import Transformer
from source.DatasetTranslation import TranslationDataset
from tqdm import tqdm
from source.Config import Config


def plot_losses(train_losses, val_losses, epoch, save_path="loss_plot.png"):
    """Plot and save training / validation loss curves up to the current epoch."""
    plt.figure(figsize=(10, 5))
    plt.title(f"Epoch {epoch} Loss")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _encode_sentence(sentence, tokenizer, config):
    """Tokenise a single sentence into a padded encoder-input tensor on config.device."""
    ids = tokenizer.encode(
        sentence,
        add_special_tokens=False,
        max_length=config.max_seq_length - 2,
        truncation=True,
    )
    ids = [config.sos_token_id] + ids + [config.eos_token_id]
    ids += [config.pad_token_id] * (config.max_seq_length - len(ids))
    return torch.tensor([ids], dtype=torch.long).to(config.device)


def _show_translations(transformer, tokenizer, config):
    """Improvement #12: print qualitative translation samples after each epoch."""
    samples = ["How are you?", "Where is the hospital?", "I need help with my homework."]
    transformer.eval()
    with torch.no_grad():
        for s in samples:
            enc_in = _encode_sentence(s, tokenizer, config)
            ids = transformer.generate(enc_in, config, beam_width=4)
            translation = tokenizer.decode(ids, skip_special_tokens=True)
            print(f"  [{s}] -> [{translation}]")
    transformer.train()


def train_model(transformer, train_dataloader, val_dataloader, config, tokenizer):
    """
    Train the Transformer with LR warmup, AMP mixed precision, and early stopping.
    Saves a rich checkpoint that includes epoch, val_loss, and config snapshot.
    """
    transformer.to(config.device)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
    # Improvement #5: mixed-precision scaler (no-op when CUDA not available)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Improvement #2: Noam-style warmup — linear ramp then inverse-sqrt decay
    def lr_lambda(step):
        step = max(step, 1)
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return (config.warmup_steps / step) ** 0.5

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=config.pad_token_id, label_smoothing=0.1
    )

    best_val_loss = float("inf")
    no_improve_epochs = 0
    train_losses, val_losses = [], []
    global_step = 0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        transformer.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
            encoder_inputs, decoder_inputs, targets = [x.to(config.device) for x in batch]

            optimizer.zero_grad()

            # Improvement #5: mixed precision forward + loss
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = transformer(encoder_inputs, decoder_inputs)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.clip_norm)
            scaler.step(optimizer)
            scaler.update()

            # Improvement #2: warmup step — only during warmup window
            global_step += 1
            if global_step <= config.warmup_steps:
                warmup_scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        transformer.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}", unit="batch"):
                encoder_inputs, decoder_inputs, targets = [x.to(config.device) for x in batch]
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = transformer(encoder_inputs, decoder_inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

        # Plateau scheduler steps per-epoch (after warmup completes)
        if global_step > config.warmup_steps:
            plateau_scheduler.step(avg_val_loss)

        # Improvement #12: qualitative samples
        print("Sample translations:")
        _show_translations(transformer, tokenizer, config)

        # Improvement #11: rich checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": transformer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": vars(config),
                },
                config.model_save_path,
            )
            print(f"Checkpoint saved at epoch {epoch + 1} — val_loss={avg_val_loss:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= config.patience:
            print(f"Early stopping after {config.patience} epochs with no improvement.")
            break

        plot_losses(train_losses, val_losses, epoch + 1)
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")


def Train():
    config = Config()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    os.makedirs(config.tokenizer_path, exist_ok=True)

    print("Initialising tokenizer...")
    # Improvement #1: multilingual tokenizer handles Spanish natively
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # FIX Bug #1: closing } was missing
    special_tokens = {
        "additional_special_tokens": [
            config.pad_token, config.unk_token,
            config.sos_token, config.eos_token,
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    # FIX Bug #6: update vocab_size after adding tokens
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
    config.unk_token_id = tokenizer.convert_tokens_to_ids(config.unk_token)
    config.sos_token_id = tokenizer.convert_tokens_to_ids(config.sos_token)
    config.eos_token_id  = tokenizer.convert_tokens_to_ids(config.eos_token)

    tokenizer.save_pretrained(config.tokenizer_path)
    print(f"Tokenizer saved to {config.tokenizer_path}")

    print("Initialising datasets...")
    train_dataset_full = TranslationDataset(
        csv_file=config.train_csv,
        tokenizer=tokenizer,
        sequence_length=config.max_seq_length,
        config=config,
    )  # FIX Bug #1: closing ) was missing

    train_size = int(0.9 * len(train_dataset_full))
    val_size   = len(train_dataset_full) - train_size
    split_generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset_full, [train_size, val_size], generator=split_generator
    )
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Improvement #9: num_workers + pin_memory for faster data loading
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print("Initialising Transformer model...")
    transformer = Transformer(config)
    transformer.to(config.device)

    print("Starting training...")
    train_model(transformer, train_dataloader, val_dataloader, config, tokenizer)
    print("Training complete.")

    # Load best checkpoint for example inference
    print("Loading best checkpoint for inference demo...")
    checkpoint = torch.load(config.model_save_path, map_location=config.device)
    transformer.load_state_dict(checkpoint["model_state_dict"])
    transformer.eval()

    example = "How are you?"
    print(f"\nTranslating: {example}")
    enc_in = _encode_sentence(example, tokenizer, config)
    ids = transformer.generate(enc_in, config, beam_width=4)
    print(f"Translated: {tokenizer.decode(ids, skip_special_tokens=True)}")
