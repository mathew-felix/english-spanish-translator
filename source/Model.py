import torch
from torch import nn
import math


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.final_layer = nn.Linear(config.embed_dim, config.vocab_size)

        # Tie embeddings (optional but recommended)
        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.final_layer.weight = self.decoder.embedding.weight

    def forward(self, encoder_input, decoder_input):
        """
        Forward pass through the Transformer.

        Parameters:
        - encoder_input (torch.Tensor): Token IDs for the source sentence [batch_size, src_seq_length]
        - decoder_input (torch.Tensor): Token IDs for the target sentence [batch_size, trg_seq_length]

        Returns:
        - logits (torch.Tensor): Raw predictions [batch_size, trg_seq_length, vocab_size]
        """
        encoder_output = self.encoder(encoder_input)  # [batch_size, src_seq_length, embed_dim]
        decoder_output = self.decoder(decoder_input, encoder_output)  # [batch_size, trg_seq_length, embed_dim]
        logits = self.final_layer(decoder_output)  # [batch_size, trg_seq_length, vocab_size]
        return logits

    def generate(self, encoder_input, config, max_length=None):
        """
        Generates translation for the given encoder_input using greedy decoding.

        Parameters:
        - encoder_input (torch.Tensor): Token IDs for the source sentence [1, src_seq_length]
        - config (Config): Configuration object with necessary parameters
        - max_length (int): Maximum length of the generated translation. Defaults to config.max_seq_length

        Returns:
        - generated_ids (list): List of generated token IDs
        """
        self.eval()
        if max_length is None:
            max_length = config.max_seq_length

        with torch.no_grad():
            encoder_output = self.encoder(encoder_input)  # [1, src_seq_length, embed_dim]

            # Initialize decoder input with <SOS>
            decoder_input = torch.tensor([[config.sos_token_id]], device=config.device)  # [1, 1]

            generated_ids = []

            for _ in range(max_length):
                decoder_output = self.decoder(decoder_input, encoder_output)  # [1, seq_length, embed_dim]
                logits = self.final_layer(decoder_output)  # [1, seq_length, vocab_size]
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1)  # [1]

                next_token_id = next_token.item()
                if next_token_id == config.eos_token_id:
                    break

                generated_ids.append(next_token_id)
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)  # [1, seq_length + 1]

            return generated_ids

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
        self.positional_encoding = self.create_sinusoidal_encoding(config.max_seq_length, config.embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config.embed_dim, config.num_heads, config.latent_dim, config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        self.dropout = nn.Dropout(config.dropout_rate)

    def create_sinusoidal_encoding(self, max_seq_length, embed_dim):
        """
        Creates fixed sinusoidal positional encodings.

        Returns:
        - pe (torch.Tensor): Positional encodings [1, max_seq_length, embed_dim]
        """
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, embed_dim]
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        """
        Forward pass through the Encoder.

        Parameters:
        - x (torch.Tensor): Token IDs [batch_size, src_seq_length]

        Returns:
        - x (torch.Tensor): Encoder outputs [batch_size, src_seq_length, embed_dim]
        """
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
        self.positional_encoding = self.create_sinusoidal_encoding(config.max_seq_length, config.embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config.embed_dim, config.num_heads, config.latent_dim, config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        self.dropout = nn.Dropout(config.dropout_rate)

    def create_sinusoidal_encoding(self, max_seq_length, embed_dim):
        """
        Creates fixed sinusoidal positional encodings.

        Returns:
        - pe (torch.Tensor): Positional encodings [1, max_seq_length, embed_dim]
        """
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, embed_dim]
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x, encoder_output):
        """
        Forward pass through the Decoder.

        Parameters:
        - x (torch.Tensor): Token IDs [batch_size, trg_seq_length]
        - encoder_output (torch.Tensor): Encoder outputs [batch_size, src_seq_length, embed_dim]

        Returns:
        - x (torch.Tensor): Decoder outputs [batch_size, trg_seq_length, embed_dim]
        """
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout_rate):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the Encoder Layer.

        Parameters:
        - x (torch.Tensor): Input embeddings [batch_size, seq_length, embed_dim]

        Returns:
        - x (torch.Tensor): Output embeddings [batch_size, seq_length, embed_dim]
        """
        attn_output, _ = self.self_attention(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout_rate):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output):
        """
        Forward pass through the Decoder Layer.

        Parameters:
        - x (torch.Tensor): Decoder input embeddings [batch_size, trg_seq_length, embed_dim]
        - encoder_output (torch.Tensor): Encoder outputs [batch_size, src_seq_length, embed_dim]

        Returns:
        - x (torch.Tensor): Output embeddings [batch_size, trg_seq_length, embed_dim]
        """
        # Self-attention with causal masking
        seq_length = x.size(1)
        subsequent_mask = torch.triu(torch.ones((seq_length, seq_length), device=x.device), diagonal=1).bool()
        attn_output, _ = self.self_attention(x, x, x, attn_mask=subsequent_mask)
        x = self.layernorm1(x + self.dropout(attn_output))

        # Cross-attention
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output)
        x = self.layernorm2(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layernorm3(x + self.dropout(ff_output))

        return x