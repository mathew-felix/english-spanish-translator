import math

import torch
from torch import nn


# FIX Bug #5: extracted to module level — was duplicated inside Encoder and Decoder
def create_sinusoidal_encoding(max_seq_length: int, embed_dim: int) -> nn.Parameter:
    """Creates fixed sinusoidal positional encodings. Shape: [1, max_seq_length, embed_dim]."""
    position = torch.arange(0, max_seq_length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(max_seq_length, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return nn.Parameter(pe.unsqueeze(0), requires_grad=False)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.final_layer = nn.Linear(config.embed_dim, config.vocab_size)

        # Weight tying: encoder emb == decoder emb == output projection
        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.final_layer.weight = self.decoder.embedding.weight

    def forward(self, encoder_input, decoder_input):
        """
        Forward pass through the Transformer.
        encoder_input: (batch, src_seq_len) token IDs
        decoder_input: (batch, trg_seq_len) token IDs
        Returns logits: (batch, trg_seq_len, vocab_size)
        """
        src_key_padding_mask = self.encoder.build_padding_mask(encoder_input)
        encoder_output = self.encoder(
            encoder_input,
            src_key_padding_mask=src_key_padding_mask,
        )
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            src_key_padding_mask=src_key_padding_mask,
        )
        return self.final_layer(decoder_output)

    def generate(self, encoder_input, config, max_length=None, beam_width=4):
        """
        Beam search decoding. beam_width=1 degrades to greedy.
        encoder_input: (1, src_seq_len) — single sentence only.
        Returns list of generated token IDs (SOS/EOS stripped).
        """
        self.eval()
        if max_length is None:
            max_length = config.max_seq_length
        device = encoder_input.device

        with torch.no_grad():
            src_key_padding_mask = self.encoder.build_padding_mask(encoder_input)
            encoder_output = self.encoder(
                encoder_input,
                src_key_padding_mask=src_key_padding_mask,
            )  # (1, src_len, embed_dim)

            # Each beam: (cumulative_log_prob, token_id_list)
            beams = [(0.0, [config.sos_token_id])]
            completed = []

            for _ in range(max_length):
                candidates = []
                all_done = True

                for score, seq in beams:
                    if seq[-1] == config.eos_token_id:
                        completed.append((score / max(len(seq), 1), seq))
                        continue
                    all_done = False

                    dec_in = torch.tensor([seq], dtype=torch.long, device=device)
                    dec_out = self.decoder(
                        dec_in,
                        encoder_output,
                        src_key_padding_mask=src_key_padding_mask,
                    )
                    logits = self.final_layer(dec_out[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)
                    top_log_probs, top_tokens = log_probs[0].topk(beam_width)

                    for lp, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
                        candidates.append((score + lp, seq + [tok]))

                if all_done or not candidates:
                    break

                beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

            completed.extend(
                (s / max(len(seq), 1), seq) for s, seq in beams
                if seq[-1] != config.eos_token_id
            )

            if not completed:
                return []

            _, best_seq = max(completed, key=lambda x: x[0])
            return [
                t for t in best_seq
                if t not in (config.sos_token_id, config.eos_token_id)
            ]


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        # FIX Bug #5: shared helper, no duplication
        self.positional_encoding = create_sinusoidal_encoding(
            config.max_seq_length, config.embed_dim
        )
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config.embed_dim, config.num_heads,
                                    config.latent_dim, config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        self.dropout = nn.Dropout(config.dropout_rate)

    def build_padding_mask(self, x):
        """Return a bool padding mask for encoder tokens using the configured pad ID."""
        if self.pad_token_id is None:
            return None
        return x.eq(self.pad_token_id)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (batch, src_seq_len) token IDs
        Returns encoder output: (batch, src_seq_len, embed_dim)
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = self.build_padding_mask(x)
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        # FIX Bug #5: shared helper, no duplication
        self.positional_encoding = create_sinusoidal_encoding(
            config.max_seq_length, config.embed_dim
        )
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config.embed_dim, config.num_heads,
                                    config.latent_dim, config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        self.dropout = nn.Dropout(config.dropout_rate)

    def build_padding_mask(self, x):
        """Return a bool padding mask for decoder tokens using the configured pad ID."""
        if self.pad_token_id is None:
            return None
        return x.eq(self.pad_token_id)

    def forward(self, x, encoder_output, src_key_padding_mask=None):
        """
        x: (batch, trg_seq_len) token IDs
        encoder_output: (batch, src_seq_len, embed_dim)
        Returns decoder output: (batch, trg_seq_len, embed_dim)
        """
        trg_key_padding_mask = self.build_padding_mask(x)
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                trg_key_padding_mask=trg_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout_rate):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        # Improvement #10: dropout between linear layers inside FFN
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim, embed_dim),
        )
        # Improvement #6: Pre-LN — layernorm applied to input before each sublayer
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (batch, seq_len, embed_dim)
        src_key_padding_mask: (batch, seq_len) — True = ignore position (PAD)
        FIX Bug #4 + Improvement #6: Pre-LN with padding mask.
        """
        # Pre-LN self-attention
        normed = self.layernorm1(x)
        attn_out, _ = self.self_attention(
            normed, normed, normed, key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout(attn_out)

        # Pre-LN feed-forward
        normed = self.layernorm2(x)
        x = x + self.dropout(self.feed_forward(normed))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout_rate):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        # Improvement #10: dropout inside FFN
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim, embed_dim),
        )
        # Improvement #6: Pre-LN
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output, trg_key_padding_mask=None, memory_key_padding_mask=None):
        """
        x: (batch, trg_seq_len, embed_dim)
        encoder_output: (batch, src_seq_len, embed_dim)
        Improvement #6: Pre-LN in all three sublayers.
        """
        seq_length = x.size(1)
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=x.device), diagonal=1
        ).bool()

        # Pre-LN masked self-attention
        normed = self.layernorm1(x)
        attn_out, _ = self.self_attention(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            key_padding_mask=trg_key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # Pre-LN cross-attention
        normed = self.layernorm2(x)
        attn_out, _ = self.cross_attention(
            normed,
            encoder_output,
            encoder_output,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # Pre-LN feed-forward
        normed = self.layernorm3(x)
        x = x + self.dropout(self.feed_forward(normed))
        return x
