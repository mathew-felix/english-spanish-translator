import os
import re
import threading

import torch
from transformers import BertTokenizer

from source.Config import Config
from source.Model import Transformer

_INFERENCE_ENGINE = None
_INFERENCE_LOCK = threading.Lock()


DEMO_TRANSLATIONS = {
    "the parliamentary session was adjourned.": "Se aplazo la sesion parlamentaria.",
    "the committee approved the amendment.": "El comite aprobo la enmienda.",
    "the council voted on the motion.": "El Consejo voto sobre la mocion.",
    "the agency published the legal notice.": "La agencia publico el aviso legal.",
    "the policy report was submitted for review.": "El informe de politica se presento para revision.",
}


def _demo_mode_enabled():
    """Return whether inference should use deterministic demo translations."""
    return os.getenv("TRANSLATOR_DEMO_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "demo",
    }


class DemoInferenceEngine:
    """Checkpoint-free inference engine for fast demos and CI smoke tests."""

    def translate(self, text):
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text must not be empty.")

        lowered = cleaned_text.lower()
        if lowered in DEMO_TRANSLATIONS:
            return DEMO_TRANSLATIONS[lowered]
        return f"Traduccion demo pendiente de revision: {cleaned_text}"


class InferenceEngine:
    """Loads the tokenizer and checkpoint once for API inference.
    The checkpoint config is applied before model construction so weight shapes match.
    """

    def __init__(self):
        """Initialise the cached inference runtime.
        The loader resolves repo-relative paths so the API works from the project root.
        """
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config, self.tokenizer, self.model = self._load_runtime()

    def _resolve_repo_path(self, path_value):
        """Resolve a config path relative to the repository root.
        Absolute paths are preserved and relative paths are normalised.
        """
        if os.path.isabs(path_value):
            return path_value

        cleaned_path = path_value.lstrip("./")
        return os.path.join(self.repo_root, cleaned_path)

    def _resolve_model_path(self, config):
        """Resolve the checkpoint path for inference.
        Falls back to the `weights/` directory when the root-level path is absent.
        """
        candidates = [
            self._resolve_repo_path(config.model_save_path),
            os.path.join(self.repo_root, "weights", os.path.basename(config.model_save_path)),
        ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(
            f"Model checkpoint not found. Checked: {', '.join(candidates)}"
        )

    def _load_checkpoint(self, model_path):
        """Load a checkpoint onto CPU first for portability.
        The model is moved to the active runtime device after construction.
        """
        checkpoint = torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint '{model_path}' is missing the expected model_state_dict."
            )
        return checkpoint

    def _apply_checkpoint_config(self, config, checkpoint_config):
        """Overlay saved checkpoint settings onto the runtime config.
        Device is always recomputed locally instead of trusting the saved value.
        """
        for key, value in checkpoint_config.items():
            if hasattr(config, key) and key != "device":
                setattr(config, key, value)
        return config

    def _load_runtime(self):
        """Build the tokenizer, config, and model used by the API.
        The tokenizer directory must already exist from a previous training run.
        """
        config = Config()
        model_path = self._resolve_model_path(config)
        checkpoint = self._load_checkpoint(model_path)
        checkpoint_config = checkpoint.get("config", {})

        if isinstance(checkpoint_config, dict):
            config = self._apply_checkpoint_config(config, checkpoint_config)

        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.model_save_path = model_path
        tokenizer_path = self._resolve_repo_path(config.tokenizer_path)

        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer directory '{tokenizer_path}' was not found."
            )

        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        config.tokenizer_path = tokenizer_path
        config.vocab_size = len(tokenizer)
        config.pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
        config.unk_token_id = tokenizer.convert_tokens_to_ids(config.unk_token)
        config.sos_token_id = tokenizer.convert_tokens_to_ids(config.sos_token)
        config.eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)

        model = Transformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(config.device)
        model.eval()

        return config, tokenizer, model

    def _encode_text(self, text):
        """Tokenise one input sentence for encoder-side inference.
        English input is lowercased to match the current training dataset format.
        """
        token_ids = self.tokenizer.encode(
            text.strip().lower(),
            add_special_tokens=False,
            max_length=self.config.max_seq_length - 2,
            truncation=True,
        )
        token_ids = [self.config.sos_token_id] + token_ids + [self.config.eos_token_id]
        token_ids += [self.config.pad_token_id] * (
            self.config.max_seq_length - len(token_ids)
        )
        return torch.tensor([token_ids], dtype=torch.long, device=self.config.device)

    def _normalise_decoded_text(self, text):
        """Clean tokenizer spacing artifacts in decoded Spanish text.
        Opening punctuation should stay attached to the following token.
        """
        text = re.sub(r"([¿¡])\s+", r"\1", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def translate(self, text):
        """Translate one English sentence into Spanish.
        Empty input is rejected before it reaches the model.
        Retries with a narrower beam when wide-beam decoding collapses to empty output.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text must not be empty.")

        encoder_input = self._encode_text(cleaned_text)
        with torch.no_grad():
            generated_ids = self.model.generate(
                encoder_input, self.config, beam_width=4
            )
            if not generated_ids:
                generated_ids = self.model.generate(
                    encoder_input, self.config, beam_width=2
                )
        translated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._normalise_decoded_text(translated_text)


def get_inference_engine():
    """Return the process-wide inference singleton.
    A lock prevents duplicate model loads under concurrent startup.
    """
    global _INFERENCE_ENGINE

    if _INFERENCE_ENGINE is None:
        with _INFERENCE_LOCK:
            if _INFERENCE_ENGINE is None:
                if _demo_mode_enabled():
                    _INFERENCE_ENGINE = DemoInferenceEngine()
                else:
                    _INFERENCE_ENGINE = InferenceEngine()
    return _INFERENCE_ENGINE


def translate(text):
    """Translate one text string with the cached inference runtime.
    The underlying model and tokenizer are loaded only once per process.
    """
    return get_inference_engine().translate(text)
