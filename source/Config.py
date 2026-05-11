import torch


class Config:
    def __init__(self):
        # Model hyperparameters
        self.embed_dim = 512
        self.latent_dim = 2048
        self.num_heads = 8
        self.dropout_rate = 0.2
        self.num_layers = 6
        self.max_seq_length = 40

        # Training hyperparameters
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.batch_size = 32
        self.clip_norm = 1.0
        self.patience = 3
        self.warmup_steps = 4000       # FIX #improvement: LR warmup steps
        self.seed = 42
        self.bleu_eval_batches = 10

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<END>"

        # Paths
        self.train_csv = "./data/train.csv"
        self.test_csv = "./data/test.csv"
        self.tokenizer_path = "./data/tokenizer/"
        self.model_save_path = "best_model.pth"
        self.wandb_project = "english-spanish-translator"
        self.wandb_entity = None
        self.wandb_mode = "online"
        self.wandb_anonymous = "allow"

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FIX Bug #6: vocab_size initialised here so Transformer() never hits AttributeError
        # Default for bert-base-multilingual-cased; overwritten after add_special_tokens()
        self.vocab_size = 119547

        # Token IDs assigned after tokenizer initialisation in Train.py / Evaluate.py
        self.pad_token_id = None
        self.unk_token_id = None
        self.sos_token_id = None
        self.eos_token_id = None
