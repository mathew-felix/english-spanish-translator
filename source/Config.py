import torch 

class Config:
    def __init__(self):
        # Model hyperparameters
        self.embed_dim = 512
        self.latent_dim = 2048
        self.num_heads = 8
        self.dropout_rate = 0.2
        self.num_layers = 6
        self.max_seq_length = 40  # Increased sequence length to prevent truncation

        # Training hyperparameters
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.batch_size = 32
        self.clip_norm = 1.0
        self.patience = 3  # For early stopping

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

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Token IDs (to be assigned after tokenizer initialization)
        self.pad_token_id = None
        self.unk_token_id = None
        self.sos_token_id = None
        self.eos_token_id = None