# English-to-Spanish Translator using Transformer

This project implements an English-to-Spanish translator using the Transformer architecture. The model processes bilingual datasets to translate English sentences into Spanish. The Transformer leverages self-attention and cross-attention mechanisms for contextual understanding and accurate translation.

---

## Features
- Preprocessing large bilingual datasets with efficient data cleaning and normalization.
- Training a Transformer-based encoder-decoder model for sequence-to-sequence translation.
- Evaluation using the BLEU metric for translation quality.
- Pretrained BERT tokenizer for text processing and vocabulary mapping.
- Achieved a BLEU score of **0.47**, with a **42% improvement** over baseline metrics.

---

## Technologies Used
### **Programming Language**
- Python

### **Libraries and Frameworks**
- PyTorch
- Hugging Face BERT Tokenizer
- Pandas, NumPy
- NLTK for BLEU score computation
- Matplotlib for visualization
- tqdm for monitoring training progress

### **Hardware**
- NVIDIA GPU with CUDA support for accelerated training

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/english-spanish-translator.git
   cd english-spanish-translator

2. Set up a virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate 
    pip install -r requirements

## Usage
    ```bash
    python run.py --step download       
    python run.py --step preprocess     
    python run.py --step train          
    python run.py --step evaluate       

## Configuration
    Modify Config.py to adjust hyperparameters such as:
        1. Learning rate
        2. Batch size
        3. Number of epochs
        4. Maximum sequence length 

## Results
    BLEU Score: 0.47
    Validation Loss: 2.09 after 10 epochs
    Dataset Size: 1.9M rows (1.5M training, 0.4M testing)
    
## License
This project is licensed under the MIT License.