# NLP DataLoader

A simple practice project designed to demonstrate how to create and use custom datasets and data loaders with tokenization and padding for Natural Language Processing (NLP) using PyTorch and TorchText.

This project covers:

- English and French sentence datasets  
- Tokenization using basic_english and SpaCy French models  
- Vocabulary creation  
- Custom PyTorch `Dataset` class  
- Batch padding with both `batch_first=True` and `batch_first=False`  
- Sentence sorting and batching strategies  

---

## 📦 Installation

### Prerequisites

- Python `>=3.10.12`
- `pip` or `poetry` (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/xantanio/ArtificialIntelligence.git
cd ArtificialIntelligence/nlpdataloader
```

### 2. Set up a virtural environment
```bash
python -m venv .venv
source .venv/bin/activate        # On Linux/macOS
.venv\Scripts\activate.bat       # On Windows
```

### 3. Install dependencies

```bash
pip install .
```

### 4. Install SpaCy french model
```bash
python -m spacy download fr_core_news_sm
```