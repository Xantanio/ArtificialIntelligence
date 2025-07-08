# 🤖 llmchatbots

This is a simple chatbot project built using the [Transformers](https://huggingface.co/transformers/) architecture. The main goal is to **practice project organization** and explore how to structure Python projects while learning to work with **Large Language Models (LLMs)**.

---

## 📚 About the Project

This chatbot project is part of my learning journey through:

- ✅ [Coursera: Generative AI and LLMs — Architecture and Data Preparation](https://www.coursera.org/learn/generative-ai-llm-architecture-data-preparation/home/module/2)
- ✅ [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) — to follow best practices for structuring Python packages

---

## 🛠️ Features

- Two chatbots using pretrained models from Hugging Face:
  - 🤖 Facebook BlenderBot (`facebook/blenderbot-400M-distill`)
  - 🤖 Google FLAN-T5 (`google/flan-t5-base`)
- Clean project structure with modular code
- Uses the Hugging Face `transformers` library and PyTorch

---

## 🚀 Running the Project

### 1. Clone the repository
git clone https://github.com/xantanio/ArtificialIntelligence.git
cd ArtificialIntelligence/llmchatbots

### 2. Set up a virtural environment
python -m venv .venv
source .venv/bin/activate        # On Linux/macOS
.venv\Scripts\activate.bat       # On Windows

### 3. Install dependencies from pyproject.toml
pip install .