# Deterministic Answer RAG OpenAI

An optimized Bengali FAQ system using Retrieval-Augmented Generation (RAG) with OpenAI's embeddings and GPT-4o for semantic question matching.

## Features

- **Dynamic File Discovery**: Automatically discovers and processes ALL `.txt` files in the `faq_data` directory
- **Bengali Text Support**: Handles Bengali and English FAQ content seamlessly  
- **Intelligent Caching**: Caches embeddings and content with automatic invalidation on file changes
- **Semantic Matching**: Uses OpenAI's text-embedding-3-large for finding semantically similar questions
- **GPT-4o Re-ranking**: Uses GPT-4o to determine the best matching FAQ from top candidates
- **Interactive CLI**: Easy-to-use command-line interface with debug mode

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Add your FAQ files (`.txt` format) to the `faq_data` directory
5. Run: `python bn_rag_openai_fixed_reply_bank.py`

## FAQ File Format

FAQ files should be in `.txt` format with the following structure:
```
Question: Your question here?
Answer: Your answer here.

Question: Another question?
Answer: Another answer.
```

Both English and Bengali markers are supported:
- Question markers: `Question:` or `প্রশ্ন:`
- Answer markers: `Answer:` or `উত্তর:`

## Usage

The system will automatically discover and process all `.txt` files in the `faq_data` directory. No need to manually specify file names - just add your FAQ files and run the system!

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ehzawad/deterministic-answer-rag-openai)
