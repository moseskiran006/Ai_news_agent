# Ai_news_agent
# AI-Powered News Aggregator - Technical Implementation Guide

## Technologies Used

### Core Technologies:
- **Web Scraping**: BeautifulSoup (HTML parsing), Newspaper3k (article extraction), Feedparser (RSS feeds)
- **Natural Language Processing**: HuggingFace Transformers (BART-large-CNN model), YAKE (keyword extraction), NLTK (text processing)
- **Messaging**: Telegram Bot API (publishing)
- **UI**: Gradio (web interface)

### Key Python Libraries:
- `Requests` (HTTP calls)
- `Pandas` (data handling)
- `Python-dotenv` (environment variables)

## Large Language Models (LLMs)

### **BART-large-CNN (Facebook)**
- Used for abstractive summarization
- Pretrained model fine-tuned for summarization tasks
- Handles long-form text (up to 1024 tokens)

### **YAKE! (Unsupervised Keyword Extraction)**
- Lightweight keyword extraction
- Language-agnostic processing

## 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install gradio requests beautifulsoup4 feedparser transformers newspaper3k nltk yake python-dotenv pandas
python -m nltk.download punkt stopwords
```

## 2. Configuration

Create a `.env` file with Telegram credentials:

```env
TELEGRAM_BOT_TOKEN="your_bot_token"
TELEGRAM_CHANNEL_ID="@yourchannel"
```

## 3. Model Initialization
- **BART model** automatically downloads on the first run (~1.6GB)
- **YAKE** initializes with default English stopwords

## 4. Running the System

```bash
# Start the web interface
python backend.py
