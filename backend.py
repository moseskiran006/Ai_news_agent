
# Add at the VERY TOP of the file
from typing import List, Dict, Optional, Union
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # Explicit CUDA device setting

# Rest of your imports
import gradio as gr
import requests
from bs4 import BeautifulSoup
import feedparser
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import datetime
import pandas as pd
from newspaper import Article
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import yake
from collections import Counter
import logging
import re

# ... rest of your existing code ...

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Telegram Bot
TELEGRAM_BOT_TOKEN = "7905911322:AAH5xaHBtifMbBQZ0svpXhUBk_tNMUFpCcw"  # Replace with your bot token
TELEGRAM_CHANNEL_ID =  1379769340  # Replace with your channel username

# Initialize LLM models (using open-source models)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# For keyword extraction
kw_extractor = yake.KeywordExtractor(top=5, stopwords=stopwords.words('english'))

# News sources by category
NEWS_SOURCES = {
    "technology": [
        {"url": "https://techcrunch.com/feed/", "type": "rss"},
        {"url": "https://www.theverge.com/tech/rss/index.xml", "type": "rss"},
        {"url": "https://www.wired.com/category/technology/feed/", "type": "rss"}
    ],
    "sports": [
        {"url": "https://www.espn.com/espn/rss/news", "type": "rss"},
        {"url": "https://www.skysports.com/rss/12040", "type": "rss"},
        {"url": "https://www.bbc.com/sport/football", "type": "scrape"}
    ],
    "politics": [
        {"url": "https://www.politico.com/rss/politics08.xml", "type": "rss"},
        {"url": "https://www.theguardian.com/politics/rss", "type": "rss"},
        {"url": "https://www.bbc.com/news/politics", "type": "scrape"}
    ],
    "entertainment": [
        {"url": "https://www.hollywoodreporter.com/feed/", "type": "rss"},
        {"url": "https://www.etonline.com/news/rss", "type": "rss"},
        {"url": "https://www.bbc.com/culture", "type": "scrape"}
    ],
    "astrology": [
        {"url": "https://www.astrology.com/rss/horoscopes/daily", "type": "rss"},
        {"url": "https://www.horoscope.com/us/horoscopes/general/index.aspx", "type": "scrape"},
        {"url": "https://cafeastrology.com/", "type": "scrape"}
    ],
    "cinema": [
        {"url": "https://www.indiewire.com/feed/", "type": "rss"},
        {"url": "https://www.screendaily.com/feed", "type": "rss"},
        {"url": "https://www.imdb.com/news/movie", "type": "scrape"}
    ],
    "science": [
        {"url": "https://www.sciencedaily.com/rss/all.xml", "type": "rss"},
        {"url": "https://www.nature.com/nature.rss", "type": "rss"},
        {"url": "https://www.newscientist.com/section/news/", "type": "scrape"}
    ]
}

def escape_markdown(text: str) -> str:
    """Escape special MarkdownV2 characters for Telegram"""
    escape_chars = r'\_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def publish_to_telegram_sync(message: str) -> bool:
    """Synchronous Telegram publishing with robust error handling"""
    try:
        # Telegram has a 4096 character limit
        if len(message) > 4000:
            message = message[:4000] + "... [truncated]"
        
        # Prepare the request
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "text": escape_markdown(message),
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True
        }
        
        # Send the request
        response = requests.post(url, json=payload, timeout=10)
        response_data = response.json()
        
        # Check response
        if not response_data.get('ok', False):
            logger.error(f"Telegram API Error: {response_data.get('description')}")
            return False
            
        logger.info("Successfully published message to Telegram")
        return True
        
    except Exception as e:
        logger.error(f"Failed to publish to Telegram: {str(e)}")
        return False

def scrape_website(url: str) -> List[Dict]:
    """Scrape news articles from a website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        articles = []
        
        # Common patterns for news websites
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') and len(link.text.strip()) > 20:
                articles.append({
                    'title': link.text.strip(),
                    'url': href
                })
        
        return articles[:10]  # Limit to 10 articles
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return []

def parse_rss_feed(url: str) -> List[Dict]:
    """Parse RSS feed and return articles"""
    try:
        feed = feedparser.parse(url)
        articles = []
        
        for entry in feed.entries[:10]:  # Limit to 10 articles
            articles.append({
                'title': entry.title,
                'url': entry.link,
                'published': entry.get('published', ''),
                'summary': entry.get('summary', '')
            })
        
        return articles
    except Exception as e:
        logger.error(f"Error parsing RSS feed {url}: {e}")
        return []

def extract_article_text(url: str) -> str:
    """Extract main text from a news article"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Error extracting article from {url}: {e}")
        return ""

def summarize_text(text: str, max_length: int = 150) -> str:
    """Summarize text using BART model"""
    if not text.strip():
        return "No content available for summarization."
    
    try:
        # Tokenize the input text
        inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        
        # Generate summary
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=4,
            max_length=max_length,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        # Fallback to extractive summarization
        sentences = nltk.sent_tokenize(text)
        if len(sentences) > 3:
            return " ".join(sentences[:3])
        return text[:500]  # Return first 500 characters if summarization fails

def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text using YAKE"""
    try:
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords[:3]]  # Return top 3 keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        # Fallback to simple word frequency
        words = [word.lower() for word in word_tokenize(text) 
                if word.lower() not in stopwords.words('english') 
                and word.lower() not in string.punctuation]
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(3)]

def generate_telegram_post(article: Dict) -> str:
    """Generate formatted Telegram post with proper Markdown escaping"""
    title = escape_markdown(article.get('title', 'No title available'))
    summary = escape_markdown(article.get('summary', 'No summary available'))
    url = article.get('url', '')
    category = escape_markdown(article.get('category', 'general'))
    keywords = [escape_markdown(kw) for kw in article.get('keywords', [])]
    
    hashtags = " ".join([f"#{kw.replace(' ', '_')}" for kw in keywords])
    
    return (
        f"*{title}*\n\n"
        f"📝 *Summary:*\n{summary}\n\n"
        f"🔗 [Read more]({url})\n\n"
        f"🏷️ Category: #{category}\n"
        f"📌 Keywords: {hashtags}\n"
        f"⏰ Published: {article.get('published', 'N/A')}"
    )

def process_news_category(category: str, num_articles: int = 3) -> List[Dict]:
    """Process news for a specific category"""
    sources = NEWS_SOURCES.get(category, [])
    articles = []
    
    for source in sources[:2]:  # Limit to 2 sources per category
        if source['type'] == 'rss':
            new_articles = parse_rss_feed(source['url'])
        else:
            new_articles = scrape_website(source['url'])
        
        for article in new_articles[:num_articles]:
            try:
                # Skip if we already have enough articles
                if len(articles) >= num_articles:
                    break
                    
                # Get full article text
                article_text = extract_article_text(article['url'])
                if not article_text:
                    continue
                
                # Summarize
                summary = summarize_text(article_text)
                
                # Extract keywords
                keywords = extract_keywords(article_text)
                
                # Add to results
                articles.append({
                    'title': article.get('title', 'No title'),
                    'url': article['url'],
                    'summary': summary,
                    'keywords': keywords,
                    'category': category,
                    'published': article.get('published', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                })
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                continue
    
    return articles

def process_and_publish(category: str, num_articles: int = 3, publish: bool = True) -> List[Dict]:
    """Process news for a category and optionally publish to Telegram"""
    articles = process_news_category(category, num_articles)
    results = []
    
    for article in articles:
        telegram_post = generate_telegram_post(article)
        article['telegram_post'] = telegram_post
        
        if publish:
            success = publish_to_telegram_sync(telegram_post)
            article['published_to_telegram'] = success
            logger.info(f"Published to Telegram: {success} | {article['title'][:50]}...")
            
            # Add delay between posts to avoid rate limiting
            time.sleep(3)
        
        results.append(article)
    
    return results

def generate_report(articles: List[Dict]) -> pd.DataFrame:
    """Generate a report DataFrame from processed articles"""
    data = []
    for article in articles:
        data.append({
            'Title': article['title'],
            'Category': article['category'],
            'Summary': article['summary'],
            'Keywords': ", ".join(article['keywords']),
            'URL': article['url'],
            'Published': article['published'],
            'Telegram Post': article.get('published_to_telegram', False)
        })
    return pd.DataFrame(data)

# Gradio Interface
def gradio_interface(category: str, num_articles: int, publish: bool):
    """Gradio interface function"""
    start_time = time.time()
    
    try:
        # First verify Telegram connection if publishing
        if publish:
            test_msg = "🔔 AI News Agent is now online and working!"
            test_success = publish_to_telegram_sync(test_msg)
            if not test_success:
                return "❌ Failed to connect to Telegram. Check your bot token and channel ID.", pd.DataFrame()
        
        articles = process_and_publish(category, num_articles, publish)
        df = generate_report(articles)
        
        # Prepare output
        output_text = f"Processed {len(articles)} {category} articles\n\n"
        for idx, article in enumerate(articles, 1):
            output_text += f"\n📰 Article {idx}:\n"
            output_text += f"Title: {article['title']}\n"
            output_text += f"Summary: {article['summary']}\n"
            output_text += f"URL: {article['url']}\n"
            if publish:
                status = "✅ Success" if article.get('published_to_telegram', False) else "❌ Failed"
                output_text += f"Telegram Status: {status}\n"
        
        elapsed_time = time.time() - start_time
        output_text += f"\n⏱️ Processing time: {elapsed_time:.2f} seconds"
        
        return output_text, df
    
    except Exception as e:
        logger.error(f"Interface error: {str(e)}")
        return f"Error: {str(e)}", pd.DataFrame()

# Create Gradio interface
with gr.Blocks(title="AI News Agent") as demo:
    gr.Markdown("# 🚀 AI-Powered News Agent")
    gr.Markdown("Scrape, summarize, and publish news articles automatically!")
    
    with gr.Row():
        category = gr.Dropdown(
            label="Select News Category",
            choices=list(NEWS_SOURCES.keys()),
            value="technology"
        )
        num_articles = gr.Slider(
            label="Number of Articles",
            minimum=1,
            maximum=5,
            step=1,
            value=3
        )
        publish = gr.Checkbox(
            label="Publish to Telegram",
            value=True
        )
    
    submit_btn = gr.Button("Process News", variant="primary")
    
    with gr.Row():
        output_text = gr.Textbox(
            label="Processing Results",
            lines=10,
            interactive=False
        )
        output_df = gr.Dataframe(
            label="Articles Summary",
            headers=["Title", "Category", "Summary", "Keywords", "URL", "Published", "Telegram Post"],
            interactive=False
        )
    
    submit_btn.click(
        fn=gradio_interface,
        inputs=[category, num_articles, publish],
        outputs=[output_text, output_df]
    )
    
    gr.Markdown("### How it works:")
    gr.Markdown("""
    1. Select a news category from the dropdown
    2. Choose how many articles to process (1-5)
    3. Toggle whether to publish to Telegram
    4. Click "Process News" to start
    5. View results in the output boxes
    """)

if __name__ == "__main__":
    # First run a connectivity test
    print("Running Telegram connectivity test...")
    test_result = publish_to_telegram_sync("🔔 AI News Agent connectivity test")
    print("Telegram test:", "✅ Success" if test_result else "❌ Failed")
    
    demo.launch(share=True)