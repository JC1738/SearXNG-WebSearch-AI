import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib.parse import urlparse
import logging
import certifi
from newspaper import Article
import PyPDF2
import io

logger = logging.getLogger(__name__)

def requests_retry_session(
    retries=0,
    backoff_factor=0.1,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def scrape_full_content(url, max_chars=3000, timeout=5, use_pydf2=True):
    try:
        logger.info(f"Scraping full content from: {url}")
        
        # Check if the URL ends with .pdf
        if url.lower().endswith('.pdf'):
            if use_pydf2:
                return scrape_pdf_content(url, max_chars, timeout)
            else:
                logger.info(f"Skipping PDF document: {url}")
                return None
        
        # Use Newspaper3k for non-PDF content
        content = scrape_with_newspaper(url)
        
        # Limit the content to max_chars
        return content[:max_chars] if content else ""
    except requests.Timeout:
        logger.error(f"Timeout error while scraping full content from {url}")
        return ""
    except Exception as e:
        logger.error(f"Error scraping full content from {url}: {e}")
        return ""

def scrape_pdf_content(url, max_chars=3000, timeout=5):
    try:
        logger.info(f"Scraping PDF content from: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
        return content[:max_chars] if content else ""
    except Exception as e:
        logger.error(f"Error scraping PDF content from {url}: {e}")
        return ""

def scrape_with_newspaper(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        content = f"Title: {article.title}\n\n"
        content += article.text
        
        if article.publish_date:
            content += f"\n\nPublish Date: {article.publish_date}"
        if article.authors:
            content += f"\n\nAuthors: {', '.join(article.authors)}"
        if article.top_image:
            content += f"\n\nTop Image URL: {article.top_image}"
        
        return content
    except Exception as e:
        logger.error(f"Error scraping {url} with Newspaper3k: {e}")
        return ""
