import os
from dotenv import load_dotenv
import logging
import datetime
import requests

# Load environment variables
load_dotenv()

# Current year
CURRENT_YEAR = datetime.datetime.now().year

# API Keys and URLs
SEARXNG_URL = os.getenv("SEARXNG_URL")
SEARXNG_KEY = os.getenv("SEARXNG_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CUSTOM_LLM = os.getenv("CUSTOM_LLM")
CUSTOM_LLM_DEFAULT_MODEL = os.getenv("CUSTOM_LLM_DEFAULT_MODEL")

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_custom_models():
    """Fetch available models from custom LLM endpoint"""
    if not CUSTOM_LLM:
        return []
    try:
        response = requests.get(f"{CUSTOM_LLM}/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        return [model["id"] for model in models]
    except Exception as e:
        logger.error(f"Error fetching custom models: {e}")
        return []
