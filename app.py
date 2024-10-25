import logging
import os
import io
import re
import sys
import json
import math
import traceback
import requests
import certifi
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from newspaper import Article
import PyPDF2
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from groq import Groq
from mistralai import Mistral
from dotenv import load_dotenv

from src.web.interface import create_interface
from src.config.settings import (
    CUSTOM_LLM,
    CUSTOM_LLM_DEFAULT_MODEL,
    GROQ_API_KEY,
    MISTRAL_API_KEY,
    SEARXNG_URL,
    CURRENT_YEAR
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the SearXNG Scraper for News")
    iface = create_interface()
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()

# Define the fetch_custom_models function here
def fetch_custom_models():
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

# Fetch custom models and determine the default model
custom_models = fetch_custom_models()
all_models = ["huggingface", "groq", "mistral"] + custom_models

# Determine the default model
default_model = CUSTOM_LLM_DEFAULT_MODEL if CUSTOM_LLM_DEFAULT_MODEL in all_models else "groq"

logger.info(f"Default model selected: {default_model}")

# Use the environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    "mistralai/Mistral-Small-Instruct-2409",
    token=HF_TOKEN,
)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Mistral client
mistral_client = Mistral(api_key=MISTRAL_API_KEY)


from src.services.search_service import SearchService

search_service = SearchService()

def search_and_scrape(
    query: str,
    chat_history: str,
    ai_model: AIModel,
    num_results: int = 10,
    max_chars: int = 1500,
    time_range: str = "",
    language: str = "en",
    category: str = "general",
    engines: List[str] = [],
    safesearch: int = 2,
    method: str = "GET",
    llm_temperature: float = 0.2,
    timeout: int = 5,
    model: str = "huggingface",
    use_pydf2: bool = True
):
    return search_service.search_and_process(
        query=query,
        chat_history=chat_history,
        ai_model=ai_model,
        num_results=num_results,
        max_chars=max_chars,
        time_range=time_range,
        language=language,
        category=category,
        engines=engines,
        safesearch=safesearch,
        method=method,
        llm_temperature=llm_temperature
    )

# Helper function to get the appropriate client for each model
def get_client_for_model(model: str) -> Any:
    if model == "huggingface":
        return InferenceClient("mistralai/Mistral-Small-Instruct-2409", token=HF_TOKEN)
    elif model == "groq":
        return Groq(api_key=GROQ_API_KEY)
    elif model == "mistral":
        return Mistral(api_key=MISTRAL_API_KEY)
    elif CUSTOM_LLM and (model in fetch_custom_models() or model == CUSTOM_LLM_DEFAULT_MODEL):
        return None  # CustomModel doesn't need a client
    else:
        raise ValueError(f"Unsupported model: {model}")

iface = gr.ChatInterface(
    chat_function,
    title="Web Scraper for News with Sentinel AI",
    description="Ask Sentinel any question. It will search the web for recent information or use its knowledge base as appropriate.",
    theme=gr.Theme.from_hub("allenai/gradio-theme"),
    additional_inputs=[
        gr.Checkbox(label="Only do web search", value=True),  # Add this line
        gr.Slider(5, 20, value=3, step=1, label="Number of initial results"),
        gr.Slider(500, 10000, value=1500, step=100, label="Max characters to retrieve"),
        gr.Dropdown(["", "day", "week", "month", "year"], value="week", label="Time Range"),
        gr.Dropdown(["", "all", "en", "fr", "de", "es", "it", "nl", "pt", "pl", "ru", "zh"], value="en", label="Language"),
        gr.Dropdown(["", "general", "news", "images", "videos", "music", "files", "it", "science", "social media"], value="general", label="Category"),
        gr.Dropdown(
            ["google", "bing", "duckduckgo", "baidu", "yahoo", "qwant", "startpage"],
            multiselect=True,
            value=["google", "duckduckgo", "bing", "qwant"],
            label="Engines"
        ),
        gr.Slider(0, 2, value=2, step=1, label="Safe Search Level"),
        gr.Radio(["GET", "POST"], value="GET", label="HTTP Method"),
        gr.Slider(0, 1, value=0.2, step=0.1, label="LLM Temperature"),
        gr.Dropdown(all_models, value=default_model, label="LLM Model"),
        gr.Checkbox(label="Use PyPDF2 for PDF scraping", value=True),
    ],
    additional_inputs_accordion=gr.Accordion("⚙️ Advanced Parameters", open=True),
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
    chatbot=gr.Chatbot(
        show_copy_button=True,
        likeable=True,
        layout="bubble",
        height=500,
    )
)

if __name__ == "__main__":
    logger.info("Starting the SearXNG Scraper for News using ChatInterface with Advanced Parameters")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
