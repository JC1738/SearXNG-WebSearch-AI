import gradio as gr
from typing import List, Tuple
from .chat_handler import ChatHandler
from ..models.ai_models import AIModelFactory
from ..config.settings import fetch_custom_models, CUSTOM_LLM, CUSTOM_LLM_DEFAULT_MODEL

def create_interface():
    # Get available models
    custom_models = fetch_custom_models()
    all_models = ["huggingface", "groq", "mistral"] + custom_models
    default_model = CUSTOM_LLM_DEFAULT_MODEL if CUSTOM_LLM_DEFAULT_MODEL in all_models else "groq"

    chat_handler = ChatHandler()

    def chat_function(
        message: str,
        history: List[Tuple[str, str]],
        only_web_search: bool,
        num_results: int,
        max_chars: int,
        time_range: str,
        language: str,
        category: str,
        engines: List[str],
        safesearch: int,
        method: str,
        llm_temperature: float,
        model: str,
        use_pydf2: bool,
        site_filter: str
    ):
        return chat_handler.process_chat(
            message, history, only_web_search, num_results, max_chars,
            time_range, language, category, engines, safesearch,
            method, llm_temperature, model, use_pydf2, site_filter
        )

    iface = gr.ChatInterface(
        chat_function,
        title="Web Scraper for News with Sentinel AI",
        description="Ask Sentinel any question. It will search the web for recent information or use its knowledge base as appropriate.",
        theme=gr.Theme.from_hub("allenai/gradio-theme"),
        additional_inputs=[
            gr.Checkbox(label="Only do web search", value=True),
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
            gr.Textbox(label="Site Filter (e.g. wikipedia.org)", value=""),
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
    
    return iface
