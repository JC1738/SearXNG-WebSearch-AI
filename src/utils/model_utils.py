import os
from huggingface_hub import InferenceClient
from groq import Groq
from mistralai import Mistral
from ..config.settings import (
    HF_TOKEN,
    GROQ_API_KEY,
    MISTRAL_API_KEY,
    CUSTOM_LLM,
    fetch_custom_models,
    CUSTOM_LLM_DEFAULT_MODEL
)

def get_client_for_model(model: str):
    """Get the appropriate client for each model"""
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
