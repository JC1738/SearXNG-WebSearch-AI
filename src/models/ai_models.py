from abc import ABC, abstractmethod
from typing import List, Dict, Any
import requests
import logging
import time
from openai import OpenAI, RateLimitError
from src.config.settings import CUSTOM_LLM, CUSTOM_LLM_KEY
from src.config.settings import fetch_custom_models

logger = logging.getLogger(__name__)

class AIModel(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        pass

class HuggingFaceModel(AIModel):
    def __init__(self, client):
        self.client = client

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

class GroqModel(AIModel):
    def __init__(self, client):
        self.client = client

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

class MistralModel(AIModel):
    def __init__(self, client):
        self.client = client

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        response = self.client.chat.complete(
            model="open-mistral-nemo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

class CustomModel(AIModel):
    def __init__(self, model_name):
        self.model_name = model_name

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            if CUSTOM_LLM_KEY:
                headers["Authorization"] = f"Bearer {CUSTOM_LLM_KEY}"

            response = requests.post(
                f"{CUSTOM_LLM}/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error generating response from custom model: {e}")
            return "Error: Unable to generate response from custom model."

class OpenAIModel(AIModel):
    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        logger.info(f"Initializing OpenAI model with API key: {api_key} and model name: {model_name}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts: {e}")
                    return "Error: Rate limit exceeded. Please try again later."
            except Exception as e:
                logger.error(f"Error generating response from OpenAI model: {e}")
                return "Error: Unable to generate response from OpenAI model."

        return "Error: Unable to generate response after multiple attempts."

class AIModelFactory:
    @staticmethod
    def create_model(model_name: str, client: Any = None) -> AIModel:
        if model_name == "huggingface":
            return HuggingFaceModel(client)
        elif model_name == "groq":
            return GroqModel(client)
        elif model_name == "mistral":
            return MistralModel(client)
        elif CUSTOM_LLM and model_name in fetch_custom_models():
            if client and isinstance(client, OpenAI):
                return OpenAIModel(CUSTOM_LLM_KEY, model_name, base_url=CUSTOM_LLM)
            else:
                return CustomModel(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
