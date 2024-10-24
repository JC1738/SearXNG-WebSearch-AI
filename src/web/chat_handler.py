import logging
import gradio as gr
from typing import List, Tuple, Generator
from ..models.ai_models import AIModelFactory
from ..services.search_service import SearchService
from ..utils.model_utils import get_client_for_model

logger = logging.getLogger(__name__)

class ChatHandler:
    def __init__(self):
        self.search_service = SearchService()


    def determine_query_type(self, message: str, chat_history: str, ai_model) -> str:
        if message.strip().lower() in ["hi", "hello", "hey"]:
            return "knowledge_base"
            
        try:
            system_prompt = """Determine if this query requires current information or can be answered from general knowledge."""
            user_prompt = f"Query: {message}\nChat history: {chat_history}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = ai_model.generate_response(
                messages=messages,
                max_tokens=10,
                temperature=0.2
            )
            return "web_search" if "web_search" in response.lower() else "knowledge_base"
        except Exception as e:
            logger.error(f"Error determining query type: {e}")
            return "web_search"

    def generate_ai_response(self, message: str, chat_history: str, ai_model, temperature: float) -> str:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"{chat_history}\nUser: {message}"}
            ]
            
            response = ai_model.generate_response(
                messages=messages,
                max_tokens=500,
                temperature=temperature
            )
            return response
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "I apologize, but I'm having trouble generating a response."

    def process_chat(
        self,
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
    ) -> str:
        
        logger.info(f"Processing chat with message: {message}")

        chat_history = "\n".join([f"{role}: {msg}" for role, msg in history])
        ai_model = AIModelFactory.create_model(model, get_client_for_model(model))
        
        if only_web_search:
            query_type = "web_search"
        else:
            query_type = self.determine_query_type(message, chat_history, ai_model)
        
        if query_type == "knowledge_base":
            return self.generate_ai_response(message, chat_history, ai_model, llm_temperature)
        else:
            gr.Info("Initiating Web Search")
            
            response = self.search_service.search_and_process(
                query=message,
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
                llm_temperature=llm_temperature,
                use_pydf2=use_pydf2,
                site_filter=site_filter
            )
            
            return response
