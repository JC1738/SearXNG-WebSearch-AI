import logging
from typing import Dict
from ..models.ai_models import AIModel

logger = logging.getLogger(__name__)

class AssessmentService:
    def assess_relevance_and_summarize(self, ai_model: AIModel, query: str, 
                                     document: Dict, temperature: float = 0.2) -> str:
        system_prompt = """You are a world-class AI assistant specializing in news analysis. 
        Your task is to assess the relevance of a given document to a user's query and 
        provide a detailed summary if it's relevant."""

        user_prompt = f"""
        Query: {query}

        Document Title: {document['title']}
        Document Content:
        {document['content'][:1000]}

        Instructions:
        1. Assess if the document is relevant to the QUERY made by the user.
        2. If relevant, provide a detailed summary that captures the unique aspects of this particular news item. Include:
           - Key facts and figures
           - Dates of events or announcements
           - Names of important entities mentioned
           - Any metrics or changes reported
           - The potential impact or significance of the news
        3. If not relevant, simply state "Not relevant".

        Your response should be in the following format:
        Relevant: [Yes/No]
        Summary: [Your detailed summary if relevant, or "Not relevant" if not]
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = ai_model.generate_response(
                messages=messages,
                max_tokens=300,
                temperature=temperature
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error assessing relevance and summarizing with LLM: {e}")
            return "Error: Unable to assess relevance and summarize"
