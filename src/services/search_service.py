import logging
import json
from typing import List, Dict, Any
import requests
from urllib.parse import urlparse
import certifi
from .ranking_service import RankingService
from ..utils.web_utils import (
    requests_retry_session,
    is_valid_url,
    scrape_full_content,
)
from .assessment_service import AssessmentService
from ..models.ai_models import AIModel
from ..config.settings import SEARXNG_URL

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.session = requests_retry_session()
        self.ranking_service = RankingService()
        self.assessment_service = AssessmentService()

    def perform_search(
        self,
        query: str,
        num_results: int = 10,
        time_range: str = "",
        language: str = "en",
        category: str = "general",
        engines: List[str] = None,
        safesearch: int = 2,
        method: str = "GET",
        site_filter: str = "",
    ) -> List[Dict]:
        if engines is None:
            engines = ['google']

        query = f"{f'site:{site_filter} ' if site_filter.strip() else ''}{query}"
        logger.info(f"Query: {query}")
        
        params = {
            'q': query,
            'format': 'json',
            'time_range': time_range,
            'language': language,
            'category': category,
            'engines': ','.join(engines),
            'safesearch': safesearch
        }
        
        # Remove empty parameters
        params = {k: v for k, v in params.items() if v != ""}

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': 'https://shreyas094-searxng-local.hf.space',
            'Referer': 'https://shreyas094-searxng-local.hf.space/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }

        scraped_content = []
        page = 1

        while len(scraped_content) < num_results:
            params['pageno'] = page
            
            try:
                if method.upper() == "GET":
                    response = self.session.get(SEARXNG_URL, params=params, headers=headers, timeout=10, verify=certifi.where())
                else:  # POST
                    response = self.session.post(SEARXNG_URL, data=params, headers=headers, timeout=10, verify=certifi.where())
                
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Error during SearXNG request: {e}")
                break

            search_results = response.json()
            results = search_results.get('results', [])
            
            if not results:
                break

            for result in results:
                if len(scraped_content) >= num_results:
                    break

                url = result.get('url', '')
                title = result.get('title', 'No title')

                if not is_valid_url(url):
                    continue

                try:
                    content = scrape_full_content(url)
                    if content:
                        scraped_content.append({
                            "title": title,
                            "url": url,
                            "content": content,
                            "scraper": "pdf" if url.lower().endswith('.pdf') else "newspaper"
                        })
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")

            page += 1

        return scraped_content

    def search_and_process(
        self,
        query: str,
        chat_history: str,
        ai_model: AIModel,
        num_results: int = 10,
        max_chars: int = 1500,
        time_range: str = "",
        language: str = "en",
        category: str = "general",
        engines: List[str] = None,
        safesearch: int = 2,
        method: str = "GET",
        llm_temperature: float = 0.2,
        use_pydf2: bool = True,
        site_filter: str = "",
    ) -> str:
        try:
            all_search_results = []
            attempts = 0
            max_attempts = 3

            query = self.assessment_service.rephrase_query_simple(ai_model, query, chat_history, llm_temperature)

            while len(all_search_results) < num_results and attempts < max_attempts:
                logger.info(f"Attempt {attempts + 1} of {max_attempts}")
                logger.info(f"Query: {query} with site_filter: {site_filter}")
                # Perform search and get results
                search_results = self.perform_search(
                    query=query,
                    num_results=num_results - len(all_search_results),
                    time_range=time_range,
                    language=language,
                    category=category,
                    engines=engines,
                    safesearch=safesearch,
                    method=method,
                    site_filter=site_filter
                )

                if not search_results:
                    break

                # Assess relevance and summarize search results
                assessed_results = []
                for result in search_results:
                    logger.info(f"Assessing result: {result}")
                    assessment = self.assessment_service.assess_relevance_and_summarize(ai_model, query, result, temperature=llm_temperature)
                    if "Relevant: Yes" in assessment:
                        # summary = assessment.split("Summary: ", 1)[1].strip()
                        # result['summary'] = summary
                        assessed_results.append(result)

                logger.info(f"Assessed results: {assessed_results}")

                all_search_results.extend(assessed_results)
                attempts += 1

                if len(all_search_results) < num_results:
                    # Modify the query for the next attempt
                    query = self.assessment_service.rephrase_query_simple(ai_model, query, chat_history, llm_temperature)

            if not all_search_results:
                return "No relevant content could be found for your query."

            # Rank the assessed results
            ranked_results = self.ranking_service.rerank_documents(query, all_search_results)


            # Limit to the top num_results
            all_search_results = ranked_results[:num_results]

            # Prepare JSON for LLM
            llm_input = {
                "query": query,
                "documents": all_search_results
            }

            # Generate response using AI model
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Please provide a comprehensive summary of the search results. Include the source of the content, url and title in the summary."},
                {"role": "user", "content": json.dumps(llm_input)}
            ]

            logger.info(f"LLM input: {messages}")

            response = ai_model.generate_response(
                messages=messages,
                max_tokens=1000,
                temperature=llm_temperature
            )

            return response

        except Exception as e:
            logger.error(f"Error in search_and_process: {e}")
            return f"An error occurred while processing your request: {e}"
