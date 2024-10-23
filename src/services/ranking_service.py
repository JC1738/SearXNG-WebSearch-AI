import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)

class RankingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def compute_similarity(self, text1: str, text2: str) -> float:
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_similarity.item()

    def is_content_unique(self, new_content: str, existing_contents: List[str], 
                         similarity_threshold: float = 0.8) -> bool:
        for existing_content in existing_contents:
            similarity = self.compute_similarity(new_content, existing_content)
            if similarity > similarity_threshold:
                return False
        return True

    def rerank_documents(self, query: str, documents: List[Dict], 
                        similarity_threshold: float = 0.95, 
                        max_results: int = 3) -> List[Dict]:
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        query_embedding = self.model.encode([query], convert_to_tensor=True)
        doc_embeddings = self.model.encode(
            [doc.get('summary', '') for doc in documents], 
            convert_to_tensor=True
        )

        similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        
        ranked_docs = sorted(
            zip(documents, similarities),
            key=lambda x: (x[1], x[0].get('is_entity_domain', False)),
            reverse=True
        )

        filtered_docs = [
            doc for doc, sim in ranked_docs 
            if sim >= similarity_threshold
        ]

        if not filtered_docs:
            filtered_docs = [doc for doc, _ in ranked_docs[:max_results]]

        return filtered_docs[:max_results]
