import numpy as np
from typing import List, Dict

class RAGEvaluator:
    """Evaluator for RAG retrieval metrics"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], 
                       relevant: List[str], 
                       k: int) -> float:

        top_k = retrieved[:k]
        print(top_k)
        relevant_in_top_k = len([doc for doc in top_k if doc in relevant])
        return relevant_in_top_k / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[str], 
                    relevant: List[str], 
                    k: int) -> float:
        
        top_k = retrieved[:k]
        relevant_in_top_k = len([doc for doc in top_k if doc in relevant])
        return relevant_in_top_k / len(relevant) if relevant else 0.0
    
    @staticmethod
    def f1_at_k(retrieved: List[str], 
                relevant: List[str], 
                k: int) -> float:
        """Calculate F1@K (harmonic mean of P and R)"""
        precision = RAGEvaluator.precision_at_k(retrieved, relevant, k)
        recall = RAGEvaluator.recall_at_k(retrieved, relevant, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], 
                            relevant: List[str]) -> float:
        """Calculate MRR (Mean Reciprocal Rank)"""
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int) -> float:
        """Calculate Discounted Cumulative Gain at K"""
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            dcg += relevance_scores[i] / np.log2(i + 2)
        return dcg
    
    @staticmethod
    def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
        """Calculate Normalized DCG at K"""
        dcg = RAGEvaluator.dcg_at_k(relevance_scores, k)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = RAGEvaluator.dcg_at_k(ideal_scores, k)
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_at_k(queries_results: List[Dict], k: int) -> float:
        """Calculate Mean Average Precision at K"""
        avg_precisions = []
        for result in queries_results:
            retrieved = result['retrieved'][:k]
            relevant = result['relevant']
            
            relevant_count = 0
            precision_sum = 0.0
            
            for i, doc in enumerate(retrieved):
                if doc in relevant:
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)
            
            avg_precision = precision_sum / len(relevant) if relevant else 0.0
            avg_precisions.append(avg_precision)
        
        return np.mean(avg_precisions) if avg_precisions else 0.0


# Example usage
if __name__ == "__main__":
    # Single query evaluation
    retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant_docs = ["doc1", "doc3", "doc5", "doc7"]
    k = 5
    
    evaluator = RAGEvaluator()
    
    print(f"Precision@{k}: {evaluator.precision_at_k(retrieved_docs, relevant_docs, k):.3f}")
    print(f"Recall@{k}: {evaluator.recall_at_k(retrieved_docs, relevant_docs, k):.3f}")
    print(f"F1@{k}: {evaluator.f1_at_k(retrieved_docs, relevant_docs, k):.3f}")
    print(f"MRR: {evaluator.mean_reciprocal_rank(retrieved_docs, relevant_docs):.3f}")
    
    # NDCG with relevance scores (0-3 scale)
    relevance_scores = [3, 0, 2, 0, 1]  # scores for retrieved docs
    print(f"NDCG@{k}: {evaluator.ndcg_at_k(relevance_scores, k):.3f}")
    
    # Multi-query evaluation (MAP)
    queries = [
        {
            'retrieved': ["doc1", "doc2", "doc3", "doc4", "doc5"],
            'relevant': ["doc1", "doc3", "doc5"]
        },
        {
            'retrieved': ["doc10", "doc11", "doc12", "doc13"],
            'relevant': ["doc10", "doc11"]
        }
    ]
    print(f"MAP@{k}: {evaluator.map_at_k(queries, k):.3f}")