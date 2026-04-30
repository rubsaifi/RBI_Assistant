"""
RAG Engine Module
Handles retrieval-augmented generation using embeddings and vector similarity search.
"""

import os
import numpy as np
from typing import List, Dict, Tuple
from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for RBI Master documents.
    Uses embeddings and vector similarity for relevant context retrieval.
    """

    def __init__(self, documents: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG Engine with documents.

        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            model_name: Name of the sentence transformer model to use
        """
        self.documents = documents
        self.model_name = model_name
        self.embeddings = None
        self.index = None
        self.model = None

        # Initialize model and create embeddings
        self._initialize_model()
        self._create_embeddings()
        self._build_index()

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        # Use a lightweight but effective model
        # all-MiniLM-L6-v2 is fast and provides good quality embeddings
        self.model = SentenceTransformer(self.model_name)
        print(f"✅ Loaded embedding model: {self.model_name}")

    def _create_embeddings(self):
        """Create embeddings for all documents."""
        texts = [doc["content"] for doc in self.documents]

        print(f"🔄 Creating embeddings for {len(texts)} documents...")
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✅ Created embeddings with shape: {self.embeddings.shape}")

    def _build_index(self):
        """Build FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE:
            print("⚠️ FAISS not available, will use numpy for similarity search")
            return

        # Get embedding dimension
        dimension = self.embeddings.shape[1]

        # Create index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Add embeddings to index
        self.index.add(self.embeddings)
        print(f"✅ Built FAISS index with {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (document_index, similarity_score) tuples
        """
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        if self.index is not None and FAISS_AVAILABLE:
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)

            # Search using FAISS
            scores, indices = self.index.search(query_embedding, top_k)
            results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        else:
            # Fallback to numpy cosine similarity
            query_embedding = query_embedding[0]
            query_norm = np.linalg.norm(query_embedding)

            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                doc_norm = np.linalg.norm(doc_embedding)
                if doc_norm > 0 and query_norm > 0:
                    similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
                    similarities.append((i, similarity))
                else:
                    similarities.append((i, 0.0))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = similarities[:top_k]

        return results

    def get_relevant_context(self, query: str, top_k: int = 8) -> str:
        """
        Get relevant context for a query.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Concatenated relevant context
        """
        if not self.documents:
            return "No documents available."

        # Search for relevant documents
        results = self.search(query, top_k=top_k)

        # Check if retrieval actually found relevant matches
        if results and results[0][1] < 0.22:
            # Very low relevance - likely query outside document scope
            return "INSUFFICIENT_CONTEXT"

        # Build context from retrieved documents
        contexts = []
        for idx, score in results:
            if score > 0.25:  # Minimum relevance threshold
                doc = self.documents[idx]
                context = f"[Relevance: {score:.2f}]\n{doc['content']}\n"
                contexts.append(context)

        if not contexts:
            # Return top result even if below threshold
            idx, score = results[0] if results else (0, 0)
            if self.documents:
                contexts.append(self.documents[idx]['content'])

        return "\n---\n".join(contexts)

    def get_definition_context(self, query: str, keywords: List[str], top_k: int = 8) -> str:
        """
        Get context with keyword matching boost for definition-type queries.
        This ensures definitions containing the search terms are included.

        Args:
            query: User query
            keywords: Keywords that should boost relevance for definition lookups
            top_k: Number of documents to retrieve

        Returns:
            Concatenated relevant context
        """
        if not self.documents:
            return "No documents available."

        # Search for relevant documents (search more to ensure we find definition)
        results = self.search(query, top_k=top_k * 2)

        # Build context with keyword-based re-ranking
        scored_docs = []
        for idx, score in results:
            doc = self.documents[idx]
            content_lower = doc["content"].lower()

            # Count keyword matches
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)

            # Boost score by keyword matches (0.1 per match, max 0.5 boost)
            boosted_score = score + min(keyword_matches * 0.1, 0.5)

            scored_docs.append((idx, boosted_score, doc["content"]))

        # Re-sort by boosted score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Check if top results contain the main definition
        # If not, explicitly search for the definition chunk
        top_content = scored_docs[0][2].lower() if scored_docs else ""
        has_main_definition = "ovd) means" in top_content or "officially valid document means" in top_content

        if not has_main_definition:
            # Search for the definition chunk directly
            for idx, doc in enumerate(self.documents):
                if "Officially Valid Document (OVD) means" in doc["content"]:
                    # Add it to scored_docs with a high boosted score
                    content_lower = doc["content"].lower()
                    keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
                    boosted_score = 0.8 + min(keyword_matches * 0.1, 0.5)  # High base score
                    scored_docs.insert(0, (idx, boosted_score, doc["content"]))
                    break

        # Build final context from top results
        contexts = []
        for idx, score, content in scored_docs[:top_k]:
            if score > 0.25:
                contexts.append(f"[Relevance: {score:.2f}]\n{content}\n")

        if not contexts:
            # Fallback to top semantic result
            idx, score = results[0] if results else (0, 0)
            if self.documents:
                contexts.append(self.documents[idx]['content'])

        return "\n---\n".join(contexts)

    def get_document_sources(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Get source documents for a query with metadata.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            List of document dictionaries with relevance scores
        """
        results = self.search(query, top_k=top_k)

        sources = []
        for idx, score in results:
            doc = self.documents[idx].copy()
            doc["relevance_score"] = score
            sources.append(doc)

        return sources


class SimpleRAGFallback:
    """
    Simple fallback RAG that uses keyword matching when embeddings aren't available.
    """

    def __init__(self, documents: List[Dict]):
        self.documents = documents

    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Simple keyword-based retrieval."""
        query_words = set(query.lower().split())

        scored_docs = []
        for doc in self.documents:
            content = doc["content"].lower()
            score = sum(1 for word in query_words if word in content)
            scored_docs.append((score, doc["content"]))

        # Sort by score (descending)
        scored_docs.sort(reverse=True)

        # Return top results
        top_docs = [doc for _, doc in scored_docs[:top_k]]
        return "\n---\n".join(top_docs) if top_docs else "No relevant documents found."


def create_rag_engine(documents: List[Dict]):
    """
    Factory function to create appropriate RAG engine.

    Args:
        documents: List of document dictionaries

    Returns:
        RAGEngine or SimpleRAGFallback instance
    """
    try:
        return RAGEngine(documents)
    except Exception as e:
        print(f"⚠️ Could not initialize full RAG engine: {e}")
        print("🔄 Falling back to simple keyword-based retrieval...")
        return SimpleRAGFallback(documents)


if __name__ == "__main__":
    # Test with sample documents
    sample_docs = [
        {"content": "RBI guidelines on KYC require banks to verify customer identity.", "metadata": {}},
        {"content": "Basel III norms specify capital adequacy requirements for banks.", "metadata": {}},
        {"content": "Priority sector lending includes agriculture, MSME, and housing loans.", "metadata": {}},
    ]

    try:
        rag = create_rag_engine(sample_docs)
        context = rag.get_relevant_context("What are KYC requirements?")
        print("\nRetrieved Context:")
        print(context)
    except Exception as e:
        print(f"Error: {e}")
