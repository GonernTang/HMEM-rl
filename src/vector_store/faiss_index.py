#!/usr/bin/env python3
"""
FAISS-based vector index for fast similarity search
"""

import logging
import numpy as np
import faiss
from typing import Tuple

# Setup logging
logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS-based vector index for fast similarity search"""
    
    def __init__(self, embedding_dim: int, index_type: str = "IVF"):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ("Flat", "IVF", "HNSW")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.is_trained = False
        
    def build_index(self, embeddings: np.ndarray, nlist: int = 100):
        """Build FAISS index from embeddings"""
        logger.info(f"Building {self.index_type} index with {len(embeddings)} vectors")
        
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif self.index_type == "IVFPQ":
            # IVFPQ: Product Quantization for compressed storage
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            m = 16  # number of subquantizers (can be tuned)
            nbits = 8  # bits per subvector (can be tuned)
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, nbits)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Train index if needed
        if not self.index.is_trained:
            logger.info("Training index...")
            self.index.train(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        self.is_trained = True
        
        logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")
    
    def search(self, query_embeddings: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors"""
        if not self.is_trained:
            raise RuntimeError("Index not trained. Call build_index first.")
        
        # Normalize query embeddings
        query_embeddings = query_embeddings.astype('float32')
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = self.index.search(query_embeddings, top_k)
        return scores, indices
    
    def save(self, file_path: str):
        """Save index to disk"""
        if self.index is not None:
            faiss.write_index(self.index, file_path)
            logger.info(f"Index saved to {file_path}")
    
    def load(self, file_path: str):
        """Load index from disk"""
        self.index = faiss.read_index(file_path)
        self.is_trained = True
        logger.info(f"Index loaded from {file_path}")