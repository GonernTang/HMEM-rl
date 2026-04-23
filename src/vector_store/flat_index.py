import numpy as np
from typing import Tuple, List


class FlatIndex:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = {}

    def add(self, embedding: np.ndarray, id: int):
        # Add embeddings into the index
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[0]}"
            )
        normalized_embedding = self.normalize_L2(embedding)
        self.index[id] = normalized_embedding

    def search(
        self, q_embedding: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Return scores and ids
        q_embedding = self.normalize_L2(q_embedding)

        if q_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {q_embedding.shape[0]}"
            )

        scores = []
        ids = []

        # Calculate cosine similarity with all stored embeddings
        for id, stored_embedding in self.index.items():
            # Cosine similarity = dot product of normalized vectors
            similarity = np.dot(q_embedding, stored_embedding)
            scores.append(similarity)
            ids.append(id)

        # Convert to numpy arrays
        scores = np.array(scores)
        ids = np.array(ids)

        # Sort by similarity (descending order)
        sorted_indices = np.argsort(scores)[::-1]

        # Return top_k results
        top_k = min(top_k, len(scores))
        return scores[sorted_indices[:top_k]], ids[sorted_indices[:top_k]]

    def remove(self, id: List[int]):
        # Remove embeddings from the index by id
        for single_id in id:
            if single_id in self.index:
                del self.index[single_id]

    def normalize_L2(self, embedding: np.ndarray) -> np.ndarray:
        return embedding / np.linalg.norm(embedding)

    def reset(self):
        self.index = {}