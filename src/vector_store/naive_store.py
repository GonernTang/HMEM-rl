import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: Change to ChromaDB, note that it only supports HNSW index.
class NaiveStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index: Dict[int, np.ndarray] = {}
        self.payload_mapping: Dict[int, str] = {}

    def add(self, embedding: np.ndarray, payload: str, id: int):
        # Add embeddings into the index
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[0]}"
            )
        normalized_embedding = self.normalize_L2(embedding)
        if id in self.index:
            logger.warning(f"Overwriting embedding for id {id}")
        self.index[id] = normalized_embedding
        if id in self.payload_mapping:
            logger.warning(f"Overwriting payload for id {id}")
        self.payload_mapping[id] = payload

    def list_memories(self) -> List[str]:
        return list(self.payload_mapping.values())
        
    def search(
        self, q_embedding: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        # Return scores and ids
        q_embedding = self.normalize_L2(q_embedding)

        if q_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {q_embedding.shape[0]}"
            )

        scores = []
        ids = []
        # Calculate cosine similarity with all stored embeddings
        # TODO: This can be optimized using numpy batch computation
        for id, stored_embedding in self.index.items():
            # Cosine similarity = dot product of normalized vectors
            similarity = np.dot(q_embedding, stored_embedding)
            scores.append(similarity)
            ids.append(id)

        # Convert to numpy arrays
        scores = np.array(scores)
        ids = np.array(ids)

        # Sort by similarity (descending order)
        sorted_indices = np.argsort(scores)[::-1][:top_k]

        # Return top_k results
        top_k = min(top_k, len(scores))
        return (
            scores[sorted_indices[:top_k]],
            ids[sorted_indices[:top_k]],
            [self.payload_mapping[id] for id in ids[sorted_indices[:top_k]]],
        )

    def remove(self, id: List[int]):
        # Remove embeddings from the index by id
        for single_id in id:
            if single_id in self.index and single_id in self.payload_mapping:
                del self.payload_mapping[single_id]
                del self.index[single_id]
            else:
                logger.warning(
                    f"Id {single_id} not found in the index or payload mapping, which should not happen"
                )

    def normalize_L2(self, embedding: np.ndarray) -> np.ndarray:
        return embedding / np.linalg.norm(embedding)

    def reset(self):
        self.index = {}
        self.payload_mapping = {}