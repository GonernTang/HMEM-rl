from src.vector_store.naive_store import NaiveStore
from mem0.embeddings.openai import OpenAIEmbedding
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SemanticMemory:
    """
    SemanticMemory is a class that stores and searches for semantic memories.
    It uses a flat vector store to store the memories.
    Each memory piece is expected to be a concise string of a piece of factual information.
    """

    def __init__(self, embedding_dim: int):
        self.mem_index = NaiveStore(embedding_dim)
        self.mem_id_counter = 0
        self.embedder = OpenAIEmbedding()

    def add_memory(self, memory: str):
        embedding: np.ndarray = np.array(self.embedder.embed(memory))
        self.mem_index.add(embedding, memory, self.mem_id_counter)
        self.mem_id_counter += 1

    def add_memories(self, memories: list[str]):
        for memory in memories:
            self.add_memory(memory)

    def search_memory(self, query: str, topk: int, threshold: float = 0):
        embedding: np.ndarray = np.array(self.embedder.embed(query))
        scores, ids, memories = self.mem_index.search(embedding, topk)
        if threshold == 0:
            return memories
        else:
            return [
                memory for memory, score in zip(memories, scores) if score > threshold
            ]

    def list_memories(self) -> list[str]:
        return self.mem_index.list_memories()

    def reset(self):
        self.mem_index.reset()
        self.mem_id_counter = 0