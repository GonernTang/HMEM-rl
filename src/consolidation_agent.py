"""
HMEMS Consolidation Agent Implementation

This module implements the consolidation decision logic for the HMEMS memory system.
The agent observes new conversation and relevant memories, then decides whether to:
- merge: merge with existing episodic memory
- augment: create new episodic from relevant vec memories
- none: store as raw memory
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import numpy as np


@dataclass
class ConsolidationAction:
    """Represents a consolidation decision."""
    action: str  # "merge", "augment", or "none"
    target_episodic_id: Optional[int] = None  # For merge
    merged_content: Optional[str] = None  # For merge result
    relevant_vec_ids: Optional[List[int]] = None  # For augment
    augmented_content: Optional[str] = None  # For augment result


class ConsolidationTool:
    """Tool interface for memory consolidation operations."""

    def merge(self, new_memory: str, target_episodic_id: int, past_episodic_content: str) -> str:
        """
        Merge new memory with existing episodic memory.

        Returns:
            Merged episodic memory content
        """
        # This would call the actual LLM-based merge in full implementation
        # For now, return combined content
        return f"{past_episodic_content}\n{new_memory}"

    def augment(self, new_memory: str, relevant_memories: List[str]) -> str:
        """
        Augment: create new episodic memory from new + relevant vec memories.

        Returns:
            Augmented episodic memory content
        """
        # This would call the actual LLM-based augmentation in full implementation
        combined = relevant_memories + [new_memory]
        return " | ".join(combined)

    def store_raw(self, memory: str) -> Dict[str, Any]:
        """
        Store raw memory without consolidation.

        Returns:
            Storage result with memory id
        """
        return {"status": "stored", "memory": memory}


class VecStoreInterface:
    """
    Interface to the vector store for retrieval.
    In full implementation, this would wrap FlatIndex or FAISSIndex.
    """

    def __init__(self):
        self.memories = {}  # id -> content
        self.embeddings = {}  # id -> embedding vector

    def add(self, memory_id: int, content: str, embedding: np.ndarray):
        """Add a memory to the store."""
        self.memories[memory_id] = content
        self.embeddings[memory_id] = embedding

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[List[float], List[int], List[str]]:
        """
        Search for similar memories.

        Returns:
            (scores, ids, contents)
        """
        # Simplified: return all memories for testing
        if not self.memories:
            return [], [], []
        ids = list(self.memories.keys())
        contents = [self.memories[i] for i in ids]
        scores = [0.8] * len(ids)  # Placeholder
        return scores[:top_k], ids[:top_k], contents[:top_k]

    def remove(self, ids: List[int]):
        """Remove memories by id."""
        for i in ids:
            self.memories.pop(i, None)
            self.embeddings.pop(i, None)

    def reset(self):
        """Reset the store."""
        self.memories.clear()
        self.embeddings.clear()


class EpisodicStoreInterface:
    """
    Interface to the episodic memory store.
    In full implementation, this would wrap NaiveAugMem.
    """

    def __init__(self):
        self.episodic_memories = {}  # id -> content
        self.id_counter = 0

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Search episodic memories by text similarity.

        Returns:
            List of (content, score) tuples
        """
        if not self.episodic_memories:
            return []
        results = [(v, 0.8) for v in self.episodic_memories.values()]
        return results[:top_k]

    def add(self, episodic_id: int, content: str):
        """Add an episodic memory."""
        self.episodic_memories[episodic_id] = content

    def update(self, episodic_id: int, content: str):
        """Update an episodic memory."""
        self.episodic_memories[episodic_id] = content

    def remove(self, episodic_id: int):
        """Remove an episodic memory."""
        self.episodic_memories.pop(episodic_id, None)

    def reset(self):
        """Reset the store."""
        self.episodic_memories.clear()
        self.id_counter = 0


class ConsolidationAgent:
    """
    The consolidation decision agent.

    In full RL implementation, this would:
    1. Receive state (new_memory + retrieval results)
    2. Output action via LLM
    3. Framework executes tool
    4. Receive reward based on downstream QA
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.vec_store = VecStoreInterface()
        self.episodic_store = EpisodicStoreInterface()
        self.tool = ConsolidationTool()

        # Retrieval thresholds
        self.min_relevant_score = self.config.get("min_relevant_score", 0.7)
        self.min_aug_count = self.config.get("min_aug_count", 3)
        self.merge_threshold = self.config.get("merge_threshold", 0.85)

    def retrieve_context(self, new_memory: str, embedding: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieve relevant memories from both stores.

        Returns:
            (episodic_results, vec_results)
            Each is a list of {"id": int, "content": str, "score": float}
        """
        # Search episodic
        episodic_raw = self.episodic_store.search(new_memory, top_k=5)
        episodic_results = [
            {"content": content, "score": score}
            for content, score in episodic_raw
        ]

        # Search vec store
        scores, ids, contents = self.vec_store.search(embedding, top_k=5)
        vec_results = [
            {"id": id_, "content": content, "score": score}
            for id_, content, score in zip(ids, contents, scores)
        ]

        return episodic_results, vec_results

    def decide_action(
        self,
        new_memory: str,
        embedding: np.ndarray,
        episodic_results: List[Dict],
        vec_results: List[Dict],
    ) -> ConsolidationAction:
        """
        Decide consolidation action based on retrieved context.

        In RL training, this is replaced by LLM decision.
        This rule-based version is for baseline/testing.
        """
        # Check episodic similarity for merge
        if episodic_results and episodic_results[0]["score"] >= self.merge_threshold:
            return ConsolidationAction(
                action="merge",
                target_episodic_id=0,  # Would be actual id
                merged_content=episodic_results[0]["content"],
            )

        # Check vec similarity for augment
        relevant_vecs = [v for v in vec_results if v["score"] >= self.min_relevant_score]
        if len(relevant_vecs) >= self.min_aug_count:
            return ConsolidationAction(
                action="augment",
                relevant_vec_ids=[v["id"] for v in relevant_vecs],
                augmented_content=" | ".join([v["content"] for v in relevant_vecs] + [new_memory]),
            )

        # Default: store raw
        return ConsolidationAction(action="none")

    def execute_action(self, action: ConsolidationAction, new_memory: str) -> Dict[str, Any]:
        """Execute the consolidation action."""
        if action.action == "merge":
            past_content = action.merged_content or ""
            merged = self.tool.merge(new_memory, action.target_episodic_id, past_content)
            return {"status": "merged", "content": merged}

        elif action.action == "augment":
            relevant_contents = action.augmented_content or new_memory
            augmented = self.tool.augment(new_memory, [relevant_contents])
            return {"status": "augmented", "content": augmented}

        elif action.action == "none":
            return self.tool.store_raw(new_memory)

        return {"status": "error", "message": "Unknown action"}

    def reset(self):
        """Reset agent state."""
        self.vec_store.reset()
        self.episodic_store.reset()
