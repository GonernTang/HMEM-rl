"""
HMEMS Conversation Memory State

Manages memory state for a single session copy during per-turn training.
Each batch contains multiple copies of the same session, each with independent memory state.

Key design:
- batch_size=4 means 4 copies of the same session, each with its own memory state
- Memory state includes vec_store (raw memories) and episodic_store (consolidated memories)
- Embeddings are stored alongside memories for retrieval
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


class ConversationMemoryState:
    """
    Manages memory state for a single session copy.

    Maintains:
    - vec_store: FlatIndex for raw conversation memories
    - episodic_store: Dict[int, Dict] for episodic memories
    - payload_mapping: id -> text (for vec store retrieval)
    - id_assigner: counter for assigning memory IDs
    """

    # Default embedding dim is 1536 for text-embedding-3-small
    # But text-embedding-v4 returns 1024 dimensions
    DEFAULT_EMBEDDING_DIM = 1024  # text-embedding-v4 uses 1024

    def __init__(self, embedding_dim: int = None):
        from src.vector_store.flat_index import FlatIndex

        if embedding_dim is None:
            embedding_dim = self.DEFAULT_EMBEDDING_DIM

        self.vec_store = FlatIndex(embedding_dim=embedding_dim)
        self.episodic_store: Dict[int, Dict[str, Any]] = {}  # id -> {content, embedding}
        self.payload_mapping: Dict[int, str] = {}  # vec_id -> text
        self.id_assigner = 0
        self.embedding_dim = embedding_dim

    def retrieve(
        self,
        new_memory: str,
        embedding: np.ndarray,
        vec_top_k: int = 5,
        episodic_top_k: int = 5
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieve similar memories from both stores.

        Args:
            new_memory: The new memory text (not used in simple embedding search)
            embedding: Query embedding vector
            vec_top_k: Number of top vec memories to return
            episodic_top_k: Number of top episodic memories to return

        Returns:
            (episodic_results, vec_results)
            Each is a list of {"id": int, "content": str, "score": float}
        """
        # 1. Search vec store
        vec_scores, vec_ids = self.vec_store.search(embedding, vec_top_k)
        vec_results = []
        for vid, score in zip(vec_ids, vec_scores):
            if vid in self.payload_mapping:
                vec_results.append({
                    "id": int(vid),
                    "content": self.payload_mapping[vid],
                    "score": float(score)
                })

        # 2. Search episodic store (using embedding similarity)
        episodic_results = []
        for eid, edata in self.episodic_store.items():
            e_embedding = edata.get("embedding")
            if e_embedding is not None:
                sim = np.dot(embedding, e_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(e_embedding) + 1e-8
                )
                if sim > 0.3:  # Lower threshold for episodic
                    episodic_results.append({
                        "id": int(eid),
                        "content": edata["content"],
                        "score": float(sim)
                    })

        episodic_results.sort(key=lambda x: x["score"], reverse=True)
        episodic_results = episodic_results[:episodic_top_k]

        return episodic_results, vec_results

    def add_vec_memory(self, text: str, embedding: np.ndarray) -> int:
        """Add a raw memory to vec store."""
        memory_id = self.id_assigner
        self.vec_store.add(embedding, memory_id)
        self.payload_mapping[memory_id] = text
        self.id_assigner += 1
        return memory_id

    def add_episodic_memory(self, content: str, embedding: Optional[np.ndarray] = None) -> int:
        """Add an episodic memory."""
        memory_id = self.id_assigner
        self.episodic_store[memory_id] = {
            "content": content,
            "embedding": embedding if embedding is not None else np.random.randn(self.embedding_dim)
        }
        self.id_assigner += 1
        return memory_id

    def update_episodic_memory(self, memory_id: int, content: str) -> bool:
        """Update an existing episodic memory."""
        if memory_id in self.episodic_store:
            self.episodic_store[memory_id]["content"] = content
            return True
        return False

    def execute_action(
        self,
        decision: Dict[str, Any],
        new_memory: str,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Execute a consolidation decision and update memory state.

        Args:
            decision: Dict with action type and content
            new_memory: The new memory text
            embedding: Optional embedding for the new memory

        Returns:
            action taken
        """
        action = decision.get("action", "none")
        embedding = embedding if embedding is not None else np.random.randn(self.embedding_dim)

        if action == "merge":
            target_id = decision.get("target_episodic_id", 0)
            merged_content = decision.get("merged_content", "")

            if not merged_content:
                # Use existing content if no merged content provided
                if target_id in self.episodic_store:
                    merged_content = f"{self.episodic_store[target_id]['content']}\n{new_memory}"
                else:
                    merged_content = new_memory

            if target_id in self.episodic_store:
                self.episodic_store[target_id]["content"] = merged_content
            else:
                self.add_episodic_memory(merged_content, embedding)

        elif action == "augment":
            augmented_content = decision.get("augmented_content", "")

            if not augmented_content:
                # Simple augmentation: concatenate new memory with relevant vec memories
                relevant_contents = []
                for eid, edata in self.episodic_store.items():
                    relevant_contents.append(edata["content"])
                augmented_content = " | ".join(relevant_contents + [new_memory])

            self.add_episodic_memory(augmented_content, embedding)

        else:  # "none"
            self.add_vec_memory(new_memory, embedding)

        return action

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of current memory state."""
        return {
            "num_vec_memories": len(self.payload_mapping),
            "num_episodic_memories": len(self.episodic_store),
            "total_memories": self.id_assigner
        }

    def reset(self):
        """Reset all memory state."""
        self.vec_store.reset()
        self.episodic_store = {}
        self.payload_mapping = {}
        self.id_assigner = 0

    def __repr__(self) -> str:
        return f"ConversationMemoryState(vec={len(self.payload_mapping)}, episodic={len(self.episodic_store)})"