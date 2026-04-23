from typing import List, Tuple, Optional, Dict
from src.vector_store.naive_store import NaiveStore
from src.episodic_memory import EpisodicNote
from src.prompt import MEMORY_AUGMENT_PROMPT, MEMORY_AUGMENT_MERGE_PROMPT
from openai import OpenAI
from src.token_monitor import TokenMonitor, OPType
import os
import json
from mem0.embeddings.openai import OpenAIEmbedding
from src.aug_methods.aug_config import AugConfig
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NaiveAugMem:
    def __init__(self, aug_config: AugConfig):
        self.vec_store: NaiveStore = NaiveStore(embedding_dim=1536)
        self.openai_client = OpenAI(base_url=os.getenv("M1_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
        self.embeder = OpenAIEmbedding()
        self.id_counter = 0
        self.monitor: TokenMonitor = None
        self.set_config(aug_config)

        # Episodic Memory Store. Note that it stores meta only (The raw reference)
        # Mapping from id to EpisodicNote
        self.raw_store: Dict[int, EpisodicNote] = {}

    def _safe_extract_json(self, response_content: str, context: str = "") -> dict:
        """
        Safely extract JSON from OpenAI response content.
        Handles markdown code blocks and None values.

        Args:
            response_content: The response content from OpenAI API
            context: Context description for error logging

        Returns:
            Parsed JSON as dictionary, or empty dict if parsing fails
        """
        if not response_content:
            logger.error(f"Empty response content in {context}")
            return {}

        content = response_content.strip()

        # Remove markdown code blocks if present (e.g., ```json ... ```)
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last line (the ``` markers)
            if len(lines) > 2:
                # Also remove the language identifier if present (e.g., ```json)
                content = "\n".join(lines[1:-1])
            else:
                logger.warning(f"Malformed code block in {context}: {content}")

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parsing failed in {context}. Error: {e}\n"
                f"Position: line {e.lineno}, column {e.colno}\n"
                f"Raw content (first 500 chars): {response_content[:500]}"
            )
            return {}

    def add(self, conversation: str, id: int, raw_ids: List[str]) -> List[str]:
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL1"),
            messages=[
                {"role": "system", "content": MEMORY_AUGMENT_PROMPT},
                {"role": "user", "content": conversation},
            ],
            temperature=0.0,
        )
        if self.monitor is not None:
            self.monitor.record_usage_from_raw(response.usage, OPType.MEMORY_AUGMENT)

        content = response.choices[0].message.content
        response_json = self._safe_extract_json(content, context="add_memory")
        if not response_json:
            # Failed to parse JSON, return empty list
            logger.error(
                f"Failed to parse response for conversation: {conversation[:100]}..."
            )
            return []

        facts = response_json.get("memories", [])

        if len(facts) == 0 and self.ask_coordinator:
            logger.warning(f"No facts found for conversation {conversation}")
            return []  # TODO: VecMem and Augment diverge, consult the coordinator

        episodic_memories: List[str] = []
        for fact in facts:
            episodic_memories.append(fact)
            self.vec_store.add(
                np.array(self.embeder.embed(fact)), fact, self.id_counter
            )
            self.raw_store[self.id_counter] = EpisodicNote(conversation, raw_ids)
            self.id_counter += 1

        return episodic_memories

    def get_episodic_note(self, id: int) -> EpisodicNote:
        return self.raw_store[id]

    def search_only(
        self, query: str, limit: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        query_embedding = self.embeder.embed(query)
        topk = limit or self.search_top_k
        scores, _, payloads = self.vec_store.search(query_embedding, topk)
        return [(payload, score) for payload, score in zip(payloads, scores)]

    def _trigger_recheck(self):
        raise NotImplementedError("NaiveAugMem does not support recheck")

    def reset(self):
        self.id_counter = 0
        self.vec_store.reset()

    def consult_coordinator(self, message: str):
        raise NotImplementedError("NaiveAugMem does not support coordinator")

    def set_config(self, aug_config: AugConfig):
        self.aug_config = aug_config
        self.ask_coordinator = aug_config.ask_coordinator
        self.search_top_k = aug_config.search_top_k
        self.reset()

    def set_monitor(self, monitor: TokenMonitor):
        self.monitor = monitor

    # Return True if it is merged in, False otherwise
    def try_merge_new_memory(
        self, new_memory: str, new_memory_embedding: np.ndarray, threshold: float
    ) -> Tuple[bool, str]:
        # Search from the augmented memory
        scores, ids, payloads = self.vec_store.search(new_memory_embedding, 1)
        if len(scores) == 0:
            return False, ""
        if scores[0] < threshold:
            return False, ""
        past_memory = payloads[0]
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL1"),
            messages=[
                {"role": "system", "content": MEMORY_AUGMENT_MERGE_PROMPT},
                {
                    "role": "user",
                    "content": f"<New memory>: {new_memory}\n<Past memory>: {past_memory}",
                },
            ],
            temperature=0.0,
        )

        if self.monitor is not None:
            self.monitor.record_usage_from_raw(response.usage, OPType.MEMORY_MERGE)

        content = response.choices[0].message.content
        response_json = self._safe_extract_json(content, context="try_merge_new_memory")
        if not response_json:
            # Failed to parse JSON, don't merge
            logger.warning(
                f"Failed to parse merge response for new memory: {new_memory[:100]}..."
            )
            return False, ""

        should_merge = response_json.get("should_merge", "no")
        if should_merge == "yes":
            new_mem = response_json.get("merged_memory")
            if not new_mem:
                logger.warning(f"Merge decision is 'yes' but no merged_memory provided")
                return False, ""

            try:
                # Use new mem to replace the past memory
                self.vec_store.remove(ids)
                self.vec_store.add(
                    np.array(self.embeder.embed(new_mem)), new_mem, self.id_counter
                )
                self.id_counter += 1
            except Exception as e:
                logger.warning(f"Error updating vector store during merge: {e}")
                return False, ""
        else:
            return False, ""

        return True, new_mem