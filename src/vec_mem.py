import json
import os
import sys

import yaml

# from functions import get_memory_tool_schemas
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import mem0
from mem0.embeddings.openai import OpenAIEmbedding
from src.vector_store.flat_index import FlatIndex
from src.aug_methods.aug_config import AugConfig
from src.aug_methods.naive_aug import NaiveAugMem
from src.episodic_memory import EpisodicNote
from src.semantic_memory import SemanticMemory
from typing import List, Dict, Tuple, Optional, Any
from conv_loader import *
import numpy as np
from embed_manager import *
from dataclasses import dataclass
from tqdm import tqdm
from jinja2 import Template
from prompt import (
    ANSWER_PROMPT_VECMEM,
    ITERATIVE_ANWSER_PROMPT_VECMEM,
    SEMANTIC_EXTRACTION_PROMPT,
    ANSWER_PROMPT_WITH_SEMANTIC,
    SEMANTIC_EXTRACTION_DURING_MERGE_PROMPT,
)
from src.agent import MemoryAgent
import json
import os
from openai import OpenAI
from src.token_monitor import TokenMonitor, OPType
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class VecMemConfig:
    # General Setting
    min_aug_count: int = 3
    min_relevant_score: float = 0.7
    merge_with_aug_thresh: float = 0.85

    # Retrieve phase topk
    retrieve_raw_topk: int = 5
    retrieve_aug_topk: int = 5

    # Iterative answer help
    enable_iter_anwser: bool = False
    iterative_raw_topk: int = 3
    iterative_aug_topk: int = 3
    iter_max_depth: int = 3

    # conv_limit: Sample the first n conversations for testing
    conv_limit: Optional[int] = None

    # Semantic Memory
    enable_semantic_memory: bool = False
    semantic_memory_topk: int = 5
    semantic_memory_threshold: float = 0.0


class VecMem:
    def __init__(self, config: VecMemConfig):
        # Initialize memory
        self.embedder = OpenAIEmbedding()
        self.vec_store = FlatIndex(embedding_dim=1536)
        self.min_aug_count = config.min_aug_count
        self.min_relevant_score = config.min_relevant_score
        self.enable_iter_anwser = config.enable_iter_anwser
        self.iterative_raw_topk = config.iterative_raw_topk
        self.iterative_aug_topk = config.iterative_aug_topk
        self.iter_max_depth = config.iter_max_depth
        self.id_assigner = 0
        self.retrieve_raw_topk = config.retrieve_raw_topk
        self.retrieve_aug_topk = config.retrieve_aug_topk
        self.merge_with_aug_thresh = config.merge_with_aug_thresh
        self.conv_limit = config.conv_limit
        self._payload_mapping = {}
        self.openai_client1 = OpenAI(base_url="M1_BASE_URL")
        self.openai_client2 = OpenAI(base_url="M2_BASE_URL")
        self.token_monitor: Optional[TokenMonitor] = None

        # Semantic Memory Part
        self.enable_semantic_memory = config.enable_semantic_memory
        self.semantic_memory_topk = config.semantic_memory_topk
        self.semantic_memory_threshold = config.semantic_memory_threshold
        if self.enable_semantic_memory:
            self.semantic_memory = SemanticMemory(embedding_dim=1536)

        # Augment Memory
        self.aug_mem = NaiveAugMem(
            aug_config=AugConfig(
                recheck_freq=10,
                augment_sim_threshold=0.6,
                search_top_k=config.retrieve_aug_topk,
                ask_coordinator=False,
            )
        )
    
    def _block(self, title: str = '', lines=None, content: str = None) -> str:
        """辅助：把 memory list 或单字符串格式化成块文本。"""
        if content is not None:
            return f"<{title}>\n{content}\n</{title}>" if title else content

        if lines is None or len(lines) == 0:
            return f"<{title}>\nEmpty.\n</{title}>" if title else "Empty."

        formatted = []
        # lines 可能是 list of dicts ({id: text}) 或 list of (id,text) tuples 或 list of strings
        for item in lines:
            if isinstance(item, dict):
                for k, v in item.items():
                    formatted.append(f"[{k}] {v}")
            elif isinstance(item, tuple) and len(item) >= 2:
                formatted.append(f"[{item[0]}] {item[1]}")
            else:
                formatted.append(str(item))
        body = "\n".join(formatted)
        return f"<{title}>\n{body}\n</{title}>" if title else body





    def enable_token_monitoring(self):
        self.token_monitor = TokenMonitor()
        self.aug_mem.set_monitor(self.token_monitor)

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
            log.error(f"Empty response content in {context}")
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
                log.warning(f"Malformed code block in {context}: {content}")

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            log.error(
                f"JSON parsing failed in {context}. Error: {e}\n"
                f"Position: line {e.lineno}, column {e.colno}\n"
                f"Raw content (first 500 chars): {response_content[:500]}"
            )
            return {}

    def process_all_conversations(self, data_path: str, memory_agent_template, output_file_path: str):
        with open('config/prompts_datasource.yaml', 'r') as f:
            prompts_datasource = yaml.safe_load(f)
        
        
        print(f"Iterative anwser: {self.enable_iter_anwser}")

        convs, questions = ConvLoader.load_locomo(data_path)
        if self.conv_limit is not None:
            convs = convs[: self.conv_limit]
            questions = questions[: self.conv_limit]
        FINAL_RESULTS = {}
        
        stats_file_path = output_file_path.replace(".json", "_token_stats.json")
        os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)

        for conv_idx in tqdm(range(len(convs)), desc="Processing conversations"):
            if self.token_monitor is not None:
                self.token_monitor.start_conversation(conv_idx)

            chat_history: Conversation = convs[conv_idx]
            conv_questions: List[Question] = questions[conv_idx]
            conv_embedding: np.ndarray = np.load(
                os.path.join(os.getenv("LOCOMO_EMBEDDING_PATH"), f"conv_{conv_idx}.npy")
            )
            question_embedding: np.ndarray = np.load(
                os.path.join(
                    os.getenv("LOCOMO_EMBEDDING_PATH"), f"question_{conv_idx}.npy"
                )
            )
            conv_mapping: Dict[int, str] = EmbedManager.construct_mapping(chat_history)

            # Init to an empty state
            self.vec_store.reset()
            self.aug_mem.reset()
            self._payload_mapping = {}
            self.id_assigner = 0

            # Initialize conversation results
            FINAL_RESULTS[str(conv_idx)] = []

            prompts = []
            # Adding memory phase:
            for i in tqdm(
                range(len(conv_embedding)), desc="Adding memories", leave=False
            ):
                processed_text = MemoryAgent.process_text_with_qwen_pipeline(
                        text=chat_history[i].content[i],
                        tokenizer=memory_agent_template.tokenizer,
                        functions=[tool["function"] for tool in get_memory_tool_schemas(self)],
                        status='memorie',
                        enable_thinking=memory_agent_template.agent_config['enable_thinking'],
                        return_text=True,
                        memory=self
                    )
                # self.add_memory(conv_embedding[i], conv_mapping[i])
            # Retrieving memory phase:
            for q_idx, item in tqdm(
                enumerate(conv_questions), desc="Answering questions", leave=False
            ):
                question: str = item.content
                answer: str = item.anwser
                category: int = item.category
                response: str = self.anwser_question(
                    question, question_embedding[q_idx]
                )
                FINAL_RESULTS[str(conv_idx)].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "response": response,
                    }
                )
            if self.token_monitor is not None:
                self.token_monitor.end_conversation(
                    len(conv_questions), 0, len(self._payload_mapping)
                )

        # Write all results at once in the correct format
        with open(output_file_path, "w") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
        if self.token_monitor is not None:
            self.token_monitor.save_stats(stats_file_path)

    def iterative_anwser(self, question: str, question_embedding: np.ndarray):
        vec_mem: List[str] = []
        aug_mem: List[str] = []
        asked_questions: List[str] = []
        semantic_memories: List[str] = []
        vec_memories, aug_memories = self.search_memory(question, question_embedding)
        if self.enable_semantic_memory:
            semantic_memories: List[str] = self.semantic_memory.search_memory(
                question, self.semantic_memory_topk, self.semantic_memory_threshold
            )
        vec_mem.extend([mem["memory"] for mem in vec_memories])
        aug_mem.extend([mem["memory"] for mem in aug_memories])
        for i in range(self.iter_max_depth):
            memories: List[str] = vec_mem + aug_mem
            if self.enable_semantic_memory:
                memories.extend(semantic_memories)
            template = Template(ITERATIVE_ANWSER_PROMPT_VECMEM)
            answer_prompt = template.render(
                memories=json.dumps(memories, indent=4),
                question=question,
                asked_questions=json.dumps(asked_questions, indent=4),
            )
            response = self.openai_client2.chat.completions.create(
                model=os.getenv("MODEL2"),
                messages=[{"role": "system", "content": answer_prompt}],
                temperature=0.0,
            )
            if self.token_monitor is not None:
                self.token_monitor.record_usage_from_raw(
                    response.usage, OPType.ITERATIVE_FILTER
                )

            response_json = self._safe_extract_json(
                response.choices[0].message.content, context="iterative_anwser"
            )
            if not response_json:
                # Failed to parse JSON, return current memories
                logger.warning(
                    f"Failed to parse iterative anwser response: {response.choices[0].message.content}"
                )
                return vec_mem, aug_mem, semantic_memories

            if response_json.get("sufficient") == "no":
                new_questions = response_json.get("search_query")
                for new_question in new_questions:
                    new_question_embedding = np.array(self.embedder.embed(new_question))
                    new_vec_memories, new_aug_memories = self.search_memory(
                        new_question,
                        new_question_embedding,
                        raw_topk=self.iterative_raw_topk,
                        aug_topk=self.iterative_aug_topk,
                    )
                    vec_mem.extend([mem["memory"] for mem in new_vec_memories])
                    aug_mem.extend([mem["memory"] for mem in new_aug_memories])
                    if self.enable_semantic_memory:
                        semantic_memories.extend(
                            self.semantic_memory.search_memory(
                                new_question,
                                self.iterative_aug_topk,
                                self.semantic_memory_threshold,
                            )
                        )
                asked_questions.extend(new_questions)
            else:
                return vec_mem, aug_mem, semantic_memories

        return vec_mem, aug_mem, semantic_memories

    def anwser_question(self, question: str, question_embedding: np.ndarray) -> str:
        if self.enable_iter_anwser:
            vec_memories, aug_memories, semantic_memories = self.iterative_anwser(
                question, question_embedding
            )
        else:
            vec_memories, aug_memories = self.search_memory(
                question, question_embedding
            )
            vec_memories = [mem["memory"] for mem in vec_memories]
            aug_memories = [mem["memory"] for mem in aug_memories]
            if self.enable_semantic_memory:
                semantic_memories: List[str] = self.semantic_memory.search_memory(
                    question, self.semantic_memory_topk, self.semantic_memory_threshold
                )

        if self.enable_semantic_memory:
            template = Template(ANSWER_PROMPT_WITH_SEMANTIC)
            answer_prompt = template.render(
                episodic_memories=json.dumps(aug_memories, indent=4),
                semantic_memories=json.dumps(semantic_memories, indent=4),
                raw_memories=json.dumps(vec_memories, indent=4),
                question=question,
            )
        else:
            template = Template(ANSWER_PROMPT_VECMEM)
            answer_prompt = template.render(
                episodic_memories=json.dumps(vec_memories, indent=4),
                raw_memories=json.dumps(aug_memories, indent=4),
                question=question,
            )
        response = self.openai_client2.chat.completions.create(
            model=os.getenv("MODEL2"),
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0,
        )
        if self.token_monitor:
            self.token_monitor.record_usage_from_raw(response.usage, OPType.ANSWER)
        return response.choices[0].message.content.strip()

    def _memory_augment(
        self,
        mem_ids: List[int],
        mem_scores: List[float],
        new_memory: Optional[str] = None,
        new_memory_embedding: Optional[np.ndarray] = None,
    ):
        # Find the vectors above the threshold
        relevant_vec_ids: List[int] = []
        for i, score in enumerate(mem_scores):
            if score > self.min_relevant_score:
                relevant_vec_ids.append(mem_ids[i])

        # Update counters for all similar vectors and new memory
        relevant_count: int = len(relevant_vec_ids)

        if new_memory:
            relevant_count += 1

        if relevant_count >= self.min_aug_count:
            raw_memories: List[str] = [
                self._payload_mapping[vec_id] for vec_id in relevant_vec_ids
            ]
            if new_memory:
                raw_memories.append(new_memory)
            # Batch Updates.
            cat_raw_memories: str = "<END_OF_CONV>".join(raw_memories)
            # TODO: Add reference id for recall testing
            episodic_memories: List[str] = self.aug_mem.add(
                cat_raw_memories, self.id_assigner, [""]
            )

            # Update Semantic Memory Store
            if self.enable_semantic_memory:
                for episodic_memory in episodic_memories:
                    semantic_memories: List[str] = self._generate_semantic_memories(
                        episodic_memory, cat_raw_memories, is_new_episodic=True
                    )
                    self.semantic_memory.add_memories(semantic_memories)

            # Remove the augmented memories from vec
            self.vec_store.remove(np.array(relevant_vec_ids))
            for vec_id in relevant_vec_ids:
                del self._payload_mapping[vec_id]
        elif new_memory:
            # No memories to augment, but new memory is in
            new_memory_id = self.id_assigner
            self.vec_store.add(new_memory_embedding, new_memory_id)
            self._payload_mapping[new_memory_id] = new_memory
            self.id_assigner += 1




    def add_memory(self, conv_embedding: np.ndarray, conv: str):
        """
        Add memory to the system using LLM-based consolidation decision.

        This replaces the hardcoded threshold-based logic with LLM reasoning
        to decide between merge/augment/none actions.
        """
        if self.token_monitor is not None:
            self.token_monitor.record_message()

        # Step 1: Retrieve memories from both stores
        # Search episodic memories (aug_mem) - returns (content, score)
        aug_raw_results = self.aug_mem.search_only(conv, limit=self.retrieve_aug_topk)
        # Convert to (id, content, score) - use index as id since episodic doesn't have stable IDs
        aug_results: List[Tuple[int, str, float]] = [
            (i, content, float(score)) for i, (content, score) in enumerate(aug_raw_results)
        ]

        # Search vector memories (flat store)
        vec_scores, vec_indices = self.vec_store.search(
            conv_embedding, self.retrieve_raw_topk
        )
        vec_results: List[Tuple[int, str, float]] = []
        for score, vec_id in zip(vec_scores, vec_indices):
            if vec_id in self._payload_mapping:
                vec_results.append((int(vec_id), self._payload_mapping[vec_id], float(score)))

        # Step 2: Let LLM decide consolidation action
        decision = self.aug_mem.decide_consolidation_action(
            new_memory=conv,
            new_memory_embedding=conv_embedding,
            retrieved_episodic_memories=aug_results,
            retrieved_vec_memories=vec_results,
        )

        action = decision.get("action", "none")

        # Step 3: Execute the decided action
        if action == "merge":
            target_id = decision.get("target_episodic_id", 0)
            merged_content = decision.get("merged_content", "")
            if merged_content:
                # Update the episodic memory with merged content
                try:
                    past_note = self.aug_mem.get_episodic_note(target_id)
                    if past_note:
                        # Remove old episodic
                        self.aug_mem.vec_store.remove([target_id])
                        # Add merged memory
                        self.aug_mem.vec_store.add(
                            np.array(self.embeder.embed(merged_content)),
                            merged_content,
                            self.aug_mem.id_counter
                        )
                        self.aug_mem.raw_store[self.aug_mem.id_counter] = EpisodicNote(
                            merged_content, [target_id]
                        )
                        self.aug_mem.id_counter += 1
                        # Generate semantic memories
                        if self.enable_semantic_memory:
                            semantic_memories = self._generate_semantic_memories(
                                merged_content, conv, is_new_episodic=False
                            )
                            self.semantic_memory.add_memories(semantic_memories)
                        logger.info(f"Merged memory into episodic {target_id}")
                except Exception as e:
                    logger.warning(f"Error during merge execution: {e}")
                    # Fall through to store as raw
                    action = "none"

        elif action == "augment":
            relevant_ids = decision.get("relevant_vec_ids", [])
            augmented_content = decision.get("augmented_content", "")
            if augmented_content and relevant_ids:
                try:
                    # Add new episodic memory
                    self.aug_mem.vec_store.add(
                        np.array(self.embeder.embed(augmented_content)),
                        augmented_content,
                        self.aug_mem.id_counter
                    )
                    self.aug_mem.raw_store[self.aug_mem.id_counter] = EpisodicNote(
                        augmented_content, relevant_ids
                    )
                    self.aug_mem.id_counter += 1
                    # Remove consumed vector memories
                    self.vec_store.remove(np.array(relevant_ids))
                    for vec_id in relevant_ids:
                        if vec_id in self._payload_mapping:
                            del self._payload_mapping[vec_id]
                    # Generate semantic memories
                    if self.enable_semantic_memory:
                        semantic_memories = self._generate_semantic_memories(
                            augmented_content, conv, is_new_episodic=True
                        )
                        self.semantic_memory.add_memories(semantic_memories)
                    logger.info(f"Augmented new episodic with {len(relevant_ids)} raw memories")
                except Exception as e:
                    logger.warning(f"Error during augment execution: {e}")
                    # Fall through to store as raw
                    action = "none"

        # Step 4: Default - store as raw vector memory
        if action == "none":
            new_memory_id = self.id_assigner
            self.vec_store.add(conv_embedding, new_memory_id)
            self._payload_mapping[new_memory_id] = conv
            self.id_assigner += 1
            logger.info(f"Stored new memory as raw vector")

    def _generate_semantic_memories(
        self, aug_memory: str, raw_conv: str, is_new_episodic: bool
    ) -> List[str]:
        # Step 1 : Try to get the facts in old semantic memories
        relevant_facts = self.semantic_memory.search_memory(
            aug_memory, self.semantic_memory_topk, self.semantic_memory_threshold
        )
        # Step 2: Try to distill more related facts about this augmented memory
        # Need to let augmented memory also provide the raw reference back
        if is_new_episodic:
            # Prompt will guide the LLM not to extract information that is already included in the episodic memory
            # or semantic memories.
            template = Template(SEMANTIC_EXTRACTION_PROMPT)
            op_type = OPType.SEMANTIC_EXTRACTION
        else:
            # Prompt will guide the LLM to extract the knowledge that is not included in the semantic memories.
            template = Template(SEMANTIC_EXTRACTION_DURING_MERGE_PROMPT)
            op_type = OPType.SEMANTIC_EXTRACTION_DURING_MERGE
        extraction_prompt = template.render(
            episodic_memory=aug_memory,
            raw_reference=raw_conv,
            old_semantic_memories=relevant_facts,
        )
        response = self.openai_client1.chat.completions.create(
            model=os.getenv("MODEL1"),
            messages=[{"role": "system", "content": extraction_prompt}],
        )
        if self.token_monitor:
            self.token_monitor.record_usage_from_raw(response.usage, op_type)

        response_json = self._safe_extract_json(
            response.choices[0].message.content, context="_generate_semantic_memories"
        )
        if not response_json:
            # Failed to parse JSON, return empty list
            return []

        try:
            res = response_json.get("facts")
            return res
        except Exception as e:
            logger.error(f"Error parsing response: {e}, no 'facts' found")
            return []

    def search_memory(
        self,
        query: str,
        query_embedding: np.ndarray,
        raw_topk: Optional[int] = None,
        aug_topk: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Search memories from both vector store and augmem.
        Returns combined results sorted by relevance.
        """
        if raw_topk is None:
            raw_topk = self.retrieve_raw_topk
        if aug_topk is None:
            aug_topk = self.retrieve_aug_topk
        vec_memories = []
        aug_memories = []
        # Search from vector store
        scores_local, indices_local = self.vec_store.search(query_embedding, raw_topk)

        for i, (score, vec_id) in enumerate(zip(scores_local, indices_local)):
            if vec_id in self._payload_mapping:
                vec_memories.append(
                    {
                        "memory": self._payload_mapping[vec_id],
                        "score": float(score),
                        # "id": int(vec_id),  # Convert numpy int64 to Python int
                    }
                )

        # Search from augmented memory
        aug_mem: List[Tuple[str, float]] = self.aug_mem.search_only(
            query, limit=aug_topk
        )

        for mem_item in aug_mem:
            aug_memories.append(
                {
                    "memory": mem_item[0],
                    "score": float(mem_item[1]),
                }
            )

        # Sort by score descending and return top_k
        # TODO: Test query time augmentation if possible
        # self._memory_augment(indices_local, scores_local, None, None)
        return vec_memories, aug_memories

