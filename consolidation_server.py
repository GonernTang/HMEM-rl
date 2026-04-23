#!/usr/bin/env python3
"""
HMEMS Consolidation Agent Server

This server handles memory consolidation decisions from the RL agent:
1. Receives new memory + retrieval context
2. Executes consolidation actions (merge/augment/none)
3. Returns results for reward computation

API Endpoints:
- POST /consolidate: Process consolidation decision
- POST /execute_tool: Execute consolidation tool call (real VecMem)
- POST /batch_process: Process batch of memories and questions (for QA evaluation)
- GET /get_consolidation_tools: Return tool schemas for RL agent
- GET /health: Health check
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from flask import Flask, request, jsonify
from openai import OpenAI
from transformers import AutoTokenizer
import dotenv
import numpy as np

# Load environment variables
dotenv.load_dotenv()

# Import consolidation tools
from consolidation_tools import get_consolidation_tool_schemas, TOOL_IMPLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# OpenAI client for QA evaluation
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None
)

# Embedding client for retrieval
try:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    embedding_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None
    )
    def get_embedding(text: str) -> np.ndarray:
        """Get embedding for text using OpenAI embedding model."""
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding)
except Exception as e:
    logger.warning(f"Failed to initialize embedding client: {e}")
    embedding_client = None
    def get_embedding(text: str) -> np.ndarray:
        return np.random.randn(1536)  # fallback random embedding

# QA model configuration
QA_MODEL = os.getenv("QA_MODEL", "gpt-4o-mini")

# Retrieval settings
RETRIEVE_TOPK_VEC = int(os.getenv("RETRIEVE_TOPK_VEC", "5"))
RETRIEVE_TOPK_EPISODIC = int(os.getenv("RETRIEVE_TOPK_EPISODIC", "5"))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Handle dimension mismatch by using minimum dimension
    min_dim = min(len(a), len(b))
    a_trunc = a[:min_dim]
    b_trunc = b[:min_dim]
    return np.dot(a_trunc, b_trunc) / (np.linalg.norm(a_trunc) * np.linalg.norm(b_trunc) + 1e-8)

def retrieve_relevant_memories(question: str, topk_vec: int = 5, topk_episodic: int = 5) -> Dict[str, List[str]]:
    """
    Retrieve relevant memories based on question embedding.

    Args:
        question: The question to retrieve memories for
        topk_vec: Maximum number of vector memories to retrieve
        topk_episodic: Maximum number of episodic memories to retrieve

    Returns:
        Dict with relevant vec_memories and episodic_memories
    """
    if embedding_client is None:
        # Fallback: return all memories if embedding not available
        return {
            "vec_memories": [v["content"] for v in consolidation_store.vec_store.values()],
            "episodic_memories": [e["content"] for e in consolidation_store.episodic_store.values()],
        }

    # Get question embedding
    try:
        question_emb = get_embedding(question)
    except Exception as e:
        logger.warning(f"Failed to get question embedding: {e}")
        return {
            "vec_memories": [],
            "episodic_memories": [],
        }

    # Retrieve vector memories
    vec_scores = []
    for vid, vmem in consolidation_store.vec_store.items():
        mem_emb = vmem.get("embedding")
        if mem_emb is not None:
            score = cosine_similarity(question_emb, mem_emb)
            vec_scores.append((score, vid, vmem["content"]))

    # Retrieve episodic memories
    epi_scores = []
    for eid, emem in consolidation_store.episodic_store.items():
        mem_emb = emem.get("embedding")
        if mem_emb is not None:
            score = cosine_similarity(question_emb, mem_emb)
            epi_scores.append((score, eid, emem["content"]))

    # Sort by score descending and take top-k
    vec_scores.sort(reverse=True)
    epi_scores.sort(reverse=True)

    relevant_vec = [content for score, vid, content in vec_scores[:topk_vec]]
    relevant_episodic = [content for score, eid, content in epi_scores[:topk_episodic]]

    logger.info(f"Retrieved {len(relevant_vec)} vec memories, {len(relevant_episodic)} episodic memories")

    return {
        "vec_memories": relevant_vec,
        "episodic_memories": relevant_episodic,
    }

# Store actual embedding dimension from first embedding call
_EMBEDDING_DIM = None

def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text using OpenAI embedding model."""
    global _EMBEDDING_DIM
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    embedding = np.array(response.data[0].embedding)
    if _EMBEDDING_DIM is None:
        _EMBEDDING_DIM = len(embedding)
        logger.info(f"Detected embedding dimension: {_EMBEDDING_DIM}")
    return embedding
try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from src.vec_mem import VecMem, VecMemConfig
    from src.aug_methods.naive_aug import NaiveAugMem
    HAS_VECMEM = True
except ImportError as e:
    logger.warning(f"Could not import VecMem: {e}. Using mock store.")
    HAS_VECMEM = False


@dataclass
class ConsolidationResult:
    """Result of a consolidation action."""
    action: str  # "merge", "augment", "none"
    status: str  # "success", "error"
    memory_id: Optional[int] = None
    content: Optional[str] = None
    merged_content: Optional[str] = None
    augmented_content: Optional[str] = None
    memory_length: int = 0
    original_length: int = 0
    error: Optional[str] = None


class ConsolidationStore:
    """
    In-memory storage for consolidation agent state.

    Maintains:
    - vec_store: Raw conversation memories
    - episodic_store: Consolidated episodic memories
    """

    def __init__(self):
        self.vec_store: Dict[int, Dict[str, Any]] = {}  # id -> {content, embedding}
        self.episodic_store: Dict[int, Dict[str, Any]] = {}  # id -> {content, embedding}
        self.vec_id_counter = 0
        self.episodic_id_counter = 0
        self.embedding_dim = 1536  # Default for OpenAI embeddings

    def add_vec_memory(self, content: str, embedding: Optional[np.ndarray] = None) -> int:
        """Add a raw memory to vec store."""
        memory_id = self.vec_id_counter
        self.vec_store[memory_id] = {
            "content": content,
            "embedding": embedding if embedding is not None else np.random.randn(self.embedding_dim)
        }
        self.vec_id_counter += 1
        return memory_id

    def add_episodic_memory(self, content: str, embedding: Optional[np.ndarray] = None) -> int:
        """Add an episodic memory."""
        memory_id = self.episodic_id_counter
        self.episodic_store[memory_id] = {
            "content": content,
            "embedding": embedding if embedding is not None else np.random.randn(self.embedding_dim)
        }
        self.episodic_id_counter += 1
        return memory_id

    def update_episodic_memory(self, memory_id: int, content: str) -> bool:
        """Update an existing episodic memory."""
        if memory_id in self.episodic_store:
            self.episodic_store[memory_id]["content"] = content
            return True
        return False

    def remove_vec_memories(self, memory_ids: List[int]) -> None:
        """Remove vec memories by id."""
        for mid in memory_ids:
            self.vec_store.pop(mid, None)

    def remove_episodic_memory(self, memory_id: int) -> None:
        """Remove an episodic memory."""
        self.episodic_store.pop(memory_id, None)

    def search_vec(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Search vec store by embedding similarity."""
        if not self.vec_store:
            return []

        results = []
        for mid, data in self.vec_store.items():
            # Cosine similarity (simplified - dot product since embeddings are normalized)
            embedding = data["embedding"]
            if embedding is not None:
                sim = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
                )
            else:
                sim = 0.5  # Default similarity
            results.append((mid, float(sim), data["content"]))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_episodic(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Search episodic store by embedding similarity."""
        if not self.episodic_store:
            return []

        results = []
        for mid, data in self.episodic_store.items():
            embedding = data["embedding"]
            if embedding is not None:
                sim = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
                )
            else:
                sim = 0.5
            results.append((mid, float(sim), data["content"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def reset(self) -> None:
        """Reset all stores."""
        self.vec_store.clear()
        self.episodic_store.clear()
        self.vec_id_counter = 0
        self.episodic_id_counter = 0


# Global consolidation store
# Use real VecMem if available, otherwise fallback to ConsolidationStore
if HAS_VECMEM:
    vecmem_config = VecMemConfig(
        min_aug_count=3,
        min_relevant_score=0.7,
        merge_with_aug_thresh=0.85,
        retrieve_raw_topk=5,
        retrieve_aug_topk=5,
    )
    vecmem = VecMem(vecmem_config)
    consolidation_store = vecmem  # Alias for compatibility
    logger.info("Using real VecMem for consolidation")
else:
    consolidation_store = ConsolidationStore()
    logger.info("Using mock ConsolidationStore (VecMem not available)")


class ConsolidationTool:
    """
    Tool for executing consolidation actions.
    In a full implementation, this would call LLM for actual merge/augment.
    """

    def __init__(self, client: Optional[OpenAI] = None, model_name: str = None):
        self.client = client
        self.model_name = model_name or os.getenv("MODEL1", "qwen3-32b")

    def merge(self, new_memory: str, target_episodic_id: int, past_episodic_content: str) -> str:
        """
        Merge new memory with existing episodic memory.

        In full implementation, this would call LLM to produce merged content.
        For now, simple concatenation.
        """
        # Simple merge: append new memory to existing
        merged = f"{past_episodic_content}\n{new_memory}"
        return merged

    def augment(self, new_memory: str, relevant_vec_contents: List[str]) -> str:
        """
        Create new episodic memory from new + relevant vec memories.

        In full implementation, this would call LLM for summarization.
        For now, simple join.
        """
        all_contents = relevant_vec_contents + [new_memory]
        augmented = " | ".join(all_contents)
        return augmented


# Global tool instance
consolidation_tool = ConsolidationTool()


def parse_consolidation_action(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse consolidation action from LLM response.

    Expected JSON format:
    {
        "action": "merge" | "augment" | "none",
        "target_episodic_id": int (for merge),
        "relevant_vec_ids": [int] (for augment),
        "merged_content": str (for merge),
        "augmented_content": str (for augment)
    }
    """
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            action_dict = json.loads(json_match.group())
            if 'action' in action_dict and action_dict['action'] in ['merge', 'augment', 'none']:
                return action_dict
    except json.JSONDecodeError:
        pass

    # Default to 'none' if parsing fails
    return {'action': 'none'}


def execute_consolidation_action(
    action_dict: Dict[str, Any],
    new_memory: str,
    original_length: int
) -> ConsolidationResult:
    """
    Execute a consolidation action based on parsed action dict.

    Args:
        action_dict: Parsed action from LLM
        new_memory: New conversation turn
        original_length: Length of original content (prev_context + new_memory)

    Returns:
        ConsolidationResult with execution details
    """
    action = action_dict.get('action', 'none')

    if action == 'merge':
        target_id = action_dict.get('target_episodic_id', 0)
        merged_content = action_dict.get('merged_content', '')

        # Get past episodic content
        past_content = ""
        if target_id in consolidation_store.episodic_store:
            past_content = consolidation_store.episodic_store[target_id]["content"]

        if not merged_content:
            merged_content = consolidation_tool.merge(new_memory, target_id, past_content)

        # Update episodic memory
        consolidation_store.update_episodic_memory(target_id, merged_content)

        return ConsolidationResult(
            action="merge",
            status="success",
            memory_id=target_id,
            content=merged_content,
            merged_content=merged_content,
            memory_length=len(merged_content),
            original_length=original_length
        )

    elif action == 'augment':
        relevant_ids = action_dict.get('relevant_vec_ids', [])
        augmented_content = action_dict.get('augmented_content', '')

        # Get relevant vec contents
        relevant_contents = []
        for vid in relevant_ids:
            if vid in consolidation_store.vec_store:
                relevant_contents.append(consolidation_store.vec_store[vid]["content"])

        if not augmented_content:
            augmented_content = consolidation_tool.augment(new_memory, relevant_contents)

        # Add as new episodic memory
        new_episodic_id = consolidation_store.add_episodic_memory(augmented_content)

        # Remove consumed vec memories
        consolidation_store.remove_vec_memories(relevant_ids)

        return ConsolidationResult(
            action="augment",
            status="success",
            memory_id=new_episodic_id,
            content=augmented_content,
            augmented_content=augmented_content,
            memory_length=len(augmented_content),
            original_length=original_length
        )

    else:  # 'none'
        # Store as raw vec memory
        memory_id = consolidation_store.add_vec_memory(new_memory)

        return ConsolidationResult(
            action="none",
            status="success",
            memory_id=memory_id,
            content=new_memory,
            memory_length=len(new_memory),
            original_length=original_length
        )


@app.route('/consolidate', methods=['POST'])
def consolidate():
    """
    Process consolidation decisions from the agent.

    Supports TWO formats:

    Format 1 (hmems_generation.py compatible):
    {
        "decisions": [
            {
                "action": "merge|augment|none",
                "memory_content": "string - the consolidated memory content",
                "source_session": "string - session identifier"
            },
            ...
        ]
    }

    Returns:
    {
        "status": "ok",
        "stored_count": int,
        "stored_ids": [string],
        "total_episodic": int,
        "total_vector": int
    }

    Format 2 (legacy):
    {
        "new_memory": "string - new conversation turn",
        "prev_context": "string - previous conversation context",
        "action": "string - merge|augment|none",
        ...
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Check for hmems_generation format first
        if "decisions" in data:
            decisions = data.get("decisions", [])
            stored_ids = []
            total_episodic = len(consolidation_store.episodic_store)
            total_vector = len(consolidation_store.vec_store)

            for decision in decisions:
                action = decision.get("action", "none")
                memory_content = decision.get("memory_content", "")
                source = decision.get("source_session", "")

                if not memory_content:
                    continue

                if action == "merge":
                    consolidation_store.add_episodic_memory(memory_content)
                    stored_ids.append(f"merged: {memory_content[:50]}...")
                elif action == "augment":
                    consolidation_store.add_episodic_memory(memory_content)
                    stored_ids.append(f"augmented: {memory_content[:50]}...")
                else:  # "none"
                    consolidation_store.add_vec_memory(memory_content)
                    stored_ids.append(f"vector: {memory_content[:50]}...")

            total_episodic = len(consolidation_store.episodic_store)
            total_vector = len(consolidation_store.vec_store)

            logger.info(f"Received {len(decisions)} consolidation decisions, stored {len(stored_ids)} memories")

            return jsonify({
                "status": "ok",
                "stored_count": len(stored_ids),
                "stored_ids": stored_ids,
                "total_episodic": total_episodic,
                "total_vector": total_vector
            })

        # Legacy format
        new_memory = data.get('new_memory', '')
        prev_context = data.get('prev_context', '')
        original_length = len(prev_context) + len(new_memory)

        # Parse action from request
        action_dict = {
            'action': data.get('action', 'none'),
            'target_episodic_id': data.get('target_episodic_id'),
            'relevant_vec_ids': data.get('relevant_vec_ids', []),
            'merged_content': data.get('merged_content'),
            'augmented_content': data.get('augmented_content'),
        }

        # Validate action
        if action_dict['action'] not in ['merge', 'augment', 'none']:
            action_dict['action'] = 'none'

        # Execute action
        result = execute_consolidation_action(action_dict, new_memory, original_length)

        return jsonify({
            "status": result.status,
            "action": result.action,
            "memory_id": result.memory_id,
            "content": result.content,
            "memory_length": result.memory_length,
            "original_length": result.original_length,
            "error": result.error
        })

    except Exception as e:
        logger.error(f"Error in /consolidate: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


def format_memories_for_prompt(memory_data: Dict[str, Any]) -> str:
    """Format memories into a prompt context string."""
    lines = []

    vec_memories = memory_data.get("vec_memories", [])
    if vec_memories:
        lines.append("=== 原始记忆 (Raw Memories) ===")
        for i, mem in enumerate(vec_memories):
            lines.append(f"[{i}] {mem}")

    episodic_memories = memory_data.get("episodic_memories", [])
    if episodic_memories:
        lines.append("\n=== 情节记忆 (Episodic Memories) ===")
        for i, mem in enumerate(episodic_memories):
            lines.append(f"[{i}] {mem}")

    semantic_memories = memory_data.get("semantic_memories", [])
    if semantic_memories:
        lines.append("\n=== 语义记忆 (Semantic Memories) ===")
        for i, mem in enumerate(semantic_memories):
            lines.append(f"[{i}] {mem}")

    return "\n".join(lines) if lines else "（无记忆）"


def generate_answer_via_openai(question: str, memory_data: Dict[str, Any]) -> str:
    """
    Generate answer using OpenAI API based on memory context.

    Args:
        question: The question to answer
        memory_data: Dict with vec_memories, episodic_memories, semantic_memories

    Returns:
        Generated answer string
    """
    context = format_memories_for_prompt(memory_data)

    system_prompt = """你是一个基于记忆回答问题的AI助手。
根据提供的记忆内容回答问题。
如果记忆中有相关信息，请基于那些信息给出准确的回答。
如果记忆中没有相关信息，请回答"我没有足够的信息来回答这个问题"。"""

    response = openai_client.chat.completions.create(
        model=QA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"问题: {question}\n\n记忆:\n{context}"}
        ],
        temperature=0.0,
        max_tokens=512,
    )

    return response.choices[0].message.content


@app.route('/batch_process', methods=['POST'])
def batch_process():
    """
    Batch process QA pairs for answering and evaluation.

    Supports TWO formats:

    Format 1 (hmems_generation.py compatible):
    {
        "qa_pairs": [
            {
                "question": "string",
                "answer": "string - ground truth answer",
                "source_session": "string - session identifier"
            },
            ...
        ]
    }

    Returns:
    {
        "results": [
            {
                "question": "string",
                "predicted_answer": "string",
                "ground_truth": "string",
                "reward": 0.0-1.0
            },
            ...
        ]
    }

    Format 2 (legacy):
    {
        "memories": [...],
        "questions": [...]
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Check for hmems_generation format first
        if "qa_pairs" in data:
            qa_pairs = data.get("qa_pairs", [])
            # Optional: direct memories passed from training (bypass store retrieval)
            direct_memories = data.get("memories", None)
            results = []

            logger.info(f"Processing {len(qa_pairs)} QA pairs using {QA_MODEL}")

            for qa in qa_pairs:
                question = qa.get("question", "")
                ground_truth = qa.get("answer", "")

                # Use direct memories if provided, otherwise retrieve from store
                if direct_memories is not None:
                    memory_data = {
                        "vec_memories": direct_memories.get("vec_memories", []),
                        "episodic_memories": direct_memories.get("episodic_memories", []),
                        "semantic_memories": []
                    }
                    logger.info(f"Using direct memories: vec={len(memory_data['vec_memories'])}, epi={len(memory_data['episodic_memories'])}")
                else:
                    # Retrieve relevant memories based on question embedding
                    memory_data = retrieve_relevant_memories(
                        question,
                        topk_vec=RETRIEVE_TOPK_VEC,
                        topk_episodic=RETRIEVE_TOPK_EPISODIC
                    )
                    memory_data["semantic_memories"] = []

                # Generate answer using OpenAI
                try:
                    predicted_answer = generate_answer_via_openai(question, memory_data)
                except Exception as e:
                    logger.warning(f"OpenAI API error for question '{question[:50]}...': {e}")
                    predicted_answer = f"[Error: {str(e)[:50]}]"

                # Compute reward using LLM judge
                reward = compute_reward_llm_judge(question, predicted_answer, ground_truth)

                results.append({
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "ground_truth": ground_truth,
                    "reward": reward
                })

                logger.info(f"QA: {question[:50]}... | GT: {ground_truth[:30]}... | Pred: {predicted_answer[:30]}... | Reward: {reward}")

            return jsonify({"results": results})

        # Legacy format
        memories = data.get('memories', [])
        questions = data.get('questions', [])

        if not memories or not questions:
            return jsonify({"error": "Both 'memories' and 'questions' are required"}), 400

        if len(memories) != len(questions):
            return jsonify({"error": "Number of memory sets must match number of question sets"}), 400

        logger.info(f"Processing batch QA for {len(memories)} memory sets using {QA_MODEL}")

        # Process each memory/question pair
        results = []
        for memory_data, question_list in zip(memories, questions):
            memory_answers = []
            for question in question_list:
                try:
                    answer = generate_answer_via_openai(question, memory_data)
                except Exception as e:
                    logger.warning(f"OpenAI API error for question '{question[:50]}...': {e}")
                    answer = f"[Error generating answer: {str(e)[:100]}]"
                memory_answers.append(answer)
            results.append(memory_answers)

        return jsonify({
            "result": results,
            "status": "success",
            "processed_count": len(memories)
        })

    except Exception as e:
        logger.error(f"Error in /batch_process: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


def compute_reward_llm_judge(question: str, predicted_answer: str, ground_truth: str) -> float:
    """
    Compute reward using LLM judge to evaluate semantic equivalence.

    Uses qwen-plus to evaluate whether the predicted answer semantically matches
    the ground truth answer according to the question's rubric.
    """
    if not ground_truth:
        return 0.0

    template = (
        "I will give you a question, a rubric for desired personalized response, and a response from a model. "
        "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
        "The model does not need to reflect all the points in the rubric. "
        "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
        "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    )
    prompt = template.format(question, ground_truth, predicted_answer)

    try:
        response = openai_client.chat.completions.create(
            model='qwen-plus',
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )

        content = response.choices[0].message.content.strip().lower()
        if "yes" in content and "no" not in content:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"LLM judge error: {e}")
        return 0.0


@app.route('/get_consolidation_tools', methods=['GET'])
def get_consolidation_tools():
    """
    Return consolidation tool schemas for RL agent.

    Returns:
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "consolidate_merge",
                    "description": "...",
                    "parameters": {...}
                }
            },
            ...
        ]
    }
    """
    return jsonify({
        "tools": get_consolidation_tool_schemas()
    })


@app.route('/execute_tool', methods=['POST'])
def execute_tool():
    """
    Execute a consolidation tool call from the RL agent.

    Expected payload:
    {
        "tool_name": "consolidate_merge|consolidate_augment|consolidate_none",
        "arguments": {
            "target_episodic_id": int,       # for merge
            "merged_content": str,          # for merge
            "relevant_vec_ids": [int],       # for augment
            "augmented_content": str,       # for augment
            "new_memory": str               # for none
        }
    }

    Returns:
    {
        "status": "ok|error",
        "action": "merge|augment|none",
        "memory_id": int,
        "content": str,
        "memory_length": int
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        tool_name = data.get('tool_name')
        arguments = data.get('arguments', {})

        if tool_name not in TOOL_IMPLS:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 400

        result = TOOL_IMPLS[tool_name](consolidation_store, arguments)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /execute_tool: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the consolidation store."""
    consolidation_store.reset()
    return jsonify({"status": "success", "message": "Store reset"})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "vec_store_size": len(consolidation_store.vec_store),
        "episodic_store_size": len(consolidation_store.episodic_store)
    })


@app.route('/stats', methods=['GET'])
def stats():
    """Get current store statistics."""
    return jsonify({
        "vec_memories": len(consolidation_store.vec_store),
        "episodic_memories": len(consolidation_store.episodic_store),
        "total_vec_id": consolidation_store.vec_id_counter,
        "total_episodic_id": consolidation_store.episodic_id_counter
    })


def main():
    parser = argparse.ArgumentParser(description="HMEMS Consolidation Agent Server")
    parser.add_argument('--port', type=int, default=5005, help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run server on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    logger.info(f"Starting HMEMS Consolidation Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
