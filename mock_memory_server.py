#!/usr/bin/env python3
"""
Mock Memory Server for HMEMS Training

This server provides memory consolidation and QA answering endpoints:
1. /consolidate - Store memories based on consolidation decisions
2. /batch_process - Answer QA pairs using memories and evaluate with LLM judge
"""

import os
import json
import logging
import requests
from flask import Flask, request, jsonify
from typing import List, Dict, Any

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory memory stores
episodic_memory = []  # List of {"id", "content", "embedding"}
vector_memory = []    # List of {"id", "content", "embedding"}

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

def get_openai_response(prompt: str, model: str = None) -> str:
    """Call OpenAI API to get a response."""
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set, returning mock response")
        return "Mock answer"

    if model is None:
        model = os.getenv("QA_MODEL", "qwen-plus")

    # Fix DashScope URL: use /v1/chat/completions directly
    base_url = OPENAI_API_BASE.rstrip('/')
    if 'dashscope' in base_url:
        # DashScope compatible mode
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return ""


def retrieve_relevant_memories(question: str, topk: int = 3) -> List[Dict]:
    """
    Retrieve relevant memories for a question using simple keyword matching.
    In production, this would use vector similarity search.
    """
    # Simple keyword-based retrieval for mock
    question_words = set(question.lower().split())
    scored = []

    # Search both episodic and vector memory
    for mem in episodic_memory + vector_memory:
        content_words = set(mem["content"].lower().split())
        overlap = len(question_words & content_words)
        scored.append((overlap, mem))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [mem for _, mem in scored[:topk]]


@app.route('/consolidate', methods=['POST'])
def consolidate():
    """
    Receive consolidation decisions and store memories.

    Expected JSON format:
    {
        "decisions": [
            {
                "action": "merge" | "augment" | "none",
                "memory_content": "...",  # The consolidated memory content
                "source_session": "conv_id_session_name"
            },
            ...
        ]
    }
    """
    data = request.get_json()
    decisions = data.get("decisions", [])
    logger.info(f"Received {len(decisions)} consolidation decisions")

    stored_ids = []
    for decision in decisions:
        action = decision.get("action", "none")
        content = decision.get("memory_content", "")
        source = decision.get("source_session", "")

        if not content:
            continue

        if action == "merge":
            # Merge with existing episodic memory (simplified: just append)
            episodic_memory.append({
                "id": f"ep_{len(episodic_memory)}",
                "content": content,
                "source": source,
                "type": "merged"
            })
            stored_ids.append(f"merged: {content[:50]}...")

        elif action == "augment":
            # Create new episodic memory
            episodic_memory.append({
                "id": f"ep_{len(episodic_memory)}",
                "content": content,
                "source": source,
                "type": "augmented"
            })
            stored_ids.append(f"augmented: {content[:50]}...")

        else:  # "none"
            # Store as raw vector memory
            vector_memory.append({
                "id": f"vec_{len(vector_memory)}",
                "content": content,
                "source": source
            })
            stored_ids.append(f"vector: {content[:50]}...")

    logger.info(f"Stored {len(stored_ids)} memories. Total: episodic={len(episodic_memory)}, vector={len(vector_memory)}")

    return jsonify({
        'status': 'ok',
        'stored_count': len(stored_ids),
        'stored_ids': stored_ids,
        'total_episodic': len(episodic_memory),
        'total_vector': len(vector_memory)
    })


@app.route('/batch_process', methods=['POST'])
def batch_process():
    """
    Answer QA pairs using memories and evaluate with LLM judge.

    Expected JSON format:
    {
        "qa_pairs": [
            {
                "question": "...",
                "answer": "...",  # Ground truth
                "source_session": "conv_id_session_name"
            },
            ...
        ]
    }

    Returns:
    {
        "results": [
            {
                "question": "...",
                "predicted_answer": "...",
                "ground_truth": "...",
                "reward": 0.0-1.0
            },
            ...
        ]
    }
    """
    data = request.get_json()
    qa_pairs = data.get("qa_pairs", [])
    logger.info(f"Received {len(qa_pairs)} QA pairs")

    results = []
    for qa in qa_pairs:
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")
        source_session = qa.get("source_session", "")

        # Step 1: Retrieve relevant memories
        relevant_memories = retrieve_relevant_memories(question)

        # Step 2: Build context with memories
        if relevant_memories:
            context = "Relevant memories:\n"
            for mem in relevant_memories:
                context += f"- {mem['content']}\n"
        else:
            context = "No relevant memories found.\n"

        # Step 3: Generate answer using OpenAI API
        answer_prompt = f"""{context}

Based on the above memories, answer the following question:

Question: {question}

Your answer (be concise):"""

        predicted_answer = get_openai_response(answer_prompt)

        # Step 4: Evaluate answer using LLM-as-judge
        judge_prompt = f"""You are evaluating whether a predicted answer matches the ground truth answer.

Ground Truth: {ground_truth}
Predicted: {predicted_answer}

Task: Determine if the predicted answer is semantically correct (matches the meaning, not exact words).
Consider the predicted answer correct if it conveys the same information as the ground truth, even if worded differently.

Respond with ONLY a number between 0.0 and 1.0:
- 1.0 = completely correct
- 0.5 = partially correct
- 0.0 = completely wrong

Your response:"""

        reward_str = get_openai_response(judge_prompt)
        try:
            reward = float(reward_str.strip())
        except:
            reward = 0.0

        results.append({
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "reward": reward,
            "relevant_memories_count": len(relevant_memories)
        })

        logger.info(f"QA: {question[:50]}... | GT: {ground_truth[:30]}... | Pred: {predicted_answer[:30]}... | Reward: {reward}")

    return jsonify({'results': results})


@app.route('/clear', methods=['POST'])
def clear_memories():
    """Clear all in-memory stores."""
    global episodic_memory, vector_memory
    episodic_memory = []
    vector_memory = []
    logger.info("Cleared all memories")
    return jsonify({'status': 'ok'})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'episodic_memory_count': len(episodic_memory),
        'vector_memory_count': len(vector_memory)
    })


@app.route('/stats', methods=['GET'])
def stats():
    """Get memory statistics."""
    return jsonify({
        'episodic_memory': len(episodic_memory),
        'vector_memory': len(vector_memory)
    })


if __name__ == '__main__':
    print("=" * 60)
    print("HMEMS Mock Memory Server")
    print("=" * 60)
    print("Endpoints:")
    print("  POST /consolidate - Store memories from consolidation decisions")
    print("  POST /batch_process - Answer QA pairs and evaluate")
    print("  POST /clear - Clear all memories")
    print("  GET /health - Health check")
    print("  GET /stats - Memory statistics")
    print("=" * 60)
    print("Starting mock memory server on port 5005...")
    app.run(host='0.0.0.0', port=5005, debug=False)
