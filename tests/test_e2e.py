"""
HMEMS Consolidation Agent End-to-End Tests

This module contains tests for:
1. Data preprocessing
2. Consolidation agent logic
3. Reward computation
4. Server endpoints
5. Integration
"""

import os
import sys
import json
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Load sample data
        with open("dataset/locomo.json", "r") as f:
            cls.dataset = json.load(f)

    def test_parse_conversation(self):
        """Test conversation parsing."""
        from scripts.preprocess_locomo import parse_conversation
        conv = self.dataset[0]["conversation"]
        turns = parse_conversation(conv)

        self.assertIsInstance(turns, list)
        self.assertGreater(len(turns), 0)

        # Check turn structure
        first_turn = turns[0]
        self.assertIn("speaker", first_turn)
        self.assertIn("text", first_turn)
        self.assertIn("dia_id", first_turn)

    def test_format_turn(self):
        """Test turn formatting."""
        from scripts.preprocess_locomo import format_turn
        turn = {"speaker": "Alice", "text": "Hello world", "dia_id": "D1:1"}
        formatted = format_turn(turn)

        self.assertEqual(formatted, "[Alice]: Hello world")

    def test_conversation_to_text(self):
        """Test conversation to text conversion."""
        from scripts.preprocess_locomo import conversation_to_text
        conv = self.dataset[0]["conversation"]
        text = conversation_to_text(conv)

        self.assertIsInstance(text, str)
        self.assertIn("[", text)  # Should contain formatted turns
        self.assertIn("]:", text)

    def test_extract_qa_pairs(self):
        """Test QA pair extraction."""
        from scripts.preprocess_locomo import extract_qa_pairs
        qa_list = self.dataset[0]["qa"]
        extracted = extract_qa_pairs(qa_list)

        self.assertIsInstance(extracted, list)

        # Check structure
        for qa in extracted:
            self.assertIn("question", qa)
            self.assertIn("answer", qa)

        # Adversarial (category 5) should be filtered
        for qa in extracted:
            self.assertNotEqual(qa.get("category"), 5)

    def test_qa_with_adversarial_answer(self):
        """Test QA with adversarial_answer field."""
        from scripts.preprocess_locomo import extract_qa_pairs
        qa = {
            "question": "Test question?",
            "adversarial_answer": "Test answer",
            "category": 5,
        }
        extracted = extract_qa_pairs([qa])

        # Should be filtered since category=5
        self.assertEqual(len(extracted), 0)


class TestConsolidationAgent(unittest.TestCase):
    """Test consolidation agent logic."""

    def setUp(self):
        """Set up agent."""
        from src.consolidation_agent import (
            ConsolidationAgent,
            ConsolidationAction,
            VecStoreInterface,
            EpisodicStoreInterface,
        )
        self.Agent = ConsolidationAgent
        self.Action = ConsolidationAction
        self.VecStore = VecStoreInterface
        self.EpisodicStore = EpisodicStoreInterface

    def test_vec_store_add_search(self):
        """Test vector store add and search."""
        store = self.VecStore()

        # Add memories
        embedding1 = np.random.randn(1536)
        embedding2 = np.random.randn(1536)

        store.add(0, "Memory 1", embedding1)
        store.add(1, "Memory 2", embedding2)

        self.assertEqual(len(store.memories), 2)

        # Search
        scores, ids, contents = store.search(embedding1, top_k=2)
        self.assertEqual(len(scores), 2)

    def test_episodic_store_add_update(self):
        """Test episodic store operations."""
        store = self.EpisodicStore()

        # Add episodic memory
        store.add(0, "Episodic 1")
        self.assertEqual(len(store.episodic_memories), 1)

        # Update
        store.update(0, "Updated episodic 1")
        self.assertEqual(store.episodic_memories[0], "Updated episodic 1")

        # Remove
        store.remove(0)
        self.assertEqual(len(store.episodic_memories), 0)

    def test_agent_retrieve_context(self):
        """Test agent retrieval."""
        agent = self.Agent()

        # Add some memories
        for i in range(5):
            agent.vec_store.add(i, f"Memory {i}", np.random.randn(1536))

        episodic, vec = agent.retrieve_context("Test query", np.random.randn(1536))

        self.assertIsInstance(episodic, list)
        self.assertIsInstance(vec, list)

    def test_agent_decide_action_merge(self):
        """Test agent decision - merge."""
        from src.consolidation_agent import ConsolidationAction
        agent = self.Agent()

        # Add relevant episodic memory - episodic store stores content directly
        agent.episodic_store.add(0, "Existing episodic memory")

        episodic_results = [{"content": "Existing episodic memory", "score": 0.9}]
        vec_results = []

        action = agent.decide_action(
            "New memory",
            np.random.randn(1536),
            episodic_results,
            vec_results,
        )

        self.assertEqual(action.action, "merge")

    def test_agent_decide_action_augment(self):
        """Test agent decision - augment."""
        agent = self.Agent()
        agent.merge_threshold = 1.0  # Disable merge

        # Add relevant vec memories
        for i in range(3):
            agent.vec_store.add(i, f"Vec memory {i}", np.random.randn(1536))

        episodic_results = []
        vec_results = [
            {"id": 0, "content": "Vec memory 0", "score": 0.8},
            {"id": 1, "content": "Vec memory 1", "score": 0.8},
            {"id": 2, "content": "Vec memory 2", "score": 0.8},
        ]

        action = agent.decide_action(
            "New memory",
            np.random.randn(1536),
            episodic_results,
            vec_results,
        )

        self.assertEqual(action.action, "augment")

    def test_agent_decide_action_none(self):
        """Test agent decision - none."""
        agent = self.Agent()
        agent.merge_threshold = 1.0  # Disable merge
        agent.min_relevant_score = 0.9  # High threshold

        episodic_results = []
        vec_results = [{"id": 0, "content": "Vec memory", "score": 0.5}]

        action = agent.decide_action(
            "New memory",
            np.random.randn(1536),
            episodic_results,
            vec_results,
        )

        self.assertEqual(action.action, "none")

    def test_execute_merge(self):
        """Test execute merge action."""
        from src.consolidation_agent import ConsolidationAction
        agent = self.Agent()
        agent.episodic_store.add(0, "Past episodic")

        action = ConsolidationAction(
            action="merge",
            target_episodic_id=0,
            merged_content="Merged content",
        )

        result = agent.execute_action(action, "New memory")

        self.assertEqual(result["status"], "merged")
        # Merged content includes new memory appended
        self.assertEqual(result["content"], "Merged content\nNew memory")

    def test_execute_augment(self):
        """Test execute augment action."""
        from src.consolidation_agent import ConsolidationAction
        agent = self.Agent()

        action = ConsolidationAction(
            action="augment",
            relevant_vec_ids=[0, 1],
            augmented_content="Augmented content",
        )

        result = agent.execute_action(action, "New memory")

        self.assertEqual(result["status"], "augmented")

    def test_execute_none(self):
        """Test execute none action."""
        from src.consolidation_agent import ConsolidationAction
        agent = self.Agent()

        action = ConsolidationAction(action="none")

        result = agent.execute_action(action, "New memory")

        self.assertEqual(result["status"], "stored")


class TestRewardFunction(unittest.TestCase):
    """Test reward computation."""

    def setUp(self):
        """Set up reward computer."""
        from src.reward_function import RewardComputer
        self.RewardComputer = RewardComputer

    def test_compression_reward_full(self):
        """Test compression reward with no compression."""
        rc = self.RewardComputer()
        reward = rc.compute_compression_reward(100, 100)
        self.assertAlmostEqual(reward, 0.0, places=5)

    def test_compression_reward_half(self):
        """Test compression reward with 50% compression."""
        rc = self.RewardComputer()
        reward = rc.compute_compression_reward(50, 100)
        self.assertAlmostEqual(reward, 0.5, places=5)

    def test_compression_reward_zero(self):
        """Test compression reward with expansion."""
        rc = self.RewardComputer()
        reward = rc.compute_compression_reward(200, 100)
        self.assertAlmostEqual(reward, -1.0, places=5)

    def test_compression_reward_zero_original(self):
        """Test compression reward with zero original length."""
        rc = self.RewardComputer()
        reward = rc.compute_compression_reward(100, 0)
        self.assertEqual(reward, 0.0)

    def test_answer_match_exact(self):
        """Test exact answer match."""
        rc = self.RewardComputer()
        score = rc.check_answer_match("Hello world", "Hello world")
        self.assertEqual(score, 1.0)

    def test_answer_match_case_insensitive(self):
        """Test case insensitive match."""
        rc = self.RewardComputer()
        score = rc.check_answer_match("HELLO WORLD", "hello world")
        self.assertEqual(score, 1.0)

    def test_answer_match_substring(self):
        """Test substring match."""
        rc = self.RewardComputer()
        score = rc.check_answer_match("The answer is hello", "hello")
        self.assertEqual(score, 1.0)

    def test_answer_match_partial(self):
        """Test partial match."""
        rc = self.RewardComputer()
        score = rc.check_answer_match("The quick brown fox", "quick brown")
        self.assertGreater(score, 0.5)

    def test_answer_match_numeric(self):
        """Test numeric answer match."""
        rc = self.RewardComputer()
        score = rc.check_answer_match("The number is 42", 42)
        self.assertEqual(score, 1.0)

    def test_answer_match_multi_keyword(self):
        """Test multi-keyword match."""
        rc = self.RewardComputer()
        score = rc.check_answer_match("apple; banana; cherry", "apple; banana; cherry")
        self.assertEqual(score, 1.0)

    def test_qa_reward_perfect(self):
        """Test QA reward with perfect answers."""
        rc = self.RewardComputer()
        preds = ["Answer1", "Answer2"]
        golds = ["Answer1", "Answer2"]
        questions = ["Q1", "Q2"]

        reward = rc.compute_qa_reward(preds, golds, questions)
        self.assertAlmostEqual(reward, 1.0, places=5)

    def test_qa_reward_partial(self):
        """Test QA reward with partial correct."""
        rc = self.RewardComputer()
        preds = ["Answer1", "Wrong"]
        golds = ["Answer1", "Answer2"]
        questions = ["Q1", "Q2"]

        reward = rc.compute_qa_reward(preds, golds, questions)
        self.assertAlmostEqual(reward, 0.5, places=5)

    def test_total_reward(self):
        """Test total reward computation."""
        rc = self.RewardComputer(compression_ratio_weight=0.05)

        total = rc.compute_total_reward(qa_scores=1.0, memory_length=50, original_length=100)

        # qa_reward=1.0, compression=0.5, total=1.0 + 0.05*0.5 = 1.025
        self.assertAlmostEqual(total, 1.025, places=5)


class TestConsolidationServer(unittest.TestCase):
    """Test consolidation server endpoints."""

    @classmethod
    def setUpClass(cls):
        """Start server in a separate process."""
        import subprocess
        import time

        # Start server
        cls.server_process = subprocess.Popen(
            [sys.executable, "consolidation_server.py", "--port", "5005"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)  # Wait for server to start

        import requests
        cls.requests = requests

        # Check if server is running
        try:
            cls.requests.get("http://localhost:5005/health", timeout=5)
        except:
            cls.server_process.kill()
            raise RuntimeError("Server failed to start")

    @classmethod
    def tearDownClass(cls):
        """Stop server."""
        cls.server_process.terminate()
        cls.server_process.wait()

    def test_health(self):
        """Test health endpoint."""
        response = self.requests.get("http://localhost:5005/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")

    def test_stats(self):
        """Test stats endpoint."""
        response = self.requests.get("http://localhost:5005/stats")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("vec_memories", data)
        self.assertIn("episodic_memories", data)

    def test_reset(self):
        """Test reset endpoint."""
        response = self.requests.post("http://localhost:5005/reset")
        self.assertEqual(response.status_code, 200)

    def test_consolidate_none(self):
        """Test consolidate with action=none."""
        response = self.requests.post(
            "http://localhost:5005/consolidate",
            json={
                "new_memory": "[Alice]: Hello",
                "prev_context": "",
                "action": "none",
            },
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["action"], "none")
        self.assertIn("memory_id", data)

    def test_consolidate_merge(self):
        """Test consolidate with action=merge."""
        # First add an episodic memory via none action
        self.requests.post(
            "http://localhost:5005/consolidate",
            json={
                "new_memory": "Existing episodic content",
                "prev_context": "",
                "action": "none",
            },
        )

        # Now merge
        response = self.requests.post(
            "http://localhost:5005/consolidate",
            json={
                "new_memory": "New memory to merge",
                "prev_context": "",
                "action": "merge",
                "target_episodic_id": 0,
                "merged_content": "Merged content",
            },
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["action"], "merge")

    def test_consolidate_augment(self):
        """Test consolidate with action=augment."""
        # First add some vec memories
        for i in range(3):
            self.requests.post(
                "http://localhost:5005/consolidate",
                json={
                    "new_memory": f"Vec memory {i}",
                    "prev_context": "",
                    "action": "none",
                },
            )

        # Now augment
        response = self.requests.post(
            "http://localhost:5005/consolidate",
            json={
                "new_memory": "New memory",
                "prev_context": "",
                "action": "augment",
                "relevant_vec_ids": [0, 1, 2],
                "augmented_content": "Augmented content",
            },
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["action"], "augment")

    def test_batch_process(self):
        """Test batch_process endpoint."""
        response = self.requests.post(
            "http://localhost:5005/batch_process",
            json={
                "memories": [{"vec_memories": [], "episodic_memories": []}],
                "questions": [["Test question?"]],
            },
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("result", data)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def test_full_consolidation_pipeline(self):
        """Test complete consolidation pipeline."""
        from src.consolidation_agent import ConsolidationAgent, ConsolidationAction
        from src.reward_function import RewardComputer
        import json

        # Load QA lookup
        with open("data/hmems_consolidation/qa_lookup.json", "r") as f:
            qa_lookup = json.load(f)

        # Initialize components
        agent = ConsolidationAgent()
        rc = RewardComputer(compression_ratio_weight=0.05)

        # Simulate conversation turns
        turns = [
            "[Alice]: Hello Bob!",
            "[Bob]: Hi Alice, how are you?",
            "[Alice]: I'm doing great, thanks for asking!",
        ]

        # Process each turn
        results = []
        for turn in turns:
            # Retrieve context (simulated)
            episodic_results = []
            vec_results = [{"id": i, "content": t, "score": 0.5} for i, t in enumerate(turns[:turns.index(turn)])]

            # Decide action
            action = agent.decide_action(
                turn,
                np.random.randn(1536),
                episodic_results,
                vec_results,
            )

            # Execute
            result = agent.execute_action(action, turn)

            results.append({
                "turn": turn,
                "action": action.action,
                "result": result,
            })

        # Check results
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("action", r)
            self.assertIn(r["action"], ["merge", "augment", "none"])

    def test_reward_with_sample_data(self):
        """Test reward computation with sample data."""
        from src.reward_function import RewardComputer

        rc = RewardComputer(compression_ratio_weight=0.05)

        # Simulated data
        predicted_answers = ["Paris", "2024"]
        gold_answers = ["Paris", "2023"]
        questions = ["What is the capital of France?", "What year is it?"]

        reward = rc.compute_total_reward(
            qa_scores=rc.compute_qa_reward(predicted_answers, gold_answers, questions),
            memory_length=50,
            original_length=100,
        )

        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 0)  # Should have some reward


def run_tests():
    """Run all tests."""
    unittest.main(module=__name__, verbosity=2, exit=False)


if __name__ == "__main__":
    # Change to project root
    os.chdir(Path(__file__).parent)

    # Run tests
    run_tests()
