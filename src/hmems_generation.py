"""
HMEMS Consolidation Agent Generation Manager

This module implements the generation manager for HMEMS consolidation agent,
which processes memory consolidation decisions (merge/augment/none).

Supports two modes:
1. run_consolidation_loop: Original session-level processing (for validation/test)
2. run_per_turn_loop: Per-dialogue-turn processing (for training)
"""

import os
import re
import json
import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from verl import DataProto
import requests

from src.consolidation_agent import ConsolidationAgent, ConsolidationAction
from src.reward_function import RewardComputer
from src.conversation_memory_state import ConversationMemoryState


@dataclass
class ConsolidationGenerationConfig:
    """Configuration for consolidation generation."""
    max_turns: int = 1
    max_prompt_length: int = 2048
    max_response_length: int = 1024
    num_gpus: int = 1
    num_rollouts: int = 1
    consolidate_url: str = "http://127.0.0.1:5005/consolidate"
    respond_url: str = "http://127.0.0.1:5005/batch_process"
    enable_thinking: bool = False
    temperature: float = 1.0


class HMEMSGenerationManager:
    """
    Generation manager for HMEMS consolidation agent.

    Unlike Mem-alpha's multi-turn memory loop, HMEMS consolidation is single-step:
    1. For each new memory turn, retrieve relevant episodic/vec memories
    2. Model outputs consolidation decision (merge/augment/none)
    3. Framework executes the action
    4. Compute reward based on QA performance and compression
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: ConsolidationGenerationConfig,
        qa_lookup: Dict[str, List[Dict]] = None,
        reward_computer: RewardComputer = None,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.qa_lookup = qa_lookup
        self.reward_computer = reward_computer

        # Memory server URLs
        self.consolidate_url = config.consolidate_url
        self.respond_url = config.respond_url

        # Initialize consolidation agent
        self.consolidation_agent = ConsolidationAgent()

        # Prompt template for consolidation decision
        self.prompt_template = self._build_prompt_template()

    def _build_prompt_template(self) -> str:
        """Build the prompt template for consolidation agent."""
        return """You are a memory consolidation agent. Your task is to decide how to handle new conversation memories.

## Memory System
The system has two types of memories:
1. Episodic Memory: High-level summaries of related conversation events
2. Vector Memory: Raw conversation turns stored for retrieval

## Input Format
You will receive:
- New conversation turn to store
- Relevant episodic memories (if any)
- Relevant vector memories (if any)

## Your Task
Decide which action to take:

1. **merge**: If the new memory is related to an existing episodic memory, merge them
2. **augment**: If multiple vector memories are relevant, create a new episodic memory
3. **none**: If no relevant memories found, store as raw vector memory

## Output Format
Respond with JSON:
{{
    "action": "merge" | "augment" | "none",
    "reasoning": "why you chose this action",
    "target_episodic_id": <for merge: id of target episodic memory>,
    "merged_content": "<for merge: combined content>",
    "relevant_vec_ids": <for augment: list of relevant vector memory ids>,
    "augmented_content": "<for augment: new episodic content>"
}}

Now process the following memory:

New Conversation:
{new_memory}

Previous Context:
{prev_context}

Relevant Episodic Memories:
{episodic_memories}

Relevant Vector Memories:
{vec_memories}

Your decision:"""

    def _batch_tokenize(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length
        )['input_ids']

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """Generate with padding for multi-GPU."""
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        padded_active_batch.meta_info = active_batch.meta_info.copy() if hasattr(active_batch, 'meta_info') else {}
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        padded_result = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from result
        result_dict = {}
        for k, v in padded_result.batch.items():
            result_dict[k] = v[:batch_size]

        result = DataProto.from_dict(result_dict)
        result.meta_info = padded_result.meta_info
        return result

    def _parse_consolidation_action(self, response_str: str) -> Optional[Dict]:
        """Parse consolidation action from model response."""
        try:
            json_match = self._extract_json(response_str)
            if json_match:
                action_dict = json.loads(json_match)
                if 'action' in action_dict and action_dict['action'] in ['merge', 'augment', 'none']:
                    return action_dict
        except json.JSONDecodeError:
            pass

        return {'action': 'none'}

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text, handling nested braces."""
        try:
            start = text.find('{')
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            return None
        except:
            return None

    def _format_retrieval_results(
        self,
        episodic_results: List[Dict],
        vec_results: List[Dict]
    ) -> Tuple[str, str]:
        """Format retrieval results for prompt."""
        if episodic_results:
            episodic_str = "\n".join([
                f"- [ID:{i}] {r['content'][:200]}... (score: {r['score']:.2f})"
                for i, r in enumerate(episodic_results)
            ])
        else:
            episodic_str = "None found"

        if vec_results:
            vec_str = "\n".join([
                f"- [ID:{r['id']}] {r['content'][:100]}... (score: {r['score']:.2f})"
                for r in vec_results
            ])
        else:
            vec_str = "None found"

        return episodic_str, vec_str

    def _format_memories_for_prompt(
        self,
        new_memory: str,
        episodic_results: List[Dict],
        vec_results: List[Dict]
    ) -> str:
        """Format retrieval results into a prompt string for per-turn processing."""
        episodic_str = "None found"
        if episodic_results:
            episodic_str = "\n".join([
                f"- [{r['id']}] {r['content'][:200]}..." if len(r['content']) > 200 else f"- [{r['id']}] {r['content']}"
                for r in episodic_results
            ])

        vec_str = "None found"
        if vec_results:
            vec_str = "\n".join([
                f"- [{r['id']}] {r['content'][:100]}..." if len(r['content']) > 100 else f"- [{r['id']}] {r['content']}"
                for r in vec_results
            ])

        return f"""Current Conversation Turn:
{new_memory}

Relevant Episodic Memories:
{episodic_str}

Relevant Vector Memories:
{vec_str}

Your decision (JSON format: {{"action": "none"|"merge"|"augment"}}):"""

    def _process_consolidation_decision(
        self,
        action_dict: Dict,
        new_memory: str,
        episodic_results: List[Dict],
        vec_results: List[Dict],
    ) -> str:
        """Process consolidation decision and return action type."""
        action = action_dict.get('action', 'none')

        if action == 'merge':
            return 'merge'
        elif action == 'augment':
            return 'augment'
        else:
            return 'none'

    def run_consolidation_loop(
        self,
        gen_batch: DataProto,
        new_memories: List[str],
        prev_contexts: List[str],
        sample_ids: List[str],
        qa_pairs_list: List[List[Dict]] = None,
        num_gpus: int = 1,
    ) -> DataProto:
        """
        Run consolidation loop for a batch (session-level processing).

        This is the original implementation for validation/test mode.
        """
        batch_size = len(new_memories)

        # Step 1: Retrieve context for each memory
        episodic_results_list = []
        vec_results_list = []

        for new_memory in new_memories:
            dummy_embedding = torch.randn(1024)  # text-embedding-v4 uses 1024 dim
            episodic_results, vec_results = self.consolidation_agent.retrieve_context(
                new_memory, dummy_embedding
            )
            episodic_results_list.append(episodic_results)
            vec_results_list.append(vec_results)

        # Step 2: Format prompts and generate consolidation decisions
        prompts = []
        for i in range(batch_size):
            episodic_str, vec_str = self._format_retrieval_results(
                episodic_results_list[i], vec_results_list[i]
            )
            prompt = self.prompt_template.format(
                new_memory=new_memories[i],
                prev_context=prev_contexts[i] if prev_contexts[i] else "(no previous context)",
                episodic_memories=episodic_str,
                vec_memories=vec_str,
            )
            prompts.append(prompt)

        # Tokenize prompts
        prompt_ids = self._batch_tokenize(prompts)
        device = gen_batch.batch['input_ids'].device
        prompt_ids = prompt_ids.to(device)

        # Create position_ids for vLLM rollout
        batch_size, seq_len = prompt_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)

        # Create generation batch
        active_batch = DataProto.from_dict({
            'input_ids': prompt_ids,
            'attention_mask': (prompt_ids != self.tokenizer.pad_token_id).long(),
            'position_ids': position_ids,
        })
        active_batch.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': True if self.config.temperature > 0 else False,
            'temperature': self.config.temperature,
        }

        # Generate
        gen_output = self._generate_with_gpu_padding(active_batch)

        # Decode responses
        responses = self.tokenizer.batch_decode(
            gen_output.batch['responses'],
            skip_special_tokens=True
        )

        # Step 3: Parse consolidation decisions
        decisions = []
        action_types = []
        memory_contents = []

        for i, response in enumerate(responses):
            action_dict = self._parse_consolidation_action(response)
            action_type = self._process_consolidation_decision(
                action_dict,
                new_memories[i],
                episodic_results_list[i],
                vec_results_list[i],
            )
            decisions.append({
                "action": action_type,
                "memory_content": memory_contents[-1] if memory_contents else new_memories[i],
                "source_session": sample_ids[i]
            })
            action_types.append(action_type)

        # Step 4: Send consolidation decisions to memory server
        try:
            consolidate_response = requests.post(
                self.consolidate_url,
                json={"decisions": decisions},
                timeout=30
            )
            consolidate_response.raise_for_status()
            consolidate_result = consolidate_response.json()
            print(f"[DEBUG] Consolidation: stored {consolidate_result.get('stored_count', 0)} memories")
        except Exception as e:
            print(f"[WARNING] Failed to send consolidation to memory server: {e}")

        # Step 5: Get QA pairs for reward computation
        questions_list = []
        ground_truth_answers_list = []
        qa_pairs_for_server = []

        if qa_pairs_list is None:
            raise ValueError("qa_pairs_list must be provided")

        for i, qa_pairs in enumerate(qa_pairs_list):
            questions = [qa['question'] for qa in qa_pairs]
            answers = []
            for qa in qa_pairs:
                if 'answer' in qa:
                    answers.append(qa['answer'])
                elif 'adversarial_answer' in qa:
                    answers.append(qa['adversarial_answer'])
                else:
                    raise ValueError(f"Missing 'answer' or 'adversarial_answer' in qa_pairs[{i}]")
            ground_truth_answers_list.append(answers)
            questions_list.append(questions)

            sample_id = sample_ids[i] if i < len(sample_ids) else f"session_{i}"
            for qa in qa_pairs:
                answer = qa.get('answer') or qa.get('adversarial_answer', '')
                qa_pairs_for_server.append({
                    "question": qa['question'],
                    "answer": answer,
                    "source_session": sample_id
                })

        # Step 6: Send QA pairs to memory server
        predicted_answers_list = []
        rewards_list = []

        print(f"[DEBUG] qa_pairs_for_server has {len(qa_pairs_for_server)} items")

        if qa_pairs_for_server:
            try:
                qa_response = requests.post(
                    self.respond_url,
                    json={"qa_pairs": qa_pairs_for_server},
                    timeout=300
                )
                qa_response.raise_for_status()
                qa_result = qa_response.json()
                results = qa_result.get('results', [])

                print(f"[DEBUG] QA Processing: got {len(results)} results from memory server")

                result_idx = 0
                for i, qa_pairs in enumerate(qa_pairs_list):
                    session_preds = []
                    session_rewards = []
                    for _ in qa_pairs:
                        if result_idx < len(results):
                            session_preds.append(results[result_idx].get('predicted_answer', ''))
                            session_rewards.append(results[result_idx].get('reward', 0.0))
                            result_idx += 1
                        else:
                            session_preds.append('')
                            session_rewards.append(0.0)
                    predicted_answers_list.append(session_preds)
                    rewards_list.append(session_rewards)

            except Exception as e:
                print(f"[WARNING] Failed to get QA answers from memory server: {e}")
                for qa_pairs in qa_pairs_list:
                    predicted_answers_list.append([''] * len(qa_pairs))
                    rewards_list.append([0.0] * len(qa_pairs))
        else:
            for qa_pairs in qa_pairs_list:
                predicted_answers_list.append([''] * len(qa_pairs))
                rewards_list.append([0.0] * len(qa_pairs))

        # Prepare output
        indices_in_batch = list(range(batch_size))

        prompt_ids = gen_output.batch['input_ids']
        responses = gen_output.batch['responses']

        prompt_pad_token = self.tokenizer.pad_token_id
        prompt_mask = (prompt_ids != prompt_pad_token).long()
        prompt_lens = prompt_mask.sum(dim=1)

        max_prompt_len = prompt_lens.max().item()
        total_seq_len = max_prompt_len + responses.size(1)

        device = prompt_ids.device
        full_input_ids_list = []
        for i in range(batch_size):
            actual_prompt_len = prompt_lens[i].item()
            prompt_tokens = prompt_ids[i, :actual_prompt_len]
            response_tokens = responses[i]
            full_input_ids = torch.cat([prompt_tokens, response_tokens])
            full_input_ids_list.append(full_input_ids)

        max_total_len = max(len(x) for x in full_input_ids_list)
        padded_input_ids = torch.zeros(batch_size, max_total_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, max_total_len, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, max_total_len, dtype=torch.long, device=device)

        for i in range(batch_size):
            actual_len = len(full_input_ids_list[i])
            padded_input_ids[i, :actual_len] = full_input_ids_list[i]
            attention_mask[i, :actual_len] = 1
            position_ids[i, :actual_len] = torch.arange(actual_len, device=device)

        final_output = {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'responses': responses,
        }

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update({
            'questions_list': questions_list,
            'predicted_answers_list': predicted_answers_list,
            'ground_truth_answers_list': ground_truth_answers_list,
            'indices_in_batch': indices_in_batch,
            'action_types': action_types,
            'memory_contents': memory_contents,
            'memory_server_rewards': rewards_list,
        })

        print(f"[DEBUG hmems_generation] predicted_answers_list lengths: {[len(x) for x in predicted_answers_list]}")
        print(f"[DEBUG hmems_generation] ground_truth_answers_list lengths: {[len(x) for x in ground_truth_answers_list]}")
        print(f"[DEBUG hmems_generation] memory_server_rewards lengths: {[len(x) for x in rewards_list]}")

        return final_output

    def run_per_turn_loop(
        self,
        gen_batch: DataProto,
        session_dialogue: List[Dict],
        qa_pairs: List[Dict],
        customized_grpo_rollout_n: int = 4,
    ) -> DataProto:
        """
        Run per-turn consolidation loop for training.

        Each session copy has its own independent memory state.
        Processes all dialogue turns sequentially, then computes rewards.

        Args:
            gen_batch: DataProto with prompt token IDs (for structure, not used directly)
            session_dialogue: List of dialogue turns [{"dia_id": "...", "speaker": "...", "text": "..."}]
            qa_pairs: List of QA pairs [{"question": "...", "answer": "...", "evidence": ["D1:2"]}]
            customized_grpo_rollout_n: Number of session copies (default 4)

        Returns:
            DataProto with per-turn responses and rewards
        """
        # Quick test mode: limit turns and QA pairs for fast gradient verification
        max_test_turns = int(os.getenv('HMEMS_MAX_TEST_TURNS', '0'))  # 0 means no limit
        max_test_qas = int(os.getenv('HMEMS_MAX_TEST_QAS', '0'))  # 0 means no limit

        # Debug: log QA pairs info
        print(f"[DEBUG run_per_turn_loop] Received qa_pairs: {len(qa_pairs)} pairs")
        if qa_pairs:
            print(f"  First QA: {qa_pairs[0]}")
        else:
            print(f"  [WARNING] Empty qa_pairs!")

        num_turns = len(session_dialogue)
        device = gen_batch.batch['input_ids'].device if 'input_ids' in gen_batch.batch else 'cuda'

        # Initialize memory states for each copy
        memory_states = [ConversationMemoryState() for _ in range(customized_grpo_rollout_n)]

        # Store responses and log_probs for each turn and each copy
        all_responses = [[] for _ in range(customized_grpo_rollout_n)]
        all_prompts = [[] for _ in range(customized_grpo_rollout_n)]
        all_action_types = [[] for _ in range(customized_grpo_rollout_n)]

        # Get OpenAI embedding function
        try:
            from openai import OpenAI
            openai_client = OpenAI()
            embedder = lambda text: np.array(
                openai_client.embeddings.create(
                    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
                    input=text[:8191]
                ).data[0].embedding
            )
        except Exception as e:
            print(f"[WARNING] Failed to initialize OpenAI embedder: {e}")
            print("[WARNING] Using random embeddings - this will not work for real training!")
            embedder = lambda text: np.random.randn(1024)  # text-embedding-v4 uses 1024 dim

        print(f"[DEBUG run_per_turn_loop] Processing {num_turns} turns, {customized_grpo_rollout_n} copies")
        if max_test_turns > 0:
            print(f"[QUICK TEST MODE] Limiting to {max_test_turns} turns")

        # Process each dialogue turn
        for turn_idx, turn in enumerate(session_dialogue):
            # Quick test mode: stop after max_test_turns
            if max_test_turns > 0 and turn_idx >= max_test_turns:
                print(f"[QUICK TEST] Stopping after {max_test_turns} turns")
                break

            new_memory = turn['text']
            dia_id = turn['dia_id']

            # Build prompts for all copies
            prompts = []
            for copy_idx in range(customized_grpo_rollout_n):
                episodic, vec = memory_states[copy_idx].retrieve(
                    new_memory,
                    embedder(new_memory),
                    vec_top_k=5,
                    episodic_top_k=5
                )
                prompt = self._format_memories_for_prompt(new_memory, episodic, vec)
                prompts.append(prompt)
                all_prompts[copy_idx].append(prompt)

            # Tokenize all prompts at once
            prompt_ids = self._batch_tokenize(prompts)
            prompt_ids = prompt_ids.to(device)

            # Create generation batch
            batch_size = customized_grpo_rollout_n
            seq_len = prompt_ids.shape[1]
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)

            active_batch = DataProto.from_dict({
                'input_ids': prompt_ids,
                'attention_mask': (prompt_ids != self.tokenizer.pad_token_id).long(),
                'position_ids': position_ids,
            })
            active_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': True,
                'do_sample': True if self.config.temperature > 0 else False,
                'temperature': self.config.temperature,
            }

            # Generate decisions for all copies
            gen_output = self._generate_with_gpu_padding(active_batch)

            # Decode responses
            responses_str = self.tokenizer.batch_decode(
                gen_output.batch['responses'],
                skip_special_tokens=True
            )

            # Parse decisions and update memory states
            for copy_idx in range(customized_grpo_rollout_n):
                action_dict = self._parse_consolidation_action(responses_str[copy_idx])
                action_type = action_dict.get('action', 'none')

                # Debug: print model response for first few turns only
                if turn_idx < 2:
                    print(f"[DEBUG] Turn {turn_idx + 1} copy {copy_idx} response: {responses_str[copy_idx][:150]}...")

                # Execute action to update memory state
                embedding = embedder(new_memory)
                memory_states[copy_idx].execute_action(action_dict, new_memory, embedding)

                # Save response and log_prob
                all_responses[copy_idx].append({
                    'dia_id': dia_id,
                    'response': responses_str[copy_idx],
                    'action': action_type,
                    'gen_output': gen_output.batch['responses'][copy_idx],
                    'prompt_ids': prompt_ids[copy_idx],
                })
                all_action_types[copy_idx].append(action_type)

            # Print every turn for clarity - show all copy memory states
            mem_summaries = [ms.get_memory_summary() for ms in memory_states]
            vec_counts = [s['num_vec_memories'] for s in mem_summaries]
            epi_counts = [s['num_episodic_memories'] for s in mem_summaries]
            print(f"[DEBUG] Processed turn {turn_idx + 1}/{num_turns}: {dia_id} -> {[a[turn_idx] for a in all_action_types]} | mem: vec={vec_counts}, epi={epi_counts}", flush=True)

        # After all turns, compute rewards using memory states
        print(f"[DEBUG run_per_turn_loop] All turns processed, computing rewards...")

        # Get all dia_ids from session dialogue
        all_dia_ids = [turn['dia_id'] for turn in session_dialogue]
        if max_test_turns > 0:
            all_dia_ids = all_dia_ids[:max_test_turns]

        # Limit QA pairs for quick test mode
        test_qa_pairs = qa_pairs[:max_test_qas] if max_test_qas > 0 else qa_pairs
        if max_test_qas > 0:
            print(f"[QUICK TEST] Limiting QA pairs to {max_test_qas} (from {len(qa_pairs)})")

        # Compute rewards for each copy
        turn_rewards_list = []
        total_rewards = []

        for copy_idx in range(customized_grpo_rollout_n):
            memory_summary = self._build_memory_summary(memory_states[copy_idx])
            qa_results = self._call_batch_process_for_rewards(test_qa_pairs, memory_summary)
            turn_rewards, total_reward = self._allocate_rewards_to_turns(
                qa_results, all_dia_ids, mode=1
            )
            turn_rewards_list.append(turn_rewards)
            total_rewards.append(total_reward)

        print(f"[DEBUG run_per_turn_loop] Rewards computed: {total_rewards}")

        # Truncate responses/prompts to match max_test_turns if needed
        actual_num_turns = len(all_responses[0]) if all_responses and all_responses[0] else 0
        if max_test_turns > 0 and actual_num_turns > max_test_turns:
            all_responses = [r[:max_test_turns] for r in all_responses]
            all_prompts = [p[:max_test_turns] for p in all_prompts]

        # Prepare output DataProto
        return self._prepare_per_turn_output(
            all_responses=all_responses,
            all_prompts=all_prompts,
            turn_rewards_list=turn_rewards_list,
            all_dia_ids=all_dia_ids,
            memory_states=memory_states,
            qa_pairs=qa_pairs,
            device=device,
        )

    def _build_memory_summary(self, memory_state: ConversationMemoryState) -> Dict[str, Any]:
        """Build memory summary dict for batch_process API call."""
        vec_memories = list(memory_state.payload_mapping.values())
        episodic_memories = [e['content'] for e in memory_state.episodic_store.values()]

        # Debug: log actual memory content
        print(f"[DEBUG _build_memory_summary] vec_count={len(vec_memories)}, epi_count={len(episodic_memories)}")
        if vec_memories:
            print(f"  First vec memory (first 100 chars): {vec_memories[0][:100]}")
        if episodic_memories:
            print(f"  First epi memory (first 100 chars): {episodic_memories[0][:100]}")

        return {
            'vec_memories': vec_memories,
            'episodic_memories': episodic_memories,
            'semantic_memories': []
        }

    def _call_batch_process_for_rewards(
        self,
        qa_pairs: List[Dict],
        memory_summary: Dict[str, Any]
    ) -> List[Dict]:
        """Call batch_process to get predicted answers and rewards."""
        if not qa_pairs:
            return []

        qa_pairs_for_server = []
        for qa in qa_pairs:
            qa_pairs_for_server.append({
                "question": qa['question'],
                "answer": qa.get('answer', ''),
                "source_session": "training"
            })

        try:
            # Debug: log memory summary
            print(f"[DEBUG batch_process] Sending {len(qa_pairs_for_server)} QAs with memories: vec={len(memory_summary.get('vec_memories', []))}, epi={len(memory_summary.get('episodic_memories', []))}")
            response = requests.post(
                self.respond_url,
                json={"qa_pairs": qa_pairs_for_server, "memories": memory_summary},
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            results = result.get('results', [])

            qa_results = []
            for i, qa in enumerate(qa_pairs):
                if i < len(results):
                    qa_results.append({
                        'question': qa['question'],
                        'answer': qa.get('answer', ''),
                        'predicted': results[i].get('predicted_answer', ''),
                        'reward': results[i].get('reward', 0.0),
                        'evidence': qa.get('evidence', [])
                    })
                else:
                    qa_results.append({
                        'question': qa['question'],
                        'answer': qa.get('answer', ''),
                        'predicted': '',
                        'reward': 0.0,
                        'evidence': qa.get('evidence', [])
                    })

            return qa_results

        except Exception as e:
            print(f"[WARNING] Failed to call batch_process: {e}")
            return [
                {'question': qa['question'], 'answer': qa.get('answer', ''),
                 'predicted': '', 'reward': 0.0, 'evidence': qa.get('evidence', [])}
                for qa in qa_pairs
            ]

    def _allocate_rewards_to_turns(
        self,
        qa_results: List[Dict],
        all_dia_ids: List[str],
        mode: int = 1
    ) -> Tuple[Dict[str, float], float]:
        """
        Allocate QA rewards to dialogue turns based on evidence mapping.

        Mode 1: All turns get the same average reward
        Mode 2: All evidence turns get the same average reward, others get 0
        Mode 3: Each turn gets the average reward from QAs that mention it as evidence
        """
        if not qa_results:
            return {did: 0.0 for did in all_dia_ids}, 0.0

        total_reward = np.mean([r['reward'] for r in qa_results])

        if mode == 1:
            turn_rewards = {did: total_reward for did in all_dia_ids}

        elif mode == 2:
            evidence_turns = set()
            for r in qa_results:
                evidence_turns.update(r.get('evidence', []))

            turn_rewards = {
                did: (total_reward if did in evidence_turns else 0.0)
                for did in all_dia_ids
            }

        elif mode == 3:
            did_to_rewards = {did: [] for did in all_dia_ids}

            for r in qa_results:
                for did in r.get('evidence', []):
                    if did in did_to_rewards:
                        did_to_rewards[did].append(r['reward'])

            turn_rewards = {
                did: (np.mean(rewards) if rewards else 0.0)
                for did, rewards in did_to_rewards.items()
            }

        else:
            turn_rewards = {did: total_reward for did in all_dia_ids}

        return turn_rewards, total_reward

    def _prepare_per_turn_output(
        self,
        all_responses: List[List[Dict]],
        all_prompts: List[List[str]],
        turn_rewards_list: List[Dict[str, float]],
        all_dia_ids: List[str],
        memory_states: List[ConversationMemoryState],
        qa_pairs: List[Dict],
        device: str,
    ) -> DataProto:
        """Prepare output DataProto for per-turn training."""
        customized_grpo_rollout_n = len(all_responses)
        num_turns = len(all_dia_ids)

        all_input_ids = []
        all_attention_masks = []
        all_position_ids = []
        all_response_ids = []

        for copy_idx in range(customized_grpo_rollout_n):
            turn_input_ids = []
            turn_response_ids = []

            for turn_idx in range(num_turns):
                turn_data = all_responses[copy_idx][turn_idx]
                prompt_ids = turn_data['prompt_ids']
                response_ids = turn_data['gen_output']

                # Remove padding from prompt_ids
                prompt_mask = (prompt_ids != self.tokenizer.pad_token_id)
                prompt_ids = prompt_ids[prompt_mask]

                turn_input_ids.append(prompt_ids)
                turn_response_ids.append(response_ids)

            # Concatenate all turns
            if turn_input_ids:
                concat_prompt_ids = torch.cat(turn_input_ids)
                concat_response_ids = torch.cat(turn_response_ids)
            else:
                concat_prompt_ids = torch.tensor([], dtype=torch.long)
                concat_response_ids = torch.tensor([], dtype=torch.long)

            full_ids = torch.cat([concat_prompt_ids, concat_response_ids])
            attention_mask = torch.ones_like(full_ids)
            position_ids = torch.arange(len(full_ids))

            all_input_ids.append(full_ids)
            all_attention_masks.append(attention_mask)
            all_position_ids.append(position_ids)
            all_response_ids.append(concat_response_ids)

        # Pad all sequences to same length
        max_len = max(len(x) for x in all_input_ids) if all_input_ids else 1

        padded_input_ids = torch.zeros(customized_grpo_rollout_n, max_len, dtype=torch.long)
        padded_attention_mask = torch.zeros(customized_grpo_rollout_n, max_len, dtype=torch.long)
        padded_position_ids = torch.zeros(customized_grpo_rollout_n, max_len, dtype=torch.long)

        for i in range(customized_grpo_rollout_n):
            actual_len = len(all_input_ids[i])
            padded_input_ids[i, :actual_len] = all_input_ids[i]
            padded_attention_mask[i, :actual_len] = all_attention_masks[i]
            padded_position_ids[i, :actual_len] = all_position_ids[i]

        max_response_len = max(len(x) for x in all_response_ids) if all_response_ids else 1
        padded_response_ids = torch.zeros(customized_grpo_rollout_n, max_response_len, dtype=torch.long)
        for i in range(customized_grpo_rollout_n):
            actual_len = len(all_response_ids[i])
            padded_response_ids[i, :actual_len] = all_response_ids[i]

        final_output = DataProto.from_dict({
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'position_ids': padded_position_ids,
            'responses': padded_response_ids,
        })

        final_output.meta_info.update({
            'all_responses': all_responses,
            'all_prompts': all_prompts,
            'turn_rewards_list': turn_rewards_list,
            'all_dia_ids': all_dia_ids,
            'num_turns': num_turns,
            'customized_grpo_rollout_n': customized_grpo_rollout_n,
            'memory_states_summary': [ms.get_memory_summary() for ms in memory_states],
            'qa_pairs': qa_pairs,
            'questions_list': [[qa['question'] for qa in qa_pairs]],
            'ground_truth_answers_list': [[qa.get('answer', '') for qa in qa_pairs]],
        })

        print(f"[DEBUG _prepare_per_turn_output] Created output with {customized_grpo_rollout_n} copies, {num_turns} turns")

        return final_output


def create_hmems_generation_manager(
    tokenizer,
    actor_rollout_wg,
    config: ConsolidationGenerationConfig,
    compression_ratio_weight: float = 0.05,
    is_validation: bool = False,
) -> HMEMSGenerationManager:
    """Factory function to create HMEMSGenerationManager."""
    reward_computer = RewardComputer(compression_ratio_weight=compression_ratio_weight)

    return HMEMSGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=actor_rollout_wg,
        config=config,
        qa_lookup=None,
        reward_computer=reward_computer,
        is_validation=is_validation,
    )
