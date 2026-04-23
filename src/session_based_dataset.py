"""
HMEMS Session-Based Dataset

Dataset for HMEMS session-level RL training.
Each session step contains:
- conv_id: conversation identifier
- session_name: session name (e.g., session_1)
- session_idx: index of the session within conversation
- session_dialogue: list of dialogue turns with speaker, dia_id, text
- qa_pairs: list of QA pairs for this session
- is_evidence_session: whether this session has evidence QA
- n_turns: number of turns in this session

In RL training:
- For evidence sessions: compute reward based on QA accuracy
- For non-evidence sessions: advantage = 0 (no policy gradient)
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
from torch.utils.data import Dataset
from omegaconf import ListConfig
import torch


class SessionBasedHMEMSDataset(Dataset):
    """
    Dataset for HMEMS session-based training.

    This dataset reads JSONL files containing session-level data:
    - conv_id: conversation identifier
    - session_name: session name (e.g., session_1)
    - session_idx: index of the session within conversation
    - session_dialogue: list of dialogue turns [{speaker, dia_id, text}, ...]
    - qa_pairs: list of QA pairs [{question, answer, evidence, category}, ...]
    - is_evidence_session: whether this session has evidence QA
    - n_turns: number of turns in this session
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer,
        processor=None,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize session-based HMEMS dataset.

        Args:
            data_files: List of JSONL data file paths
            tokenizer: Tokenizer for prompt encoding
            processor: Not used (for verl interface compatibility)
            config: Dataset configuration
        """
        # Handle single file path or list
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.tokenizer = tokenizer
        self.config = config or {}
        self.data = []

        # Load data from all files
        for data_file in data_files:
            with open(data_file, "r") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        self.data.append(record)

        print(f"Loaded {len(self.data)} session steps from {len(data_files)} files")

        # Statistics
        evidence_count = sum(1 for d in self.data if d.get('is_evidence_session', False))
        non_evidence_count = len(self.data) - evidence_count
        total_qas = sum(len(d.get('qa_pairs', [])) for d in self.data)
        print(f"  - Evidence sessions: {evidence_count}")
        print(f"  - Non-evidence sessions: {non_evidence_count}")
        print(f"  - Total QA pairs: {total_qas}")

    def __len__(self) -> int:
        # Limit to 1 session per epoch for quick testing
        # Set HMEMS_MAX_SESSIONS env var
        max_sessions = int(os.getenv('HMEMS_MAX_SESSIONS', '0'))
        if max_sessions > 0:
            return min(max_sessions, len(self.data))
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single session step."""
        record = self.data[idx]

        # Build prompt from session dialogue
        prompt = self._build_prompt(record)
        session_dialogue = record.get('session_dialogue', [])
        qa_pairs = record.get('qa_pairs', [])
        is_evidence_session = record.get('is_evidence_session', len(qa_pairs) > 0)

        # Get max_length from config, default to 4096
        max_length = 4096
        if self.config:
            max_length = int(self.config.get("max_prompt_length", max_length))

        # Tokenize prompt without padding to get raw ids
        raw_encoding = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        )

        # Tokenize prompt with explicit padding to max_length
        prompt_encoding = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

        input_ids = prompt_encoding["input_ids"].squeeze(0)
        attention_mask = prompt_encoding["attention_mask"].squeeze(0)

        # Compute position_ids (standard for causal LM)
        position_ids = torch.arange(max_length)

        # Raw prompt ids (before padding)
        raw_prompt_ids = raw_encoding["input_ids"].squeeze(0).tolist()
        if isinstance(raw_prompt_ids, int):
            raw_prompt_ids = [raw_prompt_ids]

        # Extract ground truth answers from QA pairs for reward computation
        ground_truth_answers = []
        for qa in qa_pairs:
            if isinstance(qa, dict):
                ground_truth_answers.append(qa.get('answer', ''))
            else:
                ground_truth_answers.append('')

        return {
            "conv_id": record.get('conv_id', ''),
            "session_name": record.get('session_name', ''),
            "session_idx": record.get('session_idx', 0),
            "session_dialogue": session_dialogue,
            "qa_pairs": qa_pairs,
            "is_evidence_session": is_evidence_session,
            "n_turns": record.get('n_turns', len(session_dialogue)),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt": prompt,
            "raw_prompt_ids": raw_prompt_ids,
            "chunks": [],  # Empty chunks for single-turn scenario
            "data_source": "hmems_session_based",
            "ground_truth_answers_list": ground_truth_answers,  # For reward manager
        }

    def _build_prompt(self, record: Dict) -> str:
        """
        Build the prompt for session consolidation.

        The consolidation agent's task is ONLY to decide how to consolidate memories.
        QA answering is done separately by the memory server (via external API).

        The prompt instructs the model to:
        1. Read and understand the session dialogue
        2. Decide how to consolidate memories based on the dialogue
        """
        session_dialogue = record.get('session_dialogue', [])

        # Format dialogue as text
        if isinstance(session_dialogue, list):
            dialogue_text = self._format_dialogue(session_dialogue)
        else:
            dialogue_text = str(session_dialogue)

        # Consolidation agent ONLY outputs memory consolidation decisions
        # No QA answering - that's done separately by memory server
        prompt = f"""You are a memory consolidation agent. Your task is to decide how to handle conversation memories.

## Session Dialogue
{dialogue_text}

## Your Task
Decide how to consolidate memories from the dialogue:
- **merge**: If related to existing episodic memory, merge them
- **augment**: If multiple relevant memories, create new episodic memory
- **none**: If no relevant memories, store as raw vector memory

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

Your decision:"""

        return prompt

    def _format_dialogue(self, dialogue: List[Dict]) -> str:
        """Format dialogue turns as text."""
        lines = []
        for turn in dialogue:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            dia_id = turn.get('dia_id', '')
            lines.append(f"[{dia_id}] {speaker}: {text}")
        return '\n'.join(lines)


def session_based_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for session-based HMEMS dataset.

    Pads sequences to the same length within the batch.
    """
    # Find max length
    max_len = max(item["input_ids"].shape[0] for item in batch)

    # Pad sequences
    input_ids = []
    attention_masks = []
    position_ids = []

    for item in batch:
        input_ids_tensor = item["input_ids"]
        if input_ids_tensor.shape[0] < max_len:
            padding = torch.full(
                (max_len - input_ids_tensor.shape[0],),
                0
            )
            input_ids_tensor = torch.cat([input_ids_tensor, padding])
        elif input_ids_tensor.shape[0] > max_len:
            input_ids_tensor = input_ids_tensor[:max_len]

        input_ids.append(input_ids_tensor)
        attention_masks.append((input_ids_tensor != 0).long())

        # Pad position_ids similarly
        pos_ids = item["position_ids"]
        if pos_ids.shape[0] < max_len:
            pos_padding = torch.full(
                (max_len - pos_ids.shape[0],),
                0
            )
            pos_ids = torch.cat([pos_ids, pos_padding])
        elif pos_ids.shape[0] > max_len:
            pos_ids = pos_ids[:max_len]
        position_ids.append(pos_ids)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "position_ids": torch.stack(position_ids),
        "conv_id": [item["conv_id"] for item in batch],
        "session_name": [item["session_name"] for item in batch],
        "session_idx": [item["session_idx"] for item in batch],
        "session_dialogue": [item["session_dialogue"] for item in batch],
        "qa_pairs": [item["qa_pairs"] for item in batch],
        "is_evidence_session": [item["is_evidence_session"] for item in batch],
        "n_turns": [item["n_turns"] for item in batch],
        "data_source": [item["data_source"] for item in batch],
        "ground_truth_answers_list": [item["ground_truth_answers_list"] for item in batch],
    }