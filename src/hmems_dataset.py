"""
HMEMS Consolidation Agent Dataset

Custom dataset for HMEMS consolidation agent RL training,
compatible with verl's RLHFDataset interface.
"""

import json
import os
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset
import torch


class HMEMSConsolidationDataset(Dataset):
    """
    Dataset for HMEMS consolidation agent training.

    This dataset reads JSONL files containing:
    - sample_id: conversation identifier
    - turn_idx: turn index within conversation
    - new_memory: current conversation turn
    - prev_context: previous conversation context
    - n_turns, n_qa: counts

    QA pairs are loaded from a separate qa_lookup.json file.
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        processor=None,  # Compatible with verl interface
        config: Dict[str, Any] = None,
    ):
        """
        Initialize HMEMS consolidation dataset.

        Args:
            data_files: List of JSONL data file paths
            tokenizer: Tokenizer for prompt encoding
            processor: Not used (for verl interface compatibility)
            config: Dataset configuration (must contain qa_lookup_path)
        """
        self.tokenizer = tokenizer
        self.config = config or {}
        self.data = []

        # Load QA lookup from config
        qa_lookup_path = self.config.get("qa_lookup_path", "data/hmems_consolidation/qa_lookup.json")
        with open(qa_lookup_path, "r") as f:
            self.qa_lookup = json.load(f)

        # Load data from all files
        for data_file in data_files:
            with open(data_file, "r") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        self.data.append(record)

        print(f"Loaded {len(self.data)} records from {len(data_files)} files")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data item."""
        record = self.data[idx]
        sample_id = record["sample_id"]
        new_memory = record["new_memory"]
        prev_context = record.get("prev_context", "")

        # Build prompt for consolidation decision
        prompt = self._build_prompt(new_memory, prev_context)

        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.get("max_prompt_length", 2048)
        )

        # Get QA pairs for this sample
        qa_pairs = self.qa_lookup.get(sample_id, [])

        return {
            "sample_id": sample_id,
            "new_memory": new_memory,
            "prev_context": prev_context,
            "qa_pairs": qa_pairs,
            "input_ids": prompt_encoding["input_ids"].squeeze(0),
            "attention_mask": prompt_encoding["attention_mask"].squeeze(0),
            "data_source": "hmems_consolidation",
        }

    def _build_prompt(self, new_memory: str, prev_context: str) -> str:
        """Build the consolidation prompt."""
        prompt_template = """You are a memory consolidation agent. Your task is to decide how to handle new conversation memories.

## Memory System
The system has two types of memories:
1. Episodic Memory: High-level summaries of related conversation events
2. Vector Memory: Raw conversation turns stored for retrieval

## Input Format
You will receive:
- New conversation turn to store
- Previous conversation context

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

Your decision:"""
        return prompt_template.format(
            new_memory=new_memory,
            prev_context=prev_context if prev_context else "(no previous context)"
        )


def hmems_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for HMEMS dataset.

    Pads sequences to the same length within the batch.
    """
    # Find max length
    max_len = max(item["input_ids"].shape[0] for item in batch)

    # Pad sequences
    input_ids = []
    attention_masks = []

    for item in batch:
        input_ids_tensor = item["input_ids"]
        if input_ids_tensor.shape[0] < max_len:
            padding = torch.full(
                (max_len - input_ids_tensor.shape[0],),
                item["input_ids"].new_ones(1).item() if hasattr(item["input_ids"], 'new_ones') else 0
            )
            input_ids_tensor = torch.cat([input_ids_tensor, padding])
        elif input_ids_tensor.shape[0] > max_len:
            input_ids_tensor = input_ids_tensor[:max_len]

        input_ids.append(input_ids_tensor)
        attention_masks.append((input_ids_tensor != 0).long())

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "sample_id": [item["sample_id"] for item in batch],
        "new_memory": [item["new_memory"] for item in batch],
        "prev_context": [item["prev_context"] for item in batch],
        "qa_pairs": [item["qa_pairs"] for item in batch],
        "data_source": [item["data_source"] for item in batch],
    }
