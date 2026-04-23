"""
LoCoMo Dataset Preprocessing for HMEMS Consolidation Agent RL Training

This script:
1. Parses the LoCoMo dataset
2. Splits into train/validation/test (1:1:8 ratio)
3. Converts to parquet format for verl training
4. Stores QA pairs separately for reward evaluation
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def parse_conversation(conv_data: Dict) -> List[Dict[str, str]]:
    """Parse conversation dict into flat list of turns."""
    turns = []
    for key in sorted(conv_data.keys()):
        if key.startswith("session_") and isinstance(conv_data[key], list):
            turns.extend(conv_data[key])
    turns.sort(key=lambda x: x.get("dia_id", ""))
    return turns


def format_turn(turn: Dict) -> str:
    """Format a single turn as conversation text."""
    speaker = turn.get("speaker", "Unknown")
    text = turn.get("text", "")
    return f"[{speaker}]: {text}"


def conversation_to_text(conv_data: Dict) -> str:
    """Convert conversation dict to formatted text."""
    turns = parse_conversation(conv_data)
    return "\n".join(format_turn(t) for t in turns)


def extract_qa_pairs(qa_list: List[Dict], filter_adversarial: bool = True) -> List[Dict[str, str]]:
    """Extract Q&A pairs for evaluation."""
    result = []
    for qa in qa_list:
        if filter_adversarial and qa.get("category") == 5:
            continue
        answer = qa.get("answer") or qa.get("adversarial_answer", "")
        if answer is None:
            answer = ""
        result.append({
            "question": qa["question"],
            "answer": str(answer) if answer else "",
            "category": qa.get("category", 0),
            "evidence": qa.get("evidence", []),
        })
    return result


def process_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Path]:
    """Process LoCoMo dataset and split into train/val/test."""
    random.seed(seed)

    with open(input_path, "r") as f:
        dataset = json.load(f)

    n_total = len(dataset)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    print(f"Dataset split: {n_total} total, {n_train} train, {n_val} val, {n_test} test")

    indices = list(range(n_total))
    random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    splits = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
    }

    output_paths = {}
    qa_lookup = {}  # sample_id -> list of qa pairs

    # First pass: collect QA pairs by sample_id
    for item in dataset:
        sample_id = item["sample_id"]
        qa_lookup[sample_id] = extract_qa_pairs(item["qa"])

    for split_name, split_indices in splits.items():
        records = []

        for idx in split_indices:
            item = dataset[idx]
            sample_id = item["sample_id"]
            conversation = item["conversation"]
            qa_pairs = qa_lookup[sample_id]

            turns = parse_conversation(conversation)
            conv_text = conversation_to_text(conversation)

            for turn_idx, turn in enumerate(turns):
                prev_turns = turns[:turn_idx]
                prev_text = "\n".join(format_turn(t) for t in prev_turns) if prev_turns else ""
                new_memory_text = format_turn(turn)

                record = {
                    "sample_id": sample_id,
                    "turn_idx": turn_idx,
                    "dia_id": turn.get("dia_id", ""),
                    "speaker": turn.get("speaker", ""),
                    "new_memory": new_memory_text,
                    "prev_context": prev_text,
                    "n_turns": len(turns),
                    "n_qa": len(qa_pairs),
                }
                records.append(record)

        df = pd.DataFrame(records)
        output_path = Path(output_dir) / f"{split_name}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        output_paths[split_name] = output_path
        print(f"  {split_name}: {len(records)} records from {len(split_indices)} conversations")

    # Save QA lookup as JSON for reward evaluation
    qa_path = Path(output_dir) / "qa_lookup.json"
    with open(qa_path, "w") as f:
        json.dump(qa_lookup, f)
    output_paths["qa_lookup"] = qa_path
    print(f"  QA lookup saved to {qa_path}")

    return output_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess LoCoMo dataset for RL training")
    parser.add_argument("--input", type=str, default="dataset/locomo.json")
    parser.add_argument("--output-dir", type=str, default="data/hmems_consolidation")
    parser.add_argument("--train-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_paths = process_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print("\nGenerated files:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
