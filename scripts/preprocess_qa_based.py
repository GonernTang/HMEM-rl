"""
QA-Based Dataset Preprocessing for HMEMS Consolidation Agent RL Training

设计思路：
1. 训练数据以 QA 对为中心
2. prev_context = evidence 中的对话（用于记忆决策）
3. question = 下游检索时的问题
4. answer = 用于计算 reward（回答正确=1）
5. memory_decision = 基于规则的 label (merge/augment/none)

Label 生成规则：
- evidence_turns == 1: none（单个对话直接存储）
- evidence_turns >= 2: augment（多个相关记忆需要合并创建 episodic memory）
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


def get_evidence_dialogue(conversation: Dict, evidence: List[str]) -> str:
    """Extract dialogue turns for given evidence dia_ids."""
    all_turns = parse_conversation(conversation)
    dia_to_turn = {t.get("dia_id"): t for t in all_turns}

    dialogue_turns = []
    for eid in evidence:
        if eid in dia_to_turn:
            dialogue_turns.append(dia_to_turn[eid])

    # 按 dia_id 排序保证顺序
    dialogue_turns.sort(key=lambda x: x.get("dia_id", ""))
    return "\n".join(format_turn(t) for t in dialogue_turns)


def get_evidence_turns(conversation: Dict, evidence: List[str]) -> List[Dict]:
    """Get list of turn dicts for given evidence dia_ids."""
    all_turns = parse_conversation(conversation)
    dia_to_turn = {t.get("dia_id"): t for t in all_turns}

    dialogue_turns = []
    for eid in evidence:
        if eid in dia_to_turn:
            dialogue_turns.append(dia_to_turn[eid])

    dialogue_turns.sort(key=lambda x: x.get("dia_id", ""))
    return dialogue_turns


def generate_memory_decision(evidence_turns: List[Dict], category: int = None) -> str:
    """
    基于经验规则生成记忆决策 label。

    规则：
    - 1 个 evidence turn: none（直接存储为 vector memory）
    - 2+ 个 evidence turns: augment（多个相关记忆需要创建 episodic memory）

    Args:
        evidence_turns: evidence 中的 turn 列表
        category: QA 的类别（可选，用于更精细的规则）

    Returns:
        memory_decision: "none" | "augment" | "merge"
    """
    n_turns = len(evidence_turns)

    if n_turns <= 1:
        return "none"
    else:
        # 多个相关记忆，需要创建 episodic memory
        return "augment"


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
    """Process LoCoMo dataset into QA-based training data."""
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

    for split_name, split_indices in splits.items():
        records = []

        for idx in split_indices:
            item = dataset[idx]
            sample_id = item["sample_id"]
            conversation = item["conversation"]
            qa_pairs = extract_qa_pairs(item["qa"])

            for qa_idx, qa in enumerate(qa_pairs):
                evidence = qa.get("evidence", [])
                evidence_turns = get_evidence_turns(conversation, evidence)

                # 构建 prev_context（evidence 中的对话）
                prev_context = get_evidence_dialogue(conversation, evidence)

                # 生成记忆决策 label
                memory_decision = generate_memory_decision(evidence_turns, qa.get("category"))

                # 构建训练样本
                record = {
                    "sample_id": sample_id,
                    "qa_idx": qa_idx,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "category": qa.get("category", 0),
                    "evidence": evidence,
                    "prev_context": prev_context,
                    "memory_decision": memory_decision,
                    "n_evidence_turns": len(evidence_turns),
                    "n_total_turns": len(parse_conversation(conversation)),
                }
                records.append(record)

        df = pd.DataFrame(records)
        output_path = Path(output_dir) / f"{split_name}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        output_paths[split_name] = output_path

        # 统计
        decision_counts = df["memory_decision"].value_counts().to_dict()
        print(f"  {split_name}: {len(records)} records from {len(split_indices)} conversations")
        print(f"    Memory decisions: {decision_counts}")

    return output_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess LoCoMo dataset for QA-based RL training")
    parser.add_argument("--input", type=str, default="dataset/locomo.json")
    parser.add_argument("--output-dir", type=str, default="data/hmems_qa_based")
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
