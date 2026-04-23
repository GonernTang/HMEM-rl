"""
按 session 划分 LOCOMO 数据集的 QA，构建训练集和验证集

每个 session 作为一个 step，包含：
- session_dialogue: 该 session 的对话内容（turn 列表）
- qa_pairs: 该 session 对应的 QA
- is_evidence_session: 是否有 evidence QA

对于单 evidence 的 QA，划分到该 evidence 所在的 session
对于多 evidence 的 QA，划分到时间上最后出现的 evidence 所在的 session
"""

import json
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple


def parse_dia_id(dia_id: str) -> Tuple[str, int]:
    """
    解析 dia_id 如 'D1:3' -> ('session_1', 3)
    支持 'D8:6; D9:17' 格式，取最后一个
    返回 None 如果无法解析
    """
    try:
        # 处理分号分隔的多引用
        if ';' in dia_id:
            dia_id = dia_id.split(';')[-1].strip()

        parts = dia_id.split(':')
        if len(parts) != 2:
            return None, None

        session_part = parts[0]  # 'D1'
        turn_idx = int(parts[1].strip())  # 3

        if not session_part.startswith('D') or len(session_part) < 2:
            return None, None

        session_num = int(session_part[1:])  # 1
        session_name = f"session_{session_num}"
        return session_name, turn_idx
    except (ValueError, IndexError):
        return None, None


def build_conversation_sessions(conv: Dict) -> Dict[str, List[Dict]]:
    """将 conversation 构建为 session -> dialogues 的映射"""
    sessions = {}
    conv_data = conv['conversation']

    # 收集所有 session
    session_keys = sorted([k for k in conv_data.keys()
                          if k.startswith('session_') and '_date' not in k],
                         key=lambda x: int(x.split('_')[1]))

    for session_key in session_keys:
        sessions[session_key] = conv_data[session_key]

    return sessions


def assign_qa_to_session(qa: Dict, sessions: Dict[str, List[Dict]]) -> str:
    """
    根据 evidence 将 QA 划分到对应 session

    规则：
    - 单 evidence: 划分到该 evidence 所在的 session
    - 多 evidence: 划分到时间上最后出现的 evidence 所在的 session
    - 无法解析的 evidence 跳过
    """
    evidence = qa['evidence']
    if not evidence:
        return None

    # 找出时间上最后发生的 session（session 编号最大的）
    max_session_num = -1
    target_session = None

    for ev in evidence:
        session_name, _ = parse_dia_id(ev)
        if session_name:
            # 提取 session 编号
            session_num = int(session_name.split('_')[1])
            if session_num > max_session_num:
                max_session_num = session_num
                target_session = session_name

    return target_session if target_session and target_session in sessions else None


def process_conversation(conv: Dict) -> List[Dict[str, Any]]:
    """处理单个 conversation，返回所有 session steps"""
    sessions = build_conversation_sessions(conv)
    conv_id = conv.get('sample_id', conv.get('conv_id', 'unknown'))

    # 将 QA 划分到对应 session
    session_qa_map = defaultdict(list)
    for qa in conv['qa']:
        session_name = assign_qa_to_session(qa, sessions)
        if session_name:
            session_qa_map[session_name].append(qa)

    # 按顺序创建 session steps
    session_steps = []
    session_order = sorted(sessions.keys(), key=lambda x: int(x.split('_')[1]))

    for idx, session_name in enumerate(session_order):
        session_dialogues = sessions[session_name]
        session_qa_pairs = session_qa_map.get(session_name, [])

        step = {
            "conv_id": conv_id,
            "session_name": session_name,
            "session_idx": idx,
            "session_dialogue": session_dialogues,  # 保持 turn 列表格式
            "qa_pairs": session_qa_pairs,
            "is_evidence_session": len(session_qa_pairs) > 0,
            "n_turns": len(session_dialogues)
        }
        session_steps.append(step)

    return session_steps


def split_train_val_test(all_steps: List[Dict], seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按 conversation 划分训练集、验证集和测试集（保持对话完整性）"""
    random.seed(seed)

    # 按 conv_id 分组
    conv_ids = list(set(step['conv_id'] for step in all_steps))
    random.shuffle(conv_ids)

    # 1:1:8 比例 -> 10 个 conversations: train=1, val=1, test=8
    n_train = 1
    n_val = 1
    train_conv_ids = set(conv_ids[:n_train])
    val_conv_ids = set(conv_ids[n_train:n_train + n_val])
    test_conv_ids = set(conv_ids[n_train + n_val:])

    train_steps = [step for step in all_steps if step['conv_id'] in train_conv_ids]
    val_steps = [step for step in all_steps if step['conv_id'] in val_conv_ids]
    test_steps = [step for step in all_steps if step['conv_id'] in test_conv_ids]

    return train_steps, val_steps, test_conv_ids  # 返回 conv_ids 用于重建测试集原格式


def main():
    # 加载数据
    with open('dataset/locomo.json') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversations")

    # 处理所有 conversation
    all_steps = []
    for conv in data:
        steps = process_conversation(conv)
        all_steps.extend(steps)

    print(f"Total session steps: {len(all_steps)}")

    # 统计
    evidence_sessions = sum(1 for s in all_steps if s['is_evidence_session'])
    non_evidence_sessions = len(all_steps) - evidence_sessions
    print(f"Evidence sessions: {evidence_sessions}")
    print(f"Non-evidence sessions: {non_evidence_sessions}")

    # 按 conversation 划分训练集、验证集和测试集 (1:1:8)
    train_steps, val_steps, test_conv_ids = split_train_val_test(all_steps)

    print(f"\nTrain set: {len(train_steps)} session steps")
    print(f"  - Evidence: {sum(1 for s in train_steps if s['is_evidence_session'])}")
    print(f"  - Non-evidence: {sum(1 for s in train_steps if not s['is_evidence_session'])}")
    print(f"Val set: {len(val_steps)} session steps")
    print(f"  - Evidence: {sum(1 for s in val_steps if s['is_evidence_session'])}")
    print(f"  - Non-evidence: {sum(1 for s in val_steps if not s['is_evidence_session'])}")
    print(f"Test set: {len(test_conv_ids)} conversations (original LOCOMO format)")

    # 统计 QA 分布
    train_qa_count = sum(len(s['qa_pairs']) for s in train_steps)
    val_qa_count = sum(len(s['qa_pairs']) for s in val_steps)
    test_qa_count = sum(len(conv['qa']) for conv in data if conv['sample_id'] in test_conv_ids)
    print(f"\nTrain QA pairs: {train_qa_count}")
    print(f"Val QA pairs: {val_qa_count}")
    print(f"Test QA pairs: {test_qa_count}")

    # 保存
    output_dir = 'data/hmems_session_based'
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/train.jsonl', 'w') as f:
        for step in train_steps:
            f.write(json.dumps(step, ensure_ascii=False) + '\n')

    with open(f'{output_dir}/validation.jsonl', 'w') as f:
        for step in val_steps:
            f.write(json.dumps(step, ensure_ascii=False) + '\n')

    # 测试集保留原始 LOCOMO 格式
    test_original = [conv for conv in data if conv['sample_id'] in test_conv_ids]
    with open(f'{output_dir}/test.jsonl', 'w') as f:
        for conv in test_original:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"\nSaved to {output_dir}/")

    # 展示一个样本
    print("\n" + "="*80)
    print("Sample session step:")
    sample = train_steps[0]
    print(f"conv_id: {sample['conv_id']}")
    print(f"session_name: {sample['session_name']}")
    print(f"session_idx: {sample['session_idx']}")
    print(f"is_evidence_session: {sample['is_evidence_session']}")
    print(f"n_turns: {sample['n_turns']}")
    print(f"session_dialogue: list of {len(sample['session_dialogue'])} turns")
    print(f"  First turn: {sample['session_dialogue'][0]}")
    print(f"qa_pairs count: {len(sample['qa_pairs'])}")
    if sample['qa_pairs']:
        print(f"  First QA: {sample['qa_pairs'][0]['question'][:60]}...")
        print(f"  Evidence: {sample['qa_pairs'][0]['evidence']}")


if __name__ == '__main__':
    main()