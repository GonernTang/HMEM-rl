"""
Consolidation Tool Schemas for HMEMS

定义三个 consolidation tools，让 RL agent 通过 tool calling 选择：
- consolidate_merge: 合并新记忆到现有情节记忆
- consolidate_augment: 从相关原始记忆创建新的情节记忆
- consolidate_none: 直接添加原始记忆（不触发 merge/augment）

这些工具操作真实的 VecMem 记忆存储。
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.episodic_memory import EpisodicNote
from src.prompt import MEMORY_AUGMENT_MERGE_PROMPT


@dataclass
class Parameter:
    """Represents a function parameter with its type and requirements."""
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


class ConsolidationToolFunction:
    """Base class for defining consolidation tools in a human-friendly way."""

    name: str
    description: str
    parameters: List[Parameter]

    @classmethod
    def to_schema(cls) -> Dict[str, Any]:
        """Convert the function definition to OpenAI tool schema format."""
        properties = {}
        required = []

        for param in cls.parameters:
            param_schema = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                param_schema["enum"] = param.enum
            properties[param.name] = param_schema
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @classmethod
    def execute(cls, vecmem: 'VecMem', args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given arguments.

        Args:
            vecmem: VecMem instance (the real memory store)
            args: Tool arguments
        """
        raise NotImplementedError("Subclasses must implement execute()")

    @staticmethod
    def _safe_extract_json(response_content: str) -> dict:
        """Safely extract JSON from response content."""
        if not response_content:
            return {}

        content = response_content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) > 2:
                content = "\n".join(lines[1:-1])

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}


class ConsolidateMerge(ConsolidationToolFunction):
    """
    Merge new memory with existing episodic memory.

    Use when the new memory is thematically related to an existing episodic memory
    and should be combined into a single, more comprehensive memory.
    The system will call LLM to generate the merged content.
    """
    name = "consolidate_merge"
    description = "将新记忆与现有情节记忆合并。当新记忆与某个情节记忆主题相关时使用。系统会调用LLM生成合并后的内容。"
    parameters = [
        Parameter(
            name="new_memory",
            type="string",
            description="新记忆的内容，要合并的新对话或信息"
        ),
        Parameter(
            name="target_episodic_id",
            type="integer",
            description="目标情节记忆的ID，要合并到哪个已有情节记忆中"
        ),
    ]

    @classmethod
    def execute(cls, vecmem: 'VecMem', args: Dict[str, Any]) -> Dict[str, Any]:
        new_memory = args.get("new_memory", "")
        target_id = args.get("target_episodic_id")

        try:
            # 获取目标情节记忆
            episodic_note = vecmem.aug_mem.get_episodic_note(target_id)
            if episodic_note is None:
                return {"status": "error", "message": f"Episodic memory {target_id} not found"}

            past_memory = episodic_note.raw_conv

            # 直接调用 LLM 合并新旧记忆（不使用 try_merge_new_memory，因为它会搜索最相似的）
            response = vecmem.openai_client1.chat.completions.create(
                model=os.getenv("MODEL1"),
                messages=[
                    {"role": "system", "content": MEMORY_AUGMENT_MERGE_PROMPT},
                    {"role": "user", "content": f"<New memory>: {new_memory}\n<Past memory>: {past_memory}"}
                ],
                temperature=0.0,
            )

            content = response.choices[0].message.content
            response_json = cls._safe_extract_json(content)

            should_merge = response_json.get("should_merge", "no")
            if should_merge != "yes":
                return {
                    "status": "skipped",
                    "action": "merge",
                    "message": "LLM decided not to merge",
                    "target_episodic_id": target_id,
                }

            merged_content = response_json.get("merged_memory", "")
            if not merged_content:
                return {
                    "status": "error",
                    "message": "LLM returned merge=yes but no merged_memory"
                }

            # 生成新的 embedding
            merged_embedding = np.array(vecmem.embeder.embed(merged_content))

            # 删除旧的情节记忆
            vecmem.aug_mem.vec_store.remove([target_id])

            # 添加新的合并记忆
            new_id = vecmem.id_assigner
            vecmem.aug_mem.vec_store.add(merged_embedding, merged_content, new_id)
            vecmem.aug_mem.raw_store[new_id] = EpisodicNote(merged_content, [target_id])
            vecmem.id_assigner += 1

            return {
                "status": "ok",
                "action": "merge",
                "memory_id": new_id,
                "content": merged_content,
                "merged_from_id": target_id,
                "memory_length": len(merged_content),
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}


class ConsolidateAugment(ConsolidationToolFunction):
    """
    Create new episodic memory from relevant raw memories.

    Use when multiple raw memories are thematically related but not enough
    to merge with an existing episodic memory. Creates a new, higher-level
    episodic memory. The system will call LLM to generate the augmented content.
    """
    name = "consolidate_augment"
    description = "从多个相关原始记忆生成新的情节记忆。当多条原始记忆主题相关但不足以合并到已有情节记忆时使用。系统会调用LLM生成新的情节记忆。"
    parameters = [
        Parameter(
            name="relevant_vec_ids",
            type="array",
            description="相关原始记忆的ID列表，这些记忆将被组合成新的情节记忆"
        ),
        Parameter(
            name="new_memory",
            type="string",
            description="新记忆的内容，将与相关记忆一起被综合"
        ),
    ]

    @classmethod
    def execute(cls, vecmem: 'VecMem', args: Dict[str, Any]) -> Dict[str, Any]:
        relevant_ids = args.get("relevant_vec_ids", [])
        new_memory = args.get("new_memory", "")

        try:
            # 获取相关原始记忆的内容
            raw_contents = []
            for vid in relevant_ids:
                if vid in vecmem.aug_mem.raw_store:
                    episodic_note = vecmem.aug_mem.get_episodic_note(vid)
                    if episodic_note:
                        raw_contents.append(episodic_note.raw_conv)

            # 添加新记忆的内容
            if new_memory:
                raw_contents.append(new_memory)

            if not raw_contents:
                return {"status": "error", "message": "No relevant memories found"}

            # 拼接成 cat_raw_memories 格式
            cat_raw_memories = "<END_OF_CONV>".join(raw_contents)

            # 调用 NaiveAugMem.add 生成增强记忆
            # 这个方法会调用 LLM 生成情节记忆
            episodic_memories = vecmem.aug_mem.add(
                cat_raw_memories, vecmem.id_assigner, list(relevant_ids)
            )

            if episodic_memories:
                # 使用 LLM 返回的第一个增强记忆
                augmented_content = episodic_memories[0]
                new_id = vecmem.aug_mem.id_counter - 1

                # 从向量存储中删除被消费的记忆
                if relevant_ids:
                    vecmem.aug_mem.vec_store.remove(list(relevant_ids))

                return {
                    "status": "ok",
                    "action": "augment",
                    "memory_id": new_id,
                    "content": augmented_content,
                    "consumed_vec_ids": relevant_ids,
                    "memory_length": len(augmented_content),
                }
            else:
                return {
                    "status": "error",
                    "message": "LLM failed to generate augmented memory"
                }

        except Exception as e:
            return {"status": "error", "message": str(e)}


class ConsolidateNone(ConsolidationToolFunction):
    """
    Store raw memory without consolidation.

    Use when there are no relevant memories to consolidate with,
    or when the new memory is unrelated to existing memories.
    The memory is stored as-is without calling LLM for merge/augment.
    """
    name = "consolidate_none"
    description = "不进行任何合并增强，直接将新记忆作为原始记忆存储。当没有相关记忆或不适合合并时使用。"
    parameters = [
        Parameter(
            name="new_memory",
            type="string",
            description="新记忆的内容，直接作为原始记忆存储"
        ),
    ]

    @classmethod
    def execute(cls, vecmem: 'VecMem', args: Dict[str, Any]) -> Dict[str, Any]:
        new_memory = args.get("new_memory", "")

        try:
            if not new_memory:
                return {"status": "error", "message": "new_memory is required"}

            # 生成 embedding
            embedding = np.array(vecmem.embeder.embed(new_memory))

            # 添加到 FlatIndex (VecMem.vec_store)
            memory_id = vecmem.id_assigner
            vecmem.vec_store.add(embedding, memory_id)

            # 同时添加到 aug_mem.raw_store 作为引用
            # 这样后续可以被 augment 使用
            vecmem.aug_mem.raw_store[memory_id] = EpisodicNote(new_memory, [])

            vecmem.id_assigner += 1

            return {
                "status": "ok",
                "action": "none",
                "memory_id": memory_id,
                "content": new_memory,
                "memory_length": len(new_memory),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


# List of all available consolidation tools
CONSOLIDATION_TOOLS = [
    ConsolidateMerge,
    ConsolidateAugment,
    ConsolidateNone,
]


# Tool implementations map
TOOL_IMPLS = {
    func.name: func.execute for func in CONSOLIDATION_TOOLS
}


def get_consolidation_tool_schemas() -> List[Dict[str, Any]]:
    """Generate OpenAI tool schemas for consolidation functions."""
    return [tool.to_schema() for tool in CONSOLIDATION_TOOLS]
