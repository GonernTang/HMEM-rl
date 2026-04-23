import json
from typing import Dict, Any, List, Optional
from collections import defaultdict
from openai.types.completion_usage import CompletionUsage
from enum import Enum


class OPType(Enum):
    """
    Operation types that will consume tokens.
    """

    MEMORY_AUGMENT = 1
    ITERATIVE_FILTER = 2
    ANSWER = 3
    SEMANTIC_EXTRACTION = 4
    MEMORY_MERGE = 5
    SEMANTIC_EXTRACTION_DURING_MERGE = 6


class TokenMonitor:
    def __init__(self):
        self.total_tokens_used = 0
        self.conversation_stats = []
        self.current_conv_idx = None
        self.current_conv_tokens = None
        self.current_conv_total_messages: int = 0
        # Track tokens by operation type
        self.current_conv_op_tokens: Dict[OPType, Dict[str, int]] = None
        self.total_op_tokens: Dict[OPType, Dict[str, int]] = None

    def start_conversation(self, conv_idx: int):
        self.current_conv_idx = conv_idx
        self.current_conv_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.current_conv_total_messages = 0
        # Initialize operation-specific token tracking
        self.current_conv_op_tokens = {
            op_type: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for op_type in OPType
        }
        if self.total_op_tokens is None:
            self.total_op_tokens = {
                op_type: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                for op_type in OPType
            }

    def record_message(self):
        self.current_conv_total_messages += 1

    def record_usage_from_dict(
        self, tokens: Dict[str, int], op_type: Optional[OPType] = None
    ):
        if self.current_conv_tokens is not None:
            prompt_tokens = tokens.get("prompt_tokens", 0)
            completion_tokens = tokens.get("completion_tokens", 0)
            total_tokens = tokens.get("total_tokens", 0)

            self.current_conv_tokens["prompt_tokens"] += prompt_tokens
            self.current_conv_tokens["completion_tokens"] += completion_tokens
            self.current_conv_tokens["total_tokens"] += total_tokens
            self.total_tokens_used += total_tokens

            # Track by operation type if provided
            if op_type is not None and self.current_conv_op_tokens is not None:
                self.current_conv_op_tokens[op_type]["prompt_tokens"] += prompt_tokens
                self.current_conv_op_tokens[op_type][
                    "completion_tokens"
                ] += completion_tokens
                self.current_conv_op_tokens[op_type]["total_tokens"] += total_tokens

                self.total_op_tokens[op_type]["prompt_tokens"] += prompt_tokens
                self.total_op_tokens[op_type]["completion_tokens"] += completion_tokens
                self.total_op_tokens[op_type]["total_tokens"] += total_tokens

    def record_usage_from_raw(
        self, usage: CompletionUsage, op_type: Optional[OPType] = None
    ):
        if self.current_conv_tokens is not None:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            self.current_conv_tokens["prompt_tokens"] += prompt_tokens
            self.current_conv_tokens["completion_tokens"] += completion_tokens
            self.current_conv_tokens["total_tokens"] += total_tokens
            self.total_tokens_used += total_tokens

            # Track by operation type if provided
            if op_type is not None and self.current_conv_op_tokens is not None:
                self.current_conv_op_tokens[op_type]["prompt_tokens"] += prompt_tokens
                self.current_conv_op_tokens[op_type][
                    "completion_tokens"
                ] += completion_tokens
                self.current_conv_op_tokens[op_type]["total_tokens"] += total_tokens

                self.total_op_tokens[op_type]["prompt_tokens"] += prompt_tokens
                self.total_op_tokens[op_type]["completion_tokens"] += completion_tokens
                self.total_op_tokens[op_type]["total_tokens"] += total_tokens
        else:
            raise ValueError("No conversation started")

    def end_conversation(
        self,
        num_questions: int,
        total_time: float = 0,
        vec_mem_count: Optional[int] = None,
    ):
        if vec_mem_count is not None:
            save_ratio = vec_mem_count / self.current_conv_total_messages
        else:
            save_ratio = 0
        if self.current_conv_tokens is not None:
            # Convert operation tokens to serializable format
            op_tokens_serializable = {}
            if self.current_conv_op_tokens is not None:
                for op_type, tokens in self.current_conv_op_tokens.items():
                    op_tokens_serializable[op_type.name] = tokens

            conv_stat = {
                "conversation_id": self.current_conv_idx,
                "num_questions": num_questions,
                "total_tokens": self.current_conv_tokens["total_tokens"],
                "prompt_tokens": self.current_conv_tokens["prompt_tokens"],
                "completion_tokens": self.current_conv_tokens["completion_tokens"],
                "total_messages": self.current_conv_total_messages,
                "vec_mem_count": vec_mem_count,
                "message_save_ratio": save_ratio,
                "total_response_time": total_time,
                "avg_response_time": (
                    total_time / num_questions if num_questions > 0 else 0
                ),
                "op_tokens": op_tokens_serializable,
            }
            self.conversation_stats.append(conv_stat)

            self.current_conv_tokens = None
            self.current_conv_idx = None
            self.current_conv_total_messages = 0
            self.current_conv_op_tokens = None

    def get_summary(self) -> Dict[str, Any]:
        # Convert operation tokens to serializable format
        op_tokens_serializable = {}
        if self.total_op_tokens is not None:
            for op_type, tokens in self.total_op_tokens.items():
                op_tokens_serializable[op_type.name] = {
                    "prompt_tokens": tokens["prompt_tokens"],
                    "completion_tokens": tokens["completion_tokens"],
                    "total_tokens": tokens["total_tokens"],
                    "percentage": (
                        (tokens["total_tokens"] / self.total_tokens_used * 100)
                        if self.total_tokens_used > 0
                        else 0.0
                    ),
                }

        return {
            "total_tokens_used": self.total_tokens_used,
            "conversation_stats": self.conversation_stats,
            "total_conversations": len(self.conversation_stats),
            "total_questions": sum(
                conv["num_questions"] for conv in self.conversation_stats
            ),
            "total_messages": sum(
                conv["total_messages"] for conv in self.conversation_stats
            ),
            "average_vec_mem_count": (
                sum(conv["vec_mem_count"] for conv in self.conversation_stats)
                / len(self.conversation_stats)
                if len(self.conversation_stats) > 0
                else 0
            ),
            "average_message_save_ratio": (
                sum(conv["message_save_ratio"] for conv in self.conversation_stats)
                / len(self.conversation_stats)
                if len(self.conversation_stats) > 0
                else 0
            ),
            "op_tokens": op_tokens_serializable,
        }

    def save_stats(self, file_path: str):
        stats = self.get_summary()
        with open(file_path, "w") as f:
            json.dump(stats, f, indent=4)

    def print_summary(self):
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("TOKEN USAGE SUMMARY")
        print("=" * 60)
        print(f"Total conversations processed: {summary['total_conversations']}")
        print(f"Total questions answered: {summary['total_questions']}")
        print(f"Total tokens used: {summary['total_tokens_used']:,}")
        if summary["total_conversations"] > 0:
            print(
                f"Average tokens per conversation: {summary['total_tokens_used'] / summary['total_conversations']:.1f}"
            )
        if summary["total_questions"] > 0:
            print(
                f"Average questions per conversation: {summary['total_questions'] / summary['total_conversations']:.1f}"
            )

        # Print operation-specific token usage
        if "op_tokens" in summary and summary["op_tokens"]:
            print("\n" + "-" * 60)
            print("TOKEN USAGE BY OPERATION TYPE")
            print("-" * 60)

            # Sort by total tokens (descending)
            sorted_ops = sorted(
                summary["op_tokens"].items(),
                key=lambda x: x[1]["total_tokens"],
                reverse=True,
            )

            for op_name, op_stats in sorted_ops:
                total = op_stats["total_tokens"]
                percentage = op_stats["percentage"]
                prompt = op_stats["prompt_tokens"]
                completion = op_stats["completion_tokens"]

                # Format operation name for better readability
                formatted_name = op_name.replace("_", " ").title()

                if total > 0:
                    print(f"\n{formatted_name}:")
                    print(f"  Total tokens: {total:,} ({percentage:.2f}%)")
                    print(f"  Prompt tokens: {prompt:,}")
                    print(f"  Completion tokens: {completion:,}")
                else:
                    print(f"\n{formatted_name}: 0 tokens (0.00%)")

        print(f"\n" + "-" * 60)
        print(f"Top 5 conversations by token usage:")
        sorted_stats = sorted(
            self.conversation_stats, key=lambda x: x["total_tokens"], reverse=True
        )
        for i, conv_stat in enumerate(sorted_stats[:5]):
            print(
                f"  {i+1}. Conversation {conv_stat['conversation_id']}: {conv_stat['total_tokens']:,} tokens "
                f"({conv_stat['num_questions']} questions)"
            )
        print("=" * 60)