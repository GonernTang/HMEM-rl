import json
import os
import time
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm
from src.vector_store.faiss_index import FAISSIndex
from embed_manager import *
from conv_loader import *
from src.prompt import ITERATIVE_ANWSER_PROMPT_VECMEM
from mem0.embeddings.openai import OpenAIEmbedding
load_dotenv()

import logging

logger = logging.getLogger(__name__)

PROMPT = """
# Question: 
{{QUESTION}}

# Context: 
{{CONTEXT}}

# Short answer:
"""


class RAGManager:
    def __init__(self, data_path: str = None, topk: int = 10, enable_iterative_answer: bool = False):
        self.topk = topk
        if data_path is not None:
            self.data_path = data_path
        else:
            self.data_path = os.getenv("LOCOMO_PATH")

        # Initialize OpenAI client and model
        self.client = OpenAI()
        self.model = os.getenv("MODEL")
        self.embeder = OpenAIEmbedding()
        self.enable_iterative_answer = enable_iterative_answer
        self.index = FAISSIndex(
            embedding_dim=1536, index_type="Flat"
        )
        self.current_conversation: Conversation = None
        self.current_conv_embeddings: np.ndarray = None

    def generate_response(self, question, context):
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions strictly based on the provided context."
                            "The context contains a list of conversation turns. Each turn has the format:\n[SpeakerA]: [Message] [SpeakerB]: [Message] [timestamp]: [exact date and time]"
                            "--- Rules ---"
                            "1. Answer questions strictly based on the context. Do not make up information."
                            "2. If the question involves exact timing, ALWAYS use the explicit conversation timestamp as the reference point."
                            "- Convert relative expressions (e.g., 'last week', 'last year', 'next month', 'tomorrow') into absolute dates using the timestamp."
                            "- Example: If the timestamp is '20 October, 2023' and the text says 'next month', the answer must be 'November 2023'."
                            "3. If the question involves duration timing, it is ok to use relative expressions like 'last week', 'last year', 'next month', 'tomorrow'."
                            "- Example: It is ok to say 'last year' if the question is about the year before the timestamp."
                            "3. Provide the SHORTEST possible answer that DIRECTLY addresses the question."
                            "4. Avoid introducing new subjects; focus only on the requested detail."
                            "5. Use words directly from the conversation when possible.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                t2 = time.time()
                # Extract token usage information
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                return response.choices[0].message.content.strip(), t2 - t1, token_usage
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)  # Wait before retrying

    def retrieve_memory(self, question: str) -> List[int]:
        question_embedding = self.embeder.embed(question)
        _, indices = self.index.search(question_embedding, self.topk)
        return indices

    # TODO : Avoid using os.getenv directly, use passed-in parameters instead.
    def process_all_conversations(self, output_file_path):
        convs, questions = ConvLoader.load_locomo(self.data_path)
        FINAL_RESULTS = defaultdict(list)

        # Initialize stats tracking
        stats_file_path = output_file_path.replace(".json", "_stats.txt")
        total_tokens_used = 0
        conversation_stats = []  # Store stats for each conversation

        # Ensure the directory exists
        os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)
        with open(stats_file_path, "w", encoding="utf-8") as stats_file:
            stats_file.write("Token Usage Statistics by Conversation\n")
            stats_file.write("=" * 60 + "\n\n")

        for conv_idx in tqdm(range(len(convs)), desc="Processing conversations"):
            # Load relating data
            chat_history: Conversation = convs[conv_idx]
            conv_questions: List[Question] = questions[conv_idx]
            conv_embeddings: np.ndarray = EmbedManager.load_conv_embeddings(
                os.path.join(
                    os.getenv("LOCOMO_EMBEDDING_PATH"), f"Locomoconv_{conv_idx}.npy"
                )
            )
            questions_embeddings: np.ndarray = EmbedManager.load_questions_embeddings(
                os.path.join(
                    os.getenv("LOCOMO_EMBEDDING_PATH"), f"Locomoquestion_{conv_idx}.npy"
                )
            )
            conv_mapping: Dict[int, str] = EmbedManager.construct_mapping(chat_history)
            self.index = FAISSIndex(
                embedding_dim=conv_embeddings.shape[1], index_type="Flat"
            )
            self.index.load(
                os.path.join(os.getenv("LOCOMO_INDEX_PATH"), f"index_{conv_idx}.npy")
            )
            _, indices = self.index.search(questions_embeddings, self.topk)

            # Initialize conversation-level stats
            conv_tokens = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            conv_total_time = 0

            # Start processing questions
            for q_idx, item in tqdm(
                enumerate(conv_questions), desc="Answering questions", leave=False
            ):
                question: str = item.content
                answer: str = item.anwser
                category: int = item.category
                related_indices = indices[q_idx]
                context = "\n".join([conv_mapping[i] for i in related_indices])
                response, response_time, token_usage = self.generate_response(
                    question, context
                )

                # Accumulate conversation-level stats
                conv_tokens["prompt_tokens"] += token_usage["prompt_tokens"]
                conv_tokens["completion_tokens"] += token_usage["completion_tokens"]
                conv_tokens["total_tokens"] += token_usage["total_tokens"]
                conv_total_time += response_time

                FINAL_RESULTS[conv_idx].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "response": response,
                        "response_time": response_time,
                    }
                )

            # Store conversation stats and update total
            total_tokens_used += conv_tokens["total_tokens"]
            num_questions = len(conv_questions)
            conversation_stats.append(
                {
                    "conversation_id": conv_idx,
                    "num_questions": num_questions,
                    "total_tokens": conv_tokens["total_tokens"],
                    "prompt_tokens": conv_tokens["prompt_tokens"],
                    "completion_tokens": conv_tokens["completion_tokens"],
                    "total_response_time": conv_total_time,
                    "avg_response_time": (
                        conv_total_time / num_questions if num_questions > 0 else 0
                    ),
                }
            )

            # Write conversation stats to file
            with open(stats_file_path, "a", encoding="utf-8") as stats_file:
                stats_file.write(f"Conversation {conv_idx}:\n")
                stats_file.write(f"  Number of questions: {num_questions}\n")
                stats_file.write(
                    f"  Total prompt tokens: {conv_tokens['prompt_tokens']}\n"
                )
                stats_file.write(
                    f"  Total completion tokens: {conv_tokens['completion_tokens']}\n"
                )
                stats_file.write(f"  Total tokens: {conv_tokens['total_tokens']}\n")
                stats_file.write(f"  Total response time: {conv_total_time:.2f}s\n")
                stats_file.write(
                    f"  Average response time per question: {conv_total_time / num_questions:.2f}s\n"
                )
                stats_file.write("-" * 50 + "\n\n")

            # Save intermediate results
            with open(output_file_path, "w+") as f:
                json.dump(FINAL_RESULTS, f, indent=4)

        # Save results
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)

        # Write final summary to stats file
        with open(stats_file_path, "a", encoding="utf-8") as stats_file:
            stats_file.write("\n" + "=" * 60 + "\n")
            stats_file.write("FINAL SUMMARY\n")
            stats_file.write("=" * 60 + "\n")
            stats_file.write(f"Total conversations processed: {len(convs)}\n")
            stats_file.write(
                f"Total questions answered: {sum(len(q) for q in questions)}\n"
            )
            stats_file.write(f"Total tokens used: {total_tokens_used}\n")
            stats_file.write(
                f"Average tokens per conversation: {total_tokens_used / len(convs):.1f}\n"
            )
            stats_file.write(
                f"Average questions per conversation: {sum(len(q) for q in questions) / len(convs):.1f}\n"
            )
            stats_file.write(f"\nTop 5 conversations by token usage:\n")

            # Sort conversations by token usage and show top 5
            sorted_stats = sorted(
                conversation_stats, key=lambda x: x["total_tokens"], reverse=True
            )
            for i, conv_stat in enumerate(sorted_stats[:5]):
                stats_file.write(
                    f"  {i+1}. Conversation {conv_stat['conversation_id']}: {conv_stat['total_tokens']} tokens "
                    f"({conv_stat['num_questions']} questions)\n"
                )

            stats_file.write(f"\nResults saved to: {output_file_path}\n")
            stats_file.write(f"Stats saved to: {stats_file_path}\n")