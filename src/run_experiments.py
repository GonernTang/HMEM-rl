import argparse
import os
from time import time
import numpy as np
from dotenv import load_dotenv
import yaml
from vec_mem import VecMem, VecMemConfig
import argparse
import concurrent.futures
import json
import threading
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm
from dotenv import load_dotenv
from src.pipeline import VecMemPipeline
from src.agent import MemoryAgent
from src.conv_loader import ConvLoader, Conversation, Question
from embed_manager import EmbedManager
from src.functions import *
import logging
from vllm import SamplingParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


load_dotenv()

TECHNIQUES = ["vecmem", "rag"]

METHODS = ["add", "search"]


def run_experiment(args, vecmem: VecMem, data_path: str, output_file_path: str, memory_agent_template: MemoryAgent=None, agent_config=None, use_agent_for_memory: bool = True):
    #load prompt prefix

            
    print(f"Iterative anwser: {vecmem.enable_iter_anwser}")
    
    #load convs and questions from dataset
    convs, questions = ConvLoader.load_locomo(data_path)
    if vecmem.conv_limit is not None:
        convs = convs[: vecmem.conv_limit]
        questions = questions[: vecmem.conv_limit]
    FINAL_RESULTS = {}

    #create token stats json file
    stats_file_path = output_file_path.replace(".json", "_token_stats.json")
    os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)


    for conv_idx in tqdm(range(len(convs)), desc="Processing conversations"):
        # if vecmem.token_monitor is not None:
        #     vecmem.token_monitor.start_conversation(conv_idx)

        chat_history: Conversation = convs[conv_idx]
        conv_questions: List[Question] = questions[conv_idx]
        conv_embedding: np.ndarray = np.load(
            os.path.join(os.getenv("LOCOMO_EMBEDDING_PATH"), f"conv_{conv_idx}.npy")
        )
        question_embedding: np.ndarray = np.load(
            os.path.join(
                os.getenv("LOCOMO_EMBEDDING_PATH"), f"question_{conv_idx}.npy"
            )
        )
        conv_mapping: Dict[int, str] = EmbedManager.construct_mapping(chat_history)

        # Init to an empty state
        vecmem.vec_store.reset()
        vecmem.aug_mem.reset()
        vecmem._payload_mapping = {}
        vecmem.id_assigner = 0

        # Initialize conversation results
        FINAL_RESULTS[str(conv_idx)] = []

        prompts = []
        # Adding memory phase:

        if not hasattr(vecmem, "_lock"):
            vecmem._lock = threading.Lock()

        use_agent = use_agent_for_memory
        agent_config = agent_config or {}
        with open('config/prompts_datasource.yaml', 'r') as f:
            prompts_datasource = yaml.safe_load(f)
        memory_template = memory_agent_template

        function_calls_log = []
        for _ in range(len(conv_embedding)):
            function_calls_log.append([])
        
        for session_idx in tqdm(range(len(conv_embedding)), desc="Adding memories", leave=False):
            raw_text = conv_mapping[session_idx]
            emb = conv_embedding[session_idx]
            executed_calls = []

            if not use_agent or memory_template is None:
                with vecmem._lock:
                    vecmem.add_memory(emb, raw_text)
                continue
            
            prompt_template = prompts_datasource.get('unified_prompt', "{context}")
            max_new_tokens = agent_config.get('max_new_tokens', 2048)
            prompt = prompt_template.format(context = raw_text, max_new_tokens=int(max_new_tokens*0.8))

            processed_text = MemoryAgent.process_text_with_qwen_pipeline(
                text=prompt,
                tokenizer=memory_template.tokenizer,
                functions=[tool['function'] for tool in get_memory_tool_schemas(vecmem)],
                status='memorie',
                enable_thinking=agent_config['enable_thinking'],
                return_text=True,
                memory=vecmem
            )


            # whether enable thinking 
            if agent_config.get('enable_thinking', False):
                thinking_budget = agent_config.get('thinking_budget', 1024)
                thinking_sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=thinking_budget,
                    stop_token_ids=[memory_template.tokenizer.eos_token_id]
                )
                outputs = memory_template.model.generate([processed_text], thinking_sampling_params)
                first_response = outputs[0].outputs[0].text
                
                need_continue = (memory_template.tokenizer.eos_token_id not in memory_template.tokenizer(first_response).input_ids 
                                 and "</think>" not in first_response)
                
                if need_continue:
                    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                    continued_text = processed_text + first_response + early_stopping_text
                    remaining_sampling_params = SamplingParams(
                        temperature=0.7,
                        max_tokens=agent_config.get('max_new_tokens', 2048) - thinking_budget,
                        stop_token_ids=[memory_template.tokenizer.eos_token_id]
                    )
                    second_outputs = memory_template.model.generate([continued_text], remaining_sampling_params)
                    final_response = first_response + early_stopping_text + second_outputs[0].outputs[0].text
                else:
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        max_tokens=agent_config.get('max_new_tokens', 2048) - thinking_budget,
                        stop_token_ids=[memory_template.tokenizer.eos_token_id]
                    )
                    outputs = memory_template.model.generate([processed_text], sampling_params)
                    final_response = outputs[0].outputs[0].text.strip()
            else:
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=agent_config.get('max_new_tokens', 2048),
                    stop_token_ids=[memory_template.tokenizer.eos_token_id]
                )
                outputs = memory_template.model.generate([processed_text], sampling_params)
                final_response = outputs[0].outputs[0].text.strip()
            
            assistant_messages = memory_template._parse_response(final_response)
            function_call_messages = [m for m in assistant_messages if m.get("function_call")]

            for function_msg in function_call_messages:
                function_call = function_msg["function_call"]
                try:
                    tool_result = memory_template._run_tool_from_function_call(function_call, vecmem)
                except Exception as e:
                    tool_result = {"status": "error", "error": str(e)}
                record = {
                    "function_call": function_call,
                    "tool_result": tool_result,
                    "session_idx": session_idx,
                    "timestamp": time.time()
                }
                function_calls_log[session_idx].append(record)
                executed_calls.append(record)
    
        FINAL_RESULTS[str(conv_idx)].append(
            {
                "function_calls": function_calls_log,
            },
        )
        FINAL_RESULTS[str(conv_idx)].append(vecmem.vec_store)
        FINAL_RESULTS[str(conv_idx)].append(vecmem.aug_mem)
        FINAL_RESULTS[str(conv_idx)].append(vecmem.semantic_memory)


        
        # ---- Fallback behavior: if no agent tool executed, do original automatic adding ----
        if not executed_calls:
            with vecmem._lock:
                vecmem.add_memory(emb, raw_text)
        else:
            pass

        # Retrieving memory phase:
        for q_idx, item in tqdm(
            enumerate(conv_questions), desc="Answering questions", leave=False
        ):
            question: str = item.content
            answer: str = item.anwser
            category: int = item.category
            response: str = vecmem.anwser_question(
                question, question_embedding[q_idx]
            )
            FINAL_RESULTS[str(conv_idx)].append(
                {
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "response": response,
                }
            )
        if vecmem.token_monitor is not None:
            vecmem.token_monitor.end_conversation(
                len(conv_questions), 0, len(vecmem._payload_mapping)
            )

    # Write all results at once in the correct format
    with open(output_file_path, "w") as f:
        json.dump(FINAL_RESULTS, f, indent=4)
    if vecmem.token_monitor is not None:
        vecmem.token_monitor.save_stats(stats_file_path)




def load_agent_config(config_path):
    """Load agent configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate required fields
    required_fields = ['agent_name', 'model_name']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file: {config_path}")

    return config


def add_arguments():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top memories to retrieve"
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="config/qwen3-4b.yaml",
        help="Path to agent configuration file",
    )
    parser.add_argument(
        "--min_aug_count",
        type=int,
        default=3,
        help="Number of top memories to retrieve",
    )
    parser.add_argument(
        "--min_relevant_score",
        type=float,
        default=0.7,
        help="Number of top memories to retrieve",
    )
    parser.add_argument(
        "--retrieve_raw_topk",
        type=int,
        default=5,
        help="Number of top memories to retrieve",
    )
    parser.add_argument(
        "--retrieve_aug_topk",
        type=int,
        default=5,
        help="Number of top memories to retrieve",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/t50055087/workspace/HMEMS/VecMem/results/Locomo/scores/debug.json",
        help="Output file name",
    )
    parser.add_argument(
        "--enable_iter_anwser",
        action="store_true",
        help="Whether to enable iterative anwser",
    )
    parser.add_argument(
        "--itr_raw_topk",
        type=int,
        default=3,
        help="Number of top memories to retrieve during iterative anwser",
    )
    parser.add_argument(
        "--itr_aug_topk",
        type=int,
        default=3,
        help="Number of top memories to retrieve during iterative anwser",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use",
    )
    parser.add_argument(
        "--merge_with_aug_thresh",
        type=float,
        default=0.85,
        help="Threshold to merge new memory with augmented memory",
    )
    parser.add_argument(
        "--enable_stat",
        action="store_true",
        help="Whether to enable token monitoring and saving ratio counting",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Whether to evaluate only",
    )
    parser.add_argument(
        "--init_env",
        action="store_true",
        help="Whether to initialize the environment",
    )
    parser.add_argument(
        "--conv_limit",
        type=int,
        default=None,
        help="Number of conversations to sample for testing",
    )
    parser.add_argument(
        "--enable_semantic_memory",
        action="store_true",
        help="Whether to enable semantic memory",
    )
    parser.add_argument(
        "--semantic_memory_topk",
        type=int,
        default=10,
        help="Number of top memories to retrieve from semantic memory",
    )
    parser.add_argument(
        "--semantic_memory_threshold",
        type=float,
        default=0.5,
        help="Threshold to consider a memory as semantic memory",
    )
    return parser.parse_args()


def process_item(item_data):
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])

        # Skip category 5
        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        local_results[k].append(
            {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                "llm_score": llm_score,
            }
        )

    return local_results


def eval_experiment(input_file: str, output_file: str):
    # Use gpt-4o-mini to evaluate the experiment
    os.environ["MODEL"] = "gpt-4o-mini"
    with open(input_file, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    # Use ThreadPoolExecutor with specified workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_item, item_data) for item_data in data.items()
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            local_results = future.result()
            with results_lock:
                for k, items in local_results.items():
                    results[k].extend(items)

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

    summary_path = output_file[:-5] + "_summary"

    VecMemPipeline.generate_eval_summary(output_file, summary_path)


def environment_setup():
    if not os.path.exists(os.getenv("LOCOMO_PATH")):
        raise ValueError(f"Locomo dataset not found at {os.getenv('LOCOMO_PATH')}")
    if not os.path.exists(os.getenv("LOCOMO_RES_PATH")):
        os.makedirs(os.getenv("LOCOMO_RES_PATH"))
    if not os.path.exists(os.getenv("LOCOMO_SCORE_PATH")):
        os.makedirs(os.getenv("LOCOMO_SCORE_PATH"))
    if not os.path.exists(os.getenv("LOCOMO_EMBEDDING_PATH")):
        os.makedirs(os.getenv("LOCOMO_EMBEDDING_PATH"))
    if not os.path.exists(os.getenv("LOCOMO_INDEX_PATH")):
        os.makedirs(os.getenv("LOCOMO_INDEX_PATH"))
    pipeline = VecMemPipeline()
    pipeline.generate_embeddings_and_save(
        os.getenv("LOCOMO_PATH"), os.getenv("LOCOMO_EMBEDDING_PATH")
    )
    pipeline.construct_index_and_save(
        os.getenv("LOCOMO_EMBEDDING_PATH"), os.getenv("LOCOMO_INDEX_PATH")
    )





def main():
    args = add_arguments()
    if args.init_env:
        environment_setup()
        return
    score_output_dir = os.getenv("LOCOMO_SCORE_PATH")

    if args.eval_only:
        anwser_file_name = args.output_file.split("/")[-1]
        score_file_name = score_output_dir + "/" + anwser_file_name
        logger.info(
            f"Evaluating experiment. Source file: {args.output_file}, Score file: {score_file_name}"
        )
        eval_experiment(args.output_file, score_file_name)
        return

    print(f"Using model: {os.getenv('MODEL1')}")

    cfg = VecMemConfig(
        min_aug_count=args.min_aug_count,
        min_relevant_score=args.min_relevant_score,
        retrieve_raw_topk=args.retrieve_raw_topk,
        retrieve_aug_topk=args.retrieve_aug_topk,
        enable_iter_anwser=args.enable_iter_anwser,
        iterative_raw_topk=args.itr_raw_topk,
        iterative_aug_topk=args.itr_aug_topk,
        merge_with_aug_thresh=args.merge_with_aug_thresh,
        conv_limit=args.conv_limit,
        enable_semantic_memory=args.enable_semantic_memory,
        semantic_memory_topk=args.semantic_memory_topk,
        semantic_memory_threshold=args.semantic_memory_threshold,
    )
    print(f"Enable iterative anwser: {cfg.enable_iter_anwser}")
    vec_mem = VecMem(cfg)
    print(f"Enable token monitoring: {args.enable_stat}")
    if args.enable_stat:
        vec_mem.enable_token_monitoring()
    

    agent_config = load_agent_config(args.agent_config)

    # Print loaded configuration
    print(f"Loaded agent configuration:")
    print(f"  Agent name: {agent_config['agent_name']}")
    print(f"  Model name: {agent_config['model_name']}")
    if 'enable_thinking' in agent_config:
        print(f"  Enable thinking: {agent_config['enable_thinking']}")
    print(f"  Save process (Qwen only): {args.save_process}")


    memory_agent_template = MemoryAgent(agent_config=agent_config, VecMemconfig=cfg)


    run_experiment(args, vec_mem, os.getenv("LOCOMO_PATH"), args.output_file, memory_agent_template, agent_config, use_agent_for_memory=True)

    # vec_mem.process_all_conversations(
    #     os.getenv("LOCOMO_PATH"),
    #     memory_agent_template,
    #     args.output_file,
    # )


    # Eval section
    anwser_file_name = args.output_file.split("/")[-1]
    score_file_name = score_output_dir + "/" + anwser_file_name
    logger.info(
        f"Evaluating experiment. Source file: {args.output_file}, Score file: {score_file_name}"
    )
    eval_experiment(args.output_file, score_file_name)


if __name__ == "__main__":
    main()