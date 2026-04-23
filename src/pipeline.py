from conv_loader import *
from embed_manager import *
import tqdm
import logging
from src.vector_store.faiss_index import FAISSIndex
from rag_mem import RAGManager
import pandas as pd
import json
import os
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VecMemPipeline:
    def __init__(self):
        self.conv_loader = ConvLoader()
        self.embed_manager = EmbedManager()

    def generate_embeddings_and_save(self, dataset_dir: str, save_path: str):
        """
        Generate embeddings for dataset and save them to files for reusability.

        Args:
            dataset_dir: Path to the dataset directory
            save_path: Path to save the embeddings
        """
        logger.info(f"Generating embeddings and saving to {save_path}")
        # Check if the save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        conversations, questions = self.conv_loader.load_locomo(dataset_dir)
        for i in tqdm.tqdm(range(len(conversations))):
            conv_embedding = self.embed_manager.embed_conversation(conversations[i])
            np.save(save_path + f"conv_{i}.npy", conv_embedding)
            del conv_embedding
            question_embedding = self.embed_manager.embed_questions(questions[i])
            np.save(save_path + f"question_{i}.npy", question_embedding)
            del question_embedding
        logger.info(f"Embeddings generated and saved to {save_path}")

    def construct_index_and_save(self, embeddings_dir: str, save_path: str):
        """
        Construct index for embeddings and save them to files for reusability.

        Args:
            embeddings_dir: Path to the embeddings directory
            save_path: Path to save the index
        """
        logger.info(f"Constructing index and saving to {save_path}")
        for i in tqdm.tqdm(range(0, 10)):
            conv_embedding = np.load(embeddings_dir + f"conv_{i}.npy")
            index = FAISSIndex(embedding_dim=conv_embedding.shape[1], index_type="Flat")
            index.build_index(conv_embedding)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            index.save(save_path + f"index_{i}.npy")
            del index
        logger.info(f"Index constructed and saved to {save_path}")

    @staticmethod
    def run_rag_mem(data_path: str, topk: int, save_path: str):
        rag_manager = RAGManager(data_path=data_path, topk=topk)
        rag_manager.process_all_conversations(save_path)

    @staticmethod
    def generate_eval_summary(eval_path: str, save_path: str):
        """
        Generate evaluation scores from evaluation metrics data and save results to file.

        Args:
            eval_path: Path to the evaluation metrics JSON file
            save_path: Path to save the results (without extension, will save as .txt and .json)
        """
        logger.info(f"Loading evaluation metrics from {eval_path}")

        # Category mapping
        category_mapping = {
            1: "multihop",
            2: "temporal",
            3: "open_domain",
            4: "singlehop",
            5: "adversarial",
        }

        # Load the evaluation metrics data
        with open(eval_path, "r") as f:
            data = json.load(f)

        # Flatten the data into a list of question items
        all_items = []
        for key in data:
            all_items.extend(data[key])

        # Convert to DataFrame
        df = pd.DataFrame(all_items)

        # Convert category to numeric type
        df["category"] = pd.to_numeric(df["category"])

        # Calculate mean scores by category
        result = (
            df.groupby("category")
            .agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"})
            .round(4)
        )

        # Add count of questions per category
        result["count"] = df.groupby("category").size()

        # Create a version with category names for display
        result_with_names = result.copy()
        result_with_names.index = result_with_names.index.map(
            lambda x: f"{category_mapping.get(x, f'unknown_{x}')} ({x})"
        )

        # Calculate overall means
        overall_means = df.agg(
            {"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}
        ).round(4)

        # Prepare results for saving - use original numeric categories for JSON
        results_dict = {
            "category_scores": result.to_dict(),
            "category_mapping": category_mapping,
            "overall_scores": overall_means.to_dict(),
            "total_questions": len(all_items),
        }

        # Create save directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save results as JSON
        json_path = save_path + ".json"
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        # Save results as human-readable text
        txt_path = save_path + ".txt"
        with open(txt_path, "w") as f:
            f.write("Evaluation Results\n")
            f.write("==================\n\n")

            f.write("Mean Scores Per Category:\n")
            f.write("-" * 60 + "\n")
            f.write(result_with_names.to_string())
            f.write("\n\n")

            f.write("Overall Mean Scores:\n")
            f.write("-" * 20 + "\n")
            for metric, score in overall_means.items():
                f.write(f"{metric}: {score}\n")
            f.write(f"\nTotal Questions: {len(all_items)}\n")

            f.write("\nCategory Mapping:\n")
            f.write("-" * 20 + "\n")
            for num, name in category_mapping.items():
                f.write(f"{num}: {name}\n")

        # Also print to console
        print("Mean Scores Per Category:")
        print(result_with_names)
        print("\nOverall Mean Scores:")
        print(overall_means)

        logger.info(f"Results saved to {json_path} and {txt_path}")

        return results_dict