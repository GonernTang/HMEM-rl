from mem0.embeddings.openai import OpenAIEmbedding
from conv_loader import *
import numpy as np
import os
from typing import List, Dict


class EmbedManager:
    def __init__(self):
        self.embedder = OpenAIEmbedding()

    def embed_conversation(self, conversation: Conversation) -> np.ndarray:
        embeddings = []
        for session in conversation.sessions:
            timestamp = session.date_time
            for i in range(0, len(session.messages), 2):
                message_a = session.messages[i]
                if i + 1 < len(session.messages):
                    message_b = session.messages[i + 1]
                    message = f"[{message_a.speaker}]: {message_a.content} [{message_b.speaker}]: {message_b.content} [timestamp]: {timestamp}"
                else:
                    message = f"[{message_a.speaker}]: {message_a.content} [timestamp]: {timestamp}"
                embedding = self.embedder.embed(message)
                embeddings.append(embedding)
        return np.array(embeddings)

    def embed_questions(self, questions: List[Question]) -> np.ndarray:
        embeddings = []
        for question in questions:
            embedding = self.embedder.embed(question.content)
            embeddings.append(embedding)
        return np.array(embeddings)

    @staticmethod
    def construct_mapping(conversation: Conversation) -> Dict[int, str]:
        mapping = {}
        counter = 0
        for session in conversation.sessions:
            timestamp = session.date_time
            for i in range(0, len(session.messages), 2):
                message_a = session.messages[i]
                if i + 1 < len(session.messages):
                    message_b = session.messages[i + 1]
                    message = f"[{message_a.speaker}]: {message_a.content} [{message_b.speaker}]: {message_b.content} [timestamp]: {timestamp}"
                else:
                    message = f"[{message_a.speaker}]: {message_a.content} [timestamp]: {timestamp}"
                mapping[counter] = message
                counter += 1
        return mapping

    @staticmethod
    def load_conv_embeddings(save_path: str) -> np.ndarray:
        if os.path.exists(save_path):
            return np.load(save_path)
        else:
            return None

    @staticmethod
    def load_questions_embeddings(save_path: str) -> np.ndarray:
        if os.path.exists(save_path):
            return np.load(save_path)
        else:
            return None