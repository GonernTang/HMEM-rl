from dataclasses import dataclass
from typing import List, Tuple
import json
import os


@dataclass
class Message:
    speaker: str
    content: str
    timestamp: str


@dataclass
class Session:
    date_time: str
    session_id: str
    messages: List[Message]


@dataclass
class Conversation:
    id: str
    speaker_a: str
    speaker_b: str
    sessions: List[Session]


@dataclass
class Question:
    content: str
    anwser: str
    evidence: str
    category: int


class ConvLoader:
    @staticmethod
    def load_locomo(file_path: str) -> Tuple[List[Conversation], List[List[Question]]]:
        """
        Load the conversations from the locomo dataset. And keep in memory
        """
        conversations = []
        questions = []
        with open(file_path) as f:
            dataset = json.load(f)
            for item in dataset:
                # Load questions
                ques_for_conv = []
                for q in item["qa"]:
                    if q["category"] == 5:
                        continue
                    ques_for_conv.append(
                        Question(
                            content=q["question"],
                            anwser=q["answer"],
                            evidence=q["evidence"],
                            category=q["category"],
                        )
                    )
                questions.append(ques_for_conv)
                # Load conversations
                sessions = []
                conv_id = item["sample_id"]
                conversation = Conversation(
                    id=conv_id,
                    sessions=sessions,
                    speaker_a=item["conversation"]["speaker_a"],
                    speaker_b=item["conversation"]["speaker_b"],
                )
                for key, conv in item["conversation"].items():
                    if key.startswith("speaker") or key.endswith("time"):
                        continue
                    time_key = key + "_date_time"
                    session_time = item["conversation"][time_key]
                    session_id = key.split("_")[1]
                    session_messages = []
                    for msg in conv:
                        session_messages.append(
                            Message(
                                speaker=msg["speaker"],
                                content=msg["text"],
                                timestamp=msg["dia_id"],
                            )
                        )
                    sessions.append(
                        Session(
                            date_time=session_time,
                            session_id=session_id,
                            messages=session_messages,
                        )
                    )
                conversations.append(conversation)

        return conversations, questions