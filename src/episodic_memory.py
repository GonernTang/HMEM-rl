from typing import List

class EpisodicNote:
    """
    Represent one piece of episodic memory.
    Each memory is associated with a topic.
    The content is the main content of the episodic memory, generated from the raw_source.
    The raw_source is the raw source messages that are used to generate the main content.
    The raw_source_ids are for debugging use for now, used to test recall.
    """
    def __init__(self,raw_source: str, raw_source_ids: List[str]):
        self.raw_source: str = raw_source
        self.raw_source_ids : List[str] = raw_source_ids

    def add_raw_source(self, new_raw_source: str, new_raw_source_ids: List[str]):
        self.raw_source = f"{self.raw_source}\n{new_raw_source}"
        self.raw_source_ids.extend(new_raw_source_ids)

    def get_raw_source(self):
        return self.raw_source

    def get_raw_source_ids(self):
        return self.raw_source_ids