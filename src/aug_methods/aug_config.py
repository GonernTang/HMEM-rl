from dataclasses import dataclass


@dataclass
class AugConfig:
    recheck_freq: int
    augment_sim_threshold: float
    ask_coordinator: bool
    search_top_k: int