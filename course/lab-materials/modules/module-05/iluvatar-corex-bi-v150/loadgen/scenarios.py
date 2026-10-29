"""Offline / Server arrival schedules (MLPerf-inspired)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np


ScenarioName = Literal["Offline", "Server"]


@dataclass
class Query:
    query_id: int
    issue_time_s: float


def generate_offline_schedule(num_queries: int) -> List[Query]:
    """Offline: all queries available at t=0; SUT drains ASAP."""
    return [Query(query_id=i, issue_time_s=0.0) for i in range(num_queries)]


def generate_server_schedule(
    num_queries: int,
    target_qps: float,
    arrival: str = "poisson",
    seed: int = 42,
) -> List[Query]:
    """
    Server: queries arrive over time at approximately target_qps.
    Default inter-arrival is exponential (Poisson process).
    """
    if target_qps <= 0:
        raise ValueError("target_qps must be positive")
    rng = np.random.default_rng(seed)
    queries: List[Query] = []
    t = 0.0
    for i in range(num_queries):
        queries.append(Query(query_id=i, issue_time_s=t))
        if arrival.lower() == "poisson":
            t += float(rng.exponential(1.0 / target_qps))
        elif arrival.lower() == "uniform":
            t += 1.0 / target_qps
        else:
            raise ValueError(f"unsupported arrival process: {arrival}")
    return queries
