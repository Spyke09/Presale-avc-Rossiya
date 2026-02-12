import logging
import os
import typing as tp
from functools import lru_cache

import pandas as pd
from pydantic_settings import BaseSettings

logging.basicConfig(
    format="%(asctime)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)


class AppParam(BaseSettings):
    """
    Application parameters.
    """

    time_sec_to_min: float = 1 / 60
    time_conversion: float = 1 / 5.0
    absent_interval_var_position: int = -1
    default_turnaround_minutes: int = 45
    delta_horizont: pd.Timedelta = pd.Timedelta(days=1)
    size_horizont: pd.Timedelta = pd.Timedelta(days=3)
    start_time: pd.Timestamp = pd.Timestamp(year=2025, month=2, day=14, hour=0)
    base_airports: tp.Tuple[str, ...] = ("LED", "KJA", "VKO", "SCW", "SVO")

    path_to_cpp_solver: str = os.path.abspath("../src/cpp/build/solver.exe")
    tmp_for_cpp_solver: str = os.path.abspath("../out/tmp_for_cpp_solver")

    debug: bool = True


@lru_cache
def get_app_params() -> AppParam:
    return AppParam()


app_param = get_app_params()


class SolverParam(BaseSettings):
    """
    Solver parameters.
    """

    timelimit: int = 300
    workers: int = 4
    log_verbosity: str = "Terse"
    search_type: str = "Restart"
    temporal_relaxation: str = "On"


@lru_cache
def get_solver_params() -> SolverParam:
    return SolverParam()


solver_param = get_solver_params()
