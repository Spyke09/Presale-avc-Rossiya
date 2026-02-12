import os
import typing as tp

import pandas as pd

from src.py.settings import app_param


def check_non_intersection(
    df_technical_service: pd.DataFrame,
    interval: tp.Tuple[pd.Timestamp, pd.Timestamp],
) -> bool:
    return (
        (df_technical_service["time_start"] > interval[1])
        | (df_technical_service["time_finish"] < interval[0])
    ).all()


def make_intervals(
    df: pd.DataFrame,
    base_airports: tp.Tuple[str, ...],
    time_size: pd.Timedelta,
    df_technical_service: tp.Optional[pd.DataFrame] = None,
    for_alternatives: bool = False,
) -> tp.List[tp.Tuple[int, int]]:
    arrival_sorted = df[df["arrival_airport_code"].isin(base_airports)].sort_values(
        "arrival_time"
    )
    department_sorted = df[
        df["departure_airport_code"].isin(base_airports)
    ].sort_values("departure_time")
    arrival_date = arrival_sorted["arrival_time"].to_list()
    department_date = department_sorted["departure_time"].to_list()

    intervals = []
    i, j = 0, 0
    while i < len(arrival_date):
        while j < len(department_date) and arrival_date[i] > department_date[j]:
            j += 1
        if (
            j < len(department_date)
            and (arrival_date[i] < department_date[j])
            and (department_date[j] - arrival_date[i]) >= time_size
        ):
            interval = (
                arrival_date[i],
                department_date[j] if for_alternatives else arrival_date[i] + time_size,
            )
            if df_technical_service is None or check_non_intersection(
                df_technical_service, interval
            ):
                intervals.append(interval)
        i += 1

    return intervals


def opt_time_to_date(start: pd.Timestamp, x: int) -> pd.Timestamp:
    sec = round(x / (app_param.time_sec_to_min * app_param.time_conversion))
    delta = pd.Timedelta(seconds=sec)
    return start + delta


def opt_delta_time_to_date(x: int) -> pd.Timedelta:
    sec = round(x / (app_param.time_sec_to_min * app_param.time_conversion))
    delta = pd.Timedelta(seconds=sec)
    return delta


def save_data(
    path: str,
    format_: str,
    **dfs: pd.DataFrame,
) -> None:
    for name, df in dfs.items():
        path_out = os.path.join(path, f"{name}.csv")
        df.to_csv(path_out, sep=";", index=False, date_format=format_)
