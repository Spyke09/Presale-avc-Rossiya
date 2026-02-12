import typing as tp
from collections import defaultdict

import pandas as pd

from src.py.settings import app_param
from src.py.utils import make_intervals


class DataManipulator:
    @staticmethod
    def get_start_end_time(
        df_constant: pd.DataFrame,
    ) -> tp.Tuple[pd.Timestamp, pd.Timestamp]:
        start_time = df_constant.loc[0, "start_time"]
        end_time = df_constant.loc[0, "end_time"]
        return start_time, end_time

    @staticmethod
    def get_current_solution(
        df_previous_solution: pd.DataFrame,
        df_problematic_flight_shift: pd.DataFrame,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        df_current_solution = df_previous_solution.copy()
        for _, row in df_problematic_flight_shift.iterrows():
            for col in ("arrival_time", "departure_time"):
                df_current_solution.loc[
                    df_current_solution["previous_solution_id"]
                    == row["previous_solution_id"],
                    col,
                ] = row[f"new_{col}"]

        df_current_solution = df_current_solution[
            (
                df_current_solution["arrival_time"]
                >= start_time - app_param.delta_horizont
            )
        ].copy()

        df_current_solution["is_fixed"] = False
        df_current_solution.loc[
            (df_current_solution["departure_time"] < start_time)
            | (df_current_solution["departure_time"] > end_time),
            "is_fixed",
        ] = True
        return df_current_solution

    @staticmethod
    def get_current_flight_equipments(
        df_flight_equipments: pd.DataFrame,
        df_problematic_aircraft_equipment: pd.DataFrame,
    ):
        df_current = df_flight_equipments.copy()

        equip_mapping = df_problematic_aircraft_equipment.set_index(
            "previous_solution_id"
        )["equipment_ids"]

        df_current["equipment_ids"] = (
            df_current["previous_solution_id"]
            .map(equip_mapping)
            .fillna(df_current["equipment_ids"])
        )

        def parse_equipment(x):
            if isinstance(x, str):
                return tuple(int(j) for j in x.strip("()[]").split(",") if j.strip())
            return x

        df_current["equipment_ids"] = df_current["equipment_ids"].apply(parse_equipment)

        return df_current

    @staticmethod
    def get_current_technical_service(
        df_technical_service: pd.DataFrame,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        df_current_technical_service = df_technical_service[
            (df_technical_service["time_finish"] >= start_time)
        ].copy()
        df_current_technical_service["is_fixed"] = False
        df_current_technical_service.loc[
            (df_current_technical_service["time_start"] < start_time)
            | (df_current_technical_service["time_start"] > end_time),
            "is_fixed",
        ] = True

        return df_current_technical_service

    @staticmethod
    def get_alt_current_technical_service(
        df_current_solution: pd.DataFrame,
        df_current_technical_service: pd.DataFrame,
        base_airports: tp.Tuple[str, ...],
    ) -> pd.DataFrame:
        dict_alt_current_technical_service = defaultdict(list)
        for _, row in df_current_technical_service.iterrows():
            if row["is_fixed"]:
                dict_alt_current_technical_service["aircraft_id"].append(
                    row["aircraft_id"]
                )
                dict_alt_current_technical_service["time_size"].append(row["time_size"])

                dict_alt_current_technical_service["time_start"].append(
                    row["time_start"]
                )
                dict_alt_current_technical_service["time_finish"].append(
                    row["time_finish"]
                )
                dict_alt_current_technical_service["technical_service_id"].append(
                    row["technical_service_id"]
                )
            else:
                df = df_current_solution[
                    df_current_solution["aircraft_id"] == row["aircraft_id"]
                ]
                intervals = make_intervals(
                    df, base_airports, row["time_size"], for_alternatives=True
                )
                for interval in intervals:
                    dict_alt_current_technical_service["aircraft_id"].append(
                        row["aircraft_id"]
                    )
                    dict_alt_current_technical_service["time_size"].append(
                        row["time_size"]
                    )

                    dict_alt_current_technical_service["time_start"].append(interval[0])
                    dict_alt_current_technical_service["time_finish"].append(
                        interval[1]
                    )
                    dict_alt_current_technical_service["technical_service_id"].append(
                        row["technical_service_id"]
                    )

        df_alt_current_technical_service = pd.DataFrame(
            dict_alt_current_technical_service,
            columns=[
                "technical_service_id",
                "aircraft_id",
                "time_size",
                "time_start",
                "time_finish",
            ],
        )
        df_alt_current_technical_service["alt_technical_service_id"] = (
            df_alt_current_technical_service.index
        )
        return df_alt_current_technical_service
