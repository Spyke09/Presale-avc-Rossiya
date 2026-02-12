import itertools
import typing as tp

import numpy as np
import pandas as pd

from src.py.settings import app_param
from src.py.utils import make_intervals


class DataGenerator:
    def __init__(self, seed: int):
        self._rnd = np.random.default_rng(seed)

    def generate_technical_service(
        self, df_previous_solution: pd.DataFrame, base_airports: tp.Tuple[str, ...]
    ) -> pd.DataFrame:
        df_technical_service = pd.DataFrame(
            columns=[
                "technical_service_id",
                "aircraft_id",
                "time_size",
                "time_start",
                "time_finish",
            ]
        )
        df_technical_service = self._generate_a_check(
            df_previous_solution, df_technical_service, base_airports
        )
        df_technical_service = self._generate_weekly_check(
            df_previous_solution, df_technical_service, base_airports
        )
        return df_technical_service

    def _generate_check(
        self,
        df_previous_solution: pd.DataFrame,
        df_technical_service: pd.DataFrame,
        base_airports: tp.Tuple[str, ...],
        num_days_period: int,
        check_range: tp.Tuple[int, int],
    ) -> pd.DataFrame:
        dict_technical_service_new = dict()
        dict_technical_service_new["aircraft_id"] = []
        dict_technical_service_new["time_size"] = []
        dict_technical_service_new["time_start"] = []
        dict_technical_service_new["time_finish"] = []
        for aircraft_id, df in df_previous_solution.groupby("aircraft_id"):
            min_date, max_date = df["departure_time"].min(), df["arrival_time"].max()
            num_days = (max_date - min_date).total_seconds() / 3600 / 24
            time_size = pd.Timedelta(
                hours=self._rnd.integers(check_range[0], check_range[1] + 1)
            )

            if num_days / num_days_period < self._rnd.random():
                continue

            intervals = make_intervals(
                df, base_airports, time_size, df_technical_service
            )
            if not intervals:
                continue

            alt_intervals = list(
                pd.date_range(start=min_date, end=max_date, freq=f"{num_days_period}D")
            )
            if max_date not in alt_intervals:
                alt_intervals.append(max_date)
            alt_intervals = [
                (alt_intervals[i], alt_intervals[i + 1])
                for i in range(len(alt_intervals) - 1)
            ]

            for alt_interval in alt_intervals:
                feasible_intervals = [
                    i
                    for i in intervals
                    if alt_interval[0] <= i[0] and i[1] <= alt_interval[1]
                ]
                if not feasible_intervals:
                    continue
                interval = self._rnd.choice(feasible_intervals)
                assert interval[0] <= interval[1]
                dict_technical_service_new["aircraft_id"].append(aircraft_id)
                dict_technical_service_new["time_size"].append(time_size)
                dict_technical_service_new["time_start"].append(interval[0])
                dict_technical_service_new["time_finish"].append(interval[1])

        df_technical_service = pd.concat(
            [df_technical_service, pd.DataFrame(dict_technical_service_new)],
            ignore_index=True,
        )
        df_technical_service["technical_service_id"] = df_technical_service.index
        return df_technical_service

    def _generate_weekly_check(
        self,
        df_previous_solution: pd.DataFrame,
        df_technical_service: pd.DataFrame,
        base_airports: tp.Tuple[str, ...],
    ) -> pd.DataFrame:
        return self._generate_check(
            df_previous_solution, df_technical_service, base_airports, 7, (3, 4)
        )

    def _generate_a_check(
        self,
        df_previous_solution: pd.DataFrame,
        df_technical_service: pd.DataFrame,
        base_airports: tp.Tuple[str, ...],
    ) -> pd.DataFrame:
        return self._generate_check(
            df_previous_solution, df_technical_service, base_airports, 30, (10, 15)
        )

    def generate_aircraft(
        self,
        df_previous_solution: pd.DataFrame,
        number_of_aircraft: tp.Optional[int] = None,
    ) -> pd.DataFrame:
        if number_of_aircraft is None:
            aircraft = df_previous_solution["aircraft_id"].unique().tolist()
        else:
            aircraft = list(range(1, number_of_aircraft + 1))
        df_aircraft = pd.DataFrame(columns=["aircraft_id"])
        df_aircraft["aircraft_id"] = aircraft
        df_aircraft["reserve_q"] = False
        df_aircraft["residual_resource"] = self._rnd.integers(
            100, 500, df_aircraft.shape[0]
        )
        return df_aircraft

    @staticmethod
    def generate_flight_equipments(
        df_previous_solution: pd.DataFrame,
        df_aircraft: pd.DataFrame,
    ) -> pd.DataFrame:
        df_flight_equipments = df_previous_solution[
            ["flight_id", "previous_solution_id"]
        ]
        df_flight_equipments = df_flight_equipments.reset_index(drop=True)

        # TODO: make more randomly
        aircraft = tuple(df_aircraft["aircraft_id"].unique())
        df_flight_equipments["equipment_ids"] = [
            tuple(aircraft) for _ in range(df_flight_equipments.shape[0])
        ]

        df_flight_equipments["flight_equipments_id"] = df_flight_equipments.index
        return df_flight_equipments

    def generate_problematic_aircraft_equipment(
        self,
        df_problematic_equipment: pd.DataFrame,
        df_flight_equipments: pd.DataFrame,
        df_previous_solution: pd.DataFrame,
        aircraft_id: int,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        df_problematic_equipment = df_problematic_equipment.copy()
        df_previous_solution = df_previous_solution[
            (df_previous_solution["arrival_time"] >= start)
            & (df_previous_solution["departure_time"] < end)
        ]

        dict_aircraft_problems = dict()
        dict_aircraft_problems["equipment_ids"] = []
        dict_aircraft_problems["previous_solution_id"] = []
        for i, row in df_previous_solution.iterrows():
            if (
                row["previous_solution_id"]
                in df_problematic_equipment["previous_solution_id"].values
            ):
                equipment = df_problematic_equipment.loc[
                    df_problematic_equipment["previous_solution_id"]
                    == row["previous_solution_id"],
                    "equipment_ids",
                ].iat[0]
            else:
                equipment = df_flight_equipments.loc[
                    df_flight_equipments["previous_solution_id"]
                    == row["previous_solution_id"],
                    "equipment_ids",
                ].iat[0]
            dict_aircraft_problems["equipment_ids"].append(
                tuple(i for i in equipment if i != aircraft_id)
            )
            dict_aircraft_problems["previous_solution_id"].append(
                row["previous_solution_id"]
            )

        df_aircraft_problems = pd.DataFrame(dict_aircraft_problems)
        df_aircraft_problems["problematic_aircraft_equipment_id"] = (
            df_aircraft_problems.index
        )
        return df_aircraft_problems

    def generate_problematic_flight_equipment(
        self,
        df_flight_equipments: pd.DataFrame,
        df_previous_solution: pd.DataFrame,
        number_of_problems: int,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        df_previous_solution = df_previous_solution[
            (df_previous_solution["departure_time"] >= start)
            & (df_previous_solution["departure_time"] < end)
        ]
        problems_idx = self._rnd.choice(
            df_previous_solution["previous_solution_id"],
            number_of_problems,
            replace=False,
        )
        dict_aircraft_problems = dict()
        dict_aircraft_problems["equipment_ids"] = []
        dict_aircraft_problems["previous_solution_id"] = []
        for previous_solution_id in problems_idx:
            aircraft_id = df_previous_solution.loc[
                df_previous_solution["previous_solution_id"] == previous_solution_id,
                "aircraft_id",
            ].iat[0]
            equipment_ids = df_flight_equipments.loc[
                df_flight_equipments["previous_solution_id"] == previous_solution_id,
                "equipment_ids",
            ].iat[0]
            equipment_ids = tuple(i for i in equipment_ids if i != aircraft_id)

            dict_aircraft_problems["previous_solution_id"].append(previous_solution_id)
            dict_aircraft_problems["equipment_ids"].append(equipment_ids)
        df_aircraft_problems = pd.DataFrame(dict_aircraft_problems)
        df_aircraft_problems["problematic_aircraft_equipment_id"] = (
            df_aircraft_problems.index
        )

        return df_aircraft_problems

    def generate_problematic_flight_shift(
        self,
        df_previous_solution: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        number_of_shift: int,
    ) -> pd.DataFrame:
        df_previous_solution = df_previous_solution[
            (df_previous_solution["departure_time"] >= start)
            & (df_previous_solution["departure_time"] < end)
        ]

        dict_problematic_flight_shift = dict()
        dict_problematic_flight_shift["new_arrival_time"] = []
        dict_problematic_flight_shift["new_departure_time"] = []
        dict_problematic_flight_shift["aircraft_id"] = []
        dict_problematic_flight_shift["previous_solution_id"] = []
        dict_problematic_flight_shift["shift"] = []

        flights_ids = set()

        for _ in range(number_of_shift):
            i, j = None, None
            while True:
                i = self._rnd.choice(df_previous_solution.index)
                if i in flights_ids:
                    continue
                j = self._find_flight_bundle(df_previous_solution, i, flights_ids)
                if j is not None:
                    flights_ids.add(i)
                    flights_ids.add(j)
                    break

            shift = pd.Timedelta(minutes=self._rnd.integers(5, 20))
            for f_id in (i, j):
                dict_problematic_flight_shift["new_arrival_time"].append(
                    df_previous_solution.loc[f_id, "arrival_time"] + shift
                )
                dict_problematic_flight_shift["new_departure_time"].append(
                    df_previous_solution.loc[f_id, "departure_time"] + shift
                )
                dict_problematic_flight_shift["aircraft_id"].append(
                    df_previous_solution.loc[f_id, "aircraft_id"]
                )
                dict_problematic_flight_shift["previous_solution_id"].append(
                    df_previous_solution.loc[f_id, "previous_solution_id"]
                )
                dict_problematic_flight_shift["shift"].append(shift)

        df_problematic_flight_shift = pd.DataFrame(dict_problematic_flight_shift)
        df_problematic_flight_shift["problematic_flight_shift_id"] = (
            df_problematic_flight_shift.index
        )
        return df_problematic_flight_shift

    @staticmethod
    def _find_flight_bundle(
        df_previous_solution: pd.DataFrame,
        idx_i: int,
        flights_ids: tp.Set[int],
    ) -> tp.Optional[int]:
        departure_airport_code_i = df_previous_solution.loc[
            idx_i, "departure_airport_code"
        ]
        arrival_airport_code_i = df_previous_solution.loc[idx_i, "arrival_airport_code"]
        arrival_time_i = df_previous_solution.loc[idx_i, "arrival_time"]
        departure_time_i = df_previous_solution.loc[idx_i, "departure_time"]

        df_previous_solution = df_previous_solution[
            (df_previous_solution["departure_airport_code"] == arrival_airport_code_i)
            & (df_previous_solution["arrival_airport_code"] == departure_airport_code_i)
        ]

        idx_j = None
        best_time = None
        for j, row in df_previous_solution.iterrows():
            if j in flights_ids:
                continue
            arrival_time_j = row["arrival_time"]
            departure_time_j = row["departure_time"]
            current_time = min(
                abs((arrival_time_j - departure_time_i).total_seconds()),
                abs((arrival_time_i - departure_time_j).total_seconds()),
            )
            if (idx_j is None) or (current_time < best_time):
                idx_j = j
                best_time = current_time

        if best_time and best_time <= 24 * 3600:
            return idx_j
        else:
            return None

    @staticmethod
    def generate_df_turnaround(
        df_aircraft: pd.DataFrame, df_previous_solution: pd.DataFrame
    ) -> pd.DataFrame:
        aircraft = df_aircraft["aircraft_id"].unique().tolist()
        airports = set(df_previous_solution["arrival_airport_code"].unique()) | set(
            df_previous_solution["departure_airport_code"].unique()
        )
        idx = lambda: itertools.product(aircraft, airports)
        dict_turnaround = {
            "aircraft_id": [i[0] for i in idx()],
            "airport_code": [i[1] for i in idx()],
            "turnaround_time": [
                pd.Timedelta(minutes=app_param.default_turnaround_minutes)
                for _ in idx()
            ],
        }
        return pd.DataFrame(dict_turnaround)
