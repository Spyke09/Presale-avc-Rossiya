import os

import pandas as pd


class DataImporterForGeneration:
    def __init__(self, format_="%d.%m.%Y %H:%M"):
        self._format = format_

    def import_flights(self, path: str) -> pd.DataFrame:
        df_flights = pd.read_excel(path)

        df_flights["departure_day"] = df_flights["departure_day"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        df_flights["flight_id"] = df_flights["flight_id"].apply(str)
        df_flights["num_business_passangers"] = df_flights[
            "num_business_passangers"
        ].apply(int)
        df_flights["num_economy_passangers"] = df_flights[
            "num_economy_passangers"
        ].apply(int)
        df_flights["departure_airport_code"] = df_flights[
            "departure_airport_code"
        ].apply(str)
        df_flights["departure_term"] = df_flights["departure_term"].apply(str)
        df_flights["arrival_airport_code"] = df_flights["arrival_airport_code"].apply(
            str
        )
        df_flights["arrival_term"] = df_flights["arrival_term"].apply(str)

        df_flights["departure_time"] = df_flights["departure_time"].apply(
            lambda x: pd.Timedelta(
                hours=int(x.split(":")[0]), minutes=int(x.split(":")[-1])
            )
        )
        df_flights["arrival_time"] = df_flights["arrival_time"].apply(
            lambda x: pd.Timedelta(
                hours=int(x.split(":")[0]), minutes=int(x.split(":")[-1])
            )
        )

        a = (df_flights["departure_time"] > df_flights["arrival_time"]).apply(
            lambda x: pd.Timedelta(days=int(x))
        )
        df_flights["departure_time"] += df_flights["departure_day"]
        df_flights["arrival_time"] += df_flights["departure_day"] + a
        return df_flights

    def import_previous_solution(self, path: str) -> pd.DataFrame:
        previous_solution_dict = pd.read_excel(path, sheet_name=None)
        for key, df in previous_solution_dict.items():
            df["aircraft_id"] = int(key)
        df_prev_solution = pd.concat(previous_solution_dict.values(), ignore_index=True)
        df_prev_solution.reset_index(drop=True, inplace=True)
        try:
            df_prev_solution["departure_time"] = df_prev_solution[
                "departure_time"
            ].apply(lambda x: pd.to_datetime(str(x), format=self._format))
            df_prev_solution["arrival_time"] = df_prev_solution["arrival_time"].apply(
                lambda x: pd.to_datetime(str(x), format=self._format)
            )
        except ValueError as e:
            pass
        try:
            df_prev_solution["departure_time"] = df_prev_solution[
                "departure_time"
            ].apply(lambda x: pd.to_datetime(str(x), format="%Y-%m-%d %H:%M:00"))
            df_prev_solution["arrival_time"] = df_prev_solution["arrival_time"].apply(
                lambda x: pd.to_datetime(str(x), format="%Y-%m-%d %H:%M:00")
            )
        except ValueError as e:
            pass
        df_prev_solution["flight_id"] = df_prev_solution["flight_id"].apply(str)
        df_prev_solution["departure_airport_code"] = df_prev_solution[
            "departure_airport_code"
        ].apply(str)
        df_prev_solution["arrival_airport_code"] = df_prev_solution[
            "arrival_airport_code"
        ].apply(str)
        df_prev_solution["aircraft_id"] = df_prev_solution["aircraft_id"].apply(int)

        df_prev_solution["previous_solution_id"] = df_prev_solution.index

        assert (
            df_prev_solution["arrival_time"] > df_prev_solution["departure_time"]
        ).all()

        return df_prev_solution


class DataImporter:
    def __init__(self, path: str, format_="%d.%m.%Y %H:%M"):
        self._path = path
        self._format: str = format_

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    def import_aircraft(self):
        df_aircraft = pd.read_csv(os.path.join(self._path, "df_aircraft.csv"), sep=";")
        df_aircraft["aircraft_id"] = df_aircraft["aircraft_id"].apply(int)
        df_aircraft["reserve_q"] = df_aircraft["reserve_q"].apply(bool)
        df_aircraft["residual_resource"] = df_aircraft["residual_resource"].apply(int)
        return df_aircraft

    def import_constant(self):
        try:
            df_constant = pd.read_csv(
                os.path.join(self._path, "df_constant.csv"), sep=";"
            )
            df_constant["start_time"] = df_constant["start_time"].apply(
                lambda x: pd.to_datetime(str(x), format=self._format)
            )
            df_constant["end_time"] = df_constant["end_time"].apply(
                lambda x: pd.to_datetime(str(x), format=self._format)
            )
        except:
            df_constant = pd.read_csv(
                os.path.join(self._path, "df_constant.csv"), sep=";"
            )
            df_constant["start_time"] = df_constant["start_time"].apply(
                lambda x: pd.to_datetime(str(x), format="%d.%m.%Y %H:%M")
            )
            df_constant["end_time"] = df_constant["end_time"].apply(
                lambda x: pd.to_datetime(str(x), format="%d.%m.%Y %H:%M")
            )
        return df_constant

    def import_current_flight_equipments(self):
        df_current_flight_equipments = pd.read_csv(
            os.path.join(self._path, "df_current_flight_equipments.csv"), sep=";"
        )
        df_current_flight_equipments["flight_id"] = df_current_flight_equipments[
            "flight_id"
        ].apply(str)
        df_current_flight_equipments["previous_solution_id"] = (
            df_current_flight_equipments["previous_solution_id"].apply(int)
        )
        df_current_flight_equipments["equipment_ids"] = df_current_flight_equipments[
            "equipment_ids"
        ].apply(lambda x: tuple(str(x)[1:-1].split(", ")))
        df_current_flight_equipments["flight_equipments_id"] = (
            df_current_flight_equipments["flight_equipments_id"].apply(int)
        )
        return df_current_flight_equipments

    def import_current_solution(self):
        df_current_solution = pd.read_csv(
            os.path.join(self._path, "df_current_solution.csv"), sep=";"
        )
        df_current_solution["departure_time"] = df_current_solution[
            "departure_time"
        ].apply(lambda x: pd.to_datetime(str(x), format=self._format))
        df_current_solution["arrival_time"] = df_current_solution["arrival_time"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        df_current_solution["flight_id"] = df_current_solution["flight_id"].apply(str)
        df_current_solution["departure_airport_code"] = df_current_solution[
            "departure_airport_code"
        ].apply(str)
        df_current_solution["arrival_airport_code"] = df_current_solution[
            "arrival_airport_code"
        ].apply(str)
        df_current_solution["aircraft_id"] = df_current_solution["aircraft_id"].apply(
            int
        )
        df_current_solution["previous_solution_id"] = df_current_solution[
            "previous_solution_id"
        ].apply(int)
        df_current_solution["is_fixed"] = df_current_solution["is_fixed"].apply(bool)
        return df_current_solution

    def import_current_technical_service(self):
        df_current_technical_service = pd.read_csv(
            os.path.join(self._path, "df_current_technical_service.csv"), sep=";"
        )
        df_current_technical_service["technical_service_id"] = (
            df_current_technical_service["technical_service_id"].apply(int)
        )
        df_current_technical_service["aircraft_id"] = df_current_technical_service[
            "aircraft_id"
        ].apply(int)
        df_current_technical_service["time_size"] = df_current_technical_service[
            "time_size"
        ].apply(lambda x: pd.Timedelta(str(x)))
        df_current_technical_service["time_start"] = df_current_technical_service[
            "time_start"
        ].apply(lambda x: pd.to_datetime(str(x), format=self._format))
        df_current_technical_service["time_finish"] = df_current_technical_service[
            "time_finish"
        ].apply(lambda x: pd.to_datetime(str(x), format=self._format))
        df_current_technical_service["is_fixed"] = df_current_technical_service[
            "is_fixed"
        ].apply(bool)
        return df_current_technical_service

    def import_flight_equipments(self):
        df_flight_equipments = pd.read_csv(
            os.path.join(self._path, "df_flight_equipments.csv"), sep=";"
        )
        df_flight_equipments["flight_id"] = df_flight_equipments["flight_id"].apply(str)
        df_flight_equipments["previous_solution_id"] = df_flight_equipments[
            "previous_solution_id"
        ].apply(int)
        df_flight_equipments["equipment_ids"] = df_flight_equipments[
            "equipment_ids"
        ].apply(lambda x: tuple(int(i) for i in str(x)[1:-1].split(", ")))
        df_flight_equipments["flight_equipments_id"] = df_flight_equipments[
            "flight_equipments_id"
        ].apply(int)
        return df_flight_equipments

    def import_previous_solution(self):
        df_previous_solution = pd.read_csv(
            os.path.join(self._path, "df_previous_solution.csv"), sep=";"
        )
        df_previous_solution["departure_time"] = df_previous_solution[
            "departure_time"
        ].apply(lambda x: pd.to_datetime(str(x), format=self._format))
        df_previous_solution["arrival_time"] = df_previous_solution[
            "arrival_time"
        ].apply(lambda x: pd.to_datetime(str(x), format=self._format))
        df_previous_solution["flight_id"] = df_previous_solution["flight_id"].apply(str)
        df_previous_solution["departure_airport_code"] = df_previous_solution[
            "departure_airport_code"
        ].apply(str)
        df_previous_solution["arrival_airport_code"] = df_previous_solution[
            "arrival_airport_code"
        ].apply(str)
        df_previous_solution["aircraft_id"] = df_previous_solution["aircraft_id"].apply(
            int
        )
        df_previous_solution["previous_solution_id"] = df_previous_solution[
            "previous_solution_id"
        ].apply(int)
        return df_previous_solution

    def import_problematic_aircraft_equipment(self):
        df_problematic_aircraft_equipment = pd.read_csv(
            os.path.join(self._path, "df_problematic_aircraft_equipment.csv"), sep=";"
        )
        df_problematic_aircraft_equipment["equipment_ids"] = (
            df_problematic_aircraft_equipment["equipment_ids"].apply(
                lambda x: tuple(int(i) for i in str(x)[1:-1].split(", "))
            )
        )
        df_problematic_aircraft_equipment["previous_solution_id"] = (
            df_problematic_aircraft_equipment["previous_solution_id"].apply(int)
        )
        df_problematic_aircraft_equipment["problematic_aircraft_equipment_id"] = (
            df_problematic_aircraft_equipment[
                "problematic_aircraft_equipment_id"
            ].apply(int)
        )
        return df_problematic_aircraft_equipment

    def import_problematic_flight_shift(self):
        df_problematic_flight_shift = pd.read_csv(
            os.path.join(self._path, "df_problematic_flight_shift.csv"), sep=";"
        )
        df_problematic_flight_shift["new_arrival_time"] = df_problematic_flight_shift[
            "new_arrival_time"
        ].apply(lambda x: pd.to_datetime(str(x), format="%d.%m.%Y %H:%M"))
        df_problematic_flight_shift["new_departure_time"] = df_problematic_flight_shift[
            "new_departure_time"
        ].apply(lambda x: pd.to_datetime(str(x), format="%d.%m.%Y %H:%M"))

        df_problematic_flight_shift["aircraft_id"] = df_problematic_flight_shift[
            "aircraft_id"
        ].apply(int)
        df_problematic_flight_shift["previous_solution_id"] = (
            df_problematic_flight_shift["previous_solution_id"].apply(int)
        )
        df_problematic_flight_shift["shift"] = df_problematic_flight_shift[
            "shift"
        ].apply(lambda x: pd.Timedelta(str(x)))
        df_problematic_flight_shift["problematic_flight_shift_id"] = (
            df_problematic_flight_shift["problematic_flight_shift_id"].apply(int)
        )
        return df_problematic_flight_shift

    def import_technical_service(self):
        df_technical_service = pd.read_csv(
            os.path.join(self._path, "df_technical_service.csv"), sep=";"
        )
        df_technical_service["technical_service_id"] = df_technical_service[
            "technical_service_id"
        ].apply(int)
        df_technical_service["aircraft_id"] = df_technical_service["aircraft_id"].apply(
            int
        )
        df_technical_service["time_size"] = df_technical_service["time_size"].apply(
            lambda x: pd.Timedelta(str(x))
        )
        df_technical_service["time_start"] = df_technical_service["time_start"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        df_technical_service["time_finish"] = df_technical_service["time_finish"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        return df_technical_service

    def import_turnaround(self):
        df_turnaround = pd.read_csv(
            os.path.join(self._path, "df_turnaround.csv"), sep=";"
        )
        df_turnaround["aircraft_id"] = df_turnaround["aircraft_id"].apply(int)
        df_turnaround["airport_code"] = df_turnaround["airport_code"].apply(str)
        df_turnaround["turnaround_time"] = df_turnaround["turnaround_time"].apply(
            lambda x: pd.Timedelta(str(x))
        )
        return df_turnaround

    def import_new_solution(self):
        df_new_solution = pd.read_csv(
            os.path.join(self._path, "df_new_solution.csv"), sep=";"
        )
        df_new_solution["departure_time"] = df_new_solution["departure_time"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        df_new_solution["arrival_time"] = df_new_solution["arrival_time"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        df_new_solution["flight_id"] = df_new_solution["flight_id"].apply(str)
        df_new_solution["departure_airport_code"] = df_new_solution[
            "departure_airport_code"
        ].apply(str)
        df_new_solution["arrival_airport_code"] = df_new_solution[
            "arrival_airport_code"
        ].apply(str)
        df_new_solution["aircraft_id"] = df_new_solution["aircraft_id"].apply(int)
        df_new_solution["previous_solution_id"] = df_new_solution[
            "previous_solution_id"
        ].apply(int)

        return df_new_solution
