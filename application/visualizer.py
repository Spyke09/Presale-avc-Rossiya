import os.path

import matplotlib.pyplot as plt
import pandas as pd

from src.py.settings import app_param


class Visualizer:
    fontsize = 3

    def __init__(self, path: str, pdf_name: str, format_="%d.%m.%Y %H:%M"):
        self._time_step = 20
        self._path = path
        self._format = format_
        self._pdf_name = pdf_name
        self._import_data()
        self._init_min_max_time()

        self._modify_data()

    def _import_data(self):
        self._import_previous_solution()
        self._import_current_solution()
        self._import_new_solution()
        self._import_current_technical_service()
        self._import_new_technical_service()
        self._import_problematic_aircraft_equipment()
        self._import_problematic_flight_shift()
        # self._import_flight_equipments()
        self._import_df_constant()

    def _modify_data(self):
        self._modify_current_solution()
        self._df_previous_solution = self._modify_solution_or_ts(
            self._df_previous_solution
        )
        self._df_current_solution = self._modify_solution_or_ts(
            self._df_current_solution
        )
        self._df_new_solution = self._modify_solution_or_ts(self._df_new_solution)
        self._df_current_technical_service = self._modify_solution_or_ts(
            self._df_current_technical_service
        )
        self._df_new_technical_service = self._modify_solution_or_ts(
            self._df_new_technical_service
        )

    def _init_min_max_time(self):
        min_minutes = [
            self._df_new_solution["start"].min(),
            self._df_current_solution["start"].min(),
            # self._df_current_technical_service["start"].min(),
            # self._df_new_technical_service["start"].min(),
        ]

        max_minutes = [
            self._df_new_solution["end"].max(),
            self._df_current_solution["end"].max(),
            # self._df_current_technical_service["end"].max(),
            # self._df_new_technical_service["end"].max(),
        ]

        self._min_time, self._max_time = min(min_minutes), max(max_minutes)
        # self._max_time = pd.Timestamp("2025-02-18 00:00:00")

    def _import_solution(self, path):
        df_solution = pd.read_csv(path, sep=";")
        df_solution["start"] = df_solution["departure_time"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        df_solution["end"] = df_solution["arrival_time"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        df_solution["aircraft_id"] = df_solution["aircraft_id"].apply(lambda x: int(x))
        df_solution["previous_solution_id"] = df_solution["previous_solution_id"].apply(
            int
        )
        return df_solution

    def _import_technical_service(self, path):
        df_technical_service = pd.read_csv(path, sep=";")
        df_technical_service["start"] = df_technical_service["time_start"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        df_technical_service["end"] = df_technical_service["time_finish"].apply(
            lambda x: pd.to_datetime(str(x), format=self._format)
        )
        df_technical_service["aircraft_id"] = df_technical_service["aircraft_id"].apply(
            lambda x: int(x)
        )
        return df_technical_service

    def _import_previous_solution(self):
        self._df_previous_solution = self._import_solution(
            os.path.join(self._path, "df_previous_solution.csv")
        )

    def _import_current_solution(self):

        self._df_current_solution = self._import_solution(
            os.path.join(self._path, "df_current_solution.csv")
        )

    def _import_new_solution(self):
        self._df_new_solution = self._import_solution(
            os.path.join(self._path, "df_new_solution.csv")
        )

    def _import_current_technical_service(self):
        self._df_current_technical_service = self._import_technical_service(
            os.path.join(self._path, "df_current_technical_service.csv")
        )
        self._df_current_technical_service["aircraft_id"] = (
            self._df_current_technical_service["aircraft_id"].apply(
                lambda x: f"BC {x} до"
            )
        )

    def _import_new_technical_service(self):
        self._df_new_technical_service = self._import_technical_service(
            os.path.join(self._path, "df_new_technical_service.csv")
        )
        self._df_new_technical_service["aircraft_id"] = self._df_new_technical_service[
            "aircraft_id"
        ].apply(lambda x: f"BC {x} после")

    def _import_problematic_aircraft_equipment(self):
        df_problematic_aircraft_equipment = pd.read_csv(
            os.path.join(self._path, "df_problematic_aircraft_equipment.csv"), sep=";"
        )
        df_problematic_aircraft_equipment["equipment_ids"] = (
            df_problematic_aircraft_equipment["equipment_ids"].apply(
                lambda x: tuple(x[1:-1].split(", "))
            )
        )
        df_problematic_aircraft_equipment["previous_solution_id"] = (
            df_problematic_aircraft_equipment["previous_solution_id"].apply(
                lambda x: int(x)
            )
        )
        self._df_problematic_aircraft_equipment = df_problematic_aircraft_equipment

    def _import_problematic_flight_shift(self):
        df_problematic_flight_shift = pd.read_csv(
            os.path.join(self._path, "df_problematic_flight_shift.csv"), sep=";"
        )
        df_problematic_flight_shift["aircraft_id"] = df_problematic_flight_shift[
            "aircraft_id"
        ].apply(lambda x: int(x))
        df_problematic_flight_shift["start"] = df_problematic_flight_shift[
            "new_departure_time"
        ].apply(lambda x: pd.Timestamp(x))
        df_problematic_flight_shift["end"] = df_problematic_flight_shift[
            "new_arrival_time"
        ].apply(lambda x: pd.Timestamp(x))

        df_problematic_flight_shift["size"] = df_problematic_flight_shift[
            "shift"
        ].apply(lambda x: pd.Timedelta(x))

        self._df_problematic_flight_shift = df_problematic_flight_shift

    def _import_flight_equipments(self):
        df_flight_equipments = pd.read_csv(
            os.path.join(self._path, "df_flight_equipments.csv"), sep=";"
        )
        df_flight_equipments["aircraft_id"] = df_flight_equipments["aircraft_id"].apply(
            lambda x: int(x)
        )
        df_flight_equipments["equipment_ids"] = df_flight_equipments[
            "equipment_ids"
        ].apply(lambda x: tuple(int(i) for i in x[1:-1].split(", ")))
        self._df_flight_equipments = df_flight_equipments

    def _import_df_constant(self):
        try:
            df_constant = pd.read_csv(
                os.path.join(self._path, "df_constant.csv"), sep=";"
            )
            df_constant["start_time"] = df_constant["start_time"].apply(
                lambda x: pd.to_datetime(str(x), format="%Y-%m-%d %H:%M:%S")
            )
            df_constant["end_time"] = df_constant["end_time"].apply(
                lambda x: pd.to_datetime(str(x), format="%Y-%m-%d %H:%M:%S")
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
        self._df_constant = df_constant

    def _modify_solution_or_ts(self, df) -> pd.DataFrame:
        df = df.copy()

        df["Start_minutes"] = (df["start"] - self._min_time).dt.total_seconds() / 60
        df["Duration_minutes"] = (df["end"] - df["start"]).dt.total_seconds() / 60

        return df

    def _modify_current_solution(self) -> None:
        for i, row in self._df_problematic_flight_shift.iterrows():
            idx = (
                self._df_current_solution["previous_solution_id"]
                == row["previous_solution_id"]
            )
            self._df_current_solution.loc[idx, "start"] = (
                self._df_current_solution.loc[idx, "start"] - row["size"]
            )
            self._df_current_solution.loc[idx, "end"] = (
                self._df_current_solution.loc[idx, "end"] - row["size"]
            )

    d = 12

    def _add_df(self, df, ax, color, aircraft_order) -> None:
        delta = 5

        for i, row in df.iterrows():
            if row["aircraft_id"] not in aircraft_order:
                continue
            color_ = color
            if "flight_id" not in row.index or row["previous_solution_id"] == 233:
                if Visualizer.d == 0:
                    Visualizer.d += 1
                    color_ = "yellow"
                elif Visualizer.d == 1:
                    Visualizer.d += 1
                    color_ = "#fe5e21"

            aircraft_index = aircraft_order.index(row["aircraft_id"]) + 1
            ax.barh(
                aircraft_index,
                row["Duration_minutes"] - delta,
                left=row["Start_minutes"],
                color=color_,
            )

            if "flight_id" in row.index:
                if not app_param.debug:
                    text = f"{row['flight_id']}\n{row['departure_airport_code']}-{row['arrival_airport_code']}\n{row['start'].strftime('%H:%M')}-{row['end'].strftime('%H:%M')}"
                else:
                    text = f"{row['previous_solution_id']}\n{row['departure_airport_code']}-{row['arrival_airport_code']}\n{row['start'].strftime('%d %H:%M')}-{row['end'].strftime('%H:%M')}"
                ax.text(
                    row["Start_minutes"] + row["Duration_minutes"] / 2,
                    aircraft_index,
                    text,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=self.fontsize,
                )
            else:
                text = f"{(pd.Timestamp(day=1, year=2025, month=1) + (row['end'] - row['start'])).strftime('%H:%M')}"
                ax.text(
                    row["Start_minutes"] + row["Duration_minutes"] / 2,
                    aircraft_index,
                    text,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=self.fontsize,
                )

    def get_x_ticks(self, min_time: pd.Timestamp, max_time: pd.Timestamp):
        result = []
        day = pd.Timedelta(days=1)

        current_time = min_time.normalize()
        while current_time <= max_time:
            current = (current_time - self._min_time).total_seconds() / 60
            if current >= 0:
                result.append(current)
            current_time += day

        return result

    def _add_days_lines(self, ax, min_time: pd.Timestamp, max_time: pd.Timestamp):
        xticks = self.get_x_ticks(min_time, max_time)
        for x in xticks:
            ax.axvline(
                x,
                color="gray",
                linestyle="--",
                alpha=0.7,
            )

    def _add_opt_period_lines(
        self, ax, df_constant: pd.DataFrame, min_time: pd.Timestamp
    ):
        for current_time in [
            df_constant["start_time"].iloc[0],
            df_constant["end_time"].iloc[0],
        ]:
            ax.axvline(
                (current_time - min_time).total_seconds() / 60,
                color="red",
                linestyle="solid",
                alpha=0.7,
                linewidth=3,
            )

    def _split_solution(self):
        merged = self._df_new_solution.merge(
            self._df_current_solution,
            on="previous_solution_id",
            suffixes=("_df1", "_df2"),
        )
        result = merged[merged["aircraft_id_df1"] != merged["aircraft_id_df2"]][
            "previous_solution_id"
        ].tolist()
        df_new_solution = self._df_new_solution.copy()
        df_new_solution["aircraft_id_"] = df_new_solution["aircraft_id"]
        df_new_solution["aircraft_id"] = df_new_solution["aircraft_id"].apply(
            lambda x: f"BC {x} после"
        )
        df_current_solution = self._df_current_solution.copy()
        df_current_solution["aircraft_id_"] = df_current_solution["aircraft_id"]
        df_current_solution["aircraft_id"] = df_current_solution["aircraft_id"].apply(
            lambda x: f"BC {x} до"
        )

        df_solution_new_1 = df_new_solution[
            df_new_solution["previous_solution_id"].isin(result)
        ]
        df_solution_cur_1 = df_current_solution[
            df_current_solution["previous_solution_id"].isin(result)
        ]
        df_solution_new_2 = df_new_solution[
            ~df_new_solution["previous_solution_id"].isin(result)
        ]
        df_solution_cur_2 = df_current_solution[
            ~df_current_solution["previous_solution_id"].isin(result)
        ]
        idx = []
        for i, row in df_solution_cur_1.iterrows():
            if (
                row["previous_solution_id"]
                in self._df_problematic_aircraft_equipment[
                    "previous_solution_id"
                ].tolist()
            ):
                row_2 = self._df_problematic_aircraft_equipment[
                    self._df_problematic_aircraft_equipment["previous_solution_id"]
                    == row["previous_solution_id"]
                ].iloc[0]
                if str(row["aircraft_id_"]) not in row_2["equipment_ids"]:
                    idx.append(row["previous_solution_id"])

        df_solution_cur_1_a = df_solution_cur_1[
            df_solution_cur_1["previous_solution_id"].isin(idx)
        ]
        df_solution_cur_1_b = df_solution_cur_1[
            ~df_solution_cur_1["previous_solution_id"].isin(idx)
        ]
        return (
            df_solution_new_1,
            df_solution_new_2,
            df_solution_cur_1_a,
            df_solution_cur_1_b,
            df_solution_cur_2,
        )

    def visualise(self):

        max_minutes = round((self._max_time - self._min_time).total_seconds() / 60)
        fig, ax = plt.subplots(figsize=(40, 6 * 1.6))
        df_new_1, df_new_2, df_cur_1_a, df_cur_1_b, df_cur_2 = self._split_solution()

        aircraft_set = set(
            df_new_1["aircraft_id"].unique().tolist()
            + df_new_2["aircraft_id"].unique().tolist()
            + df_cur_1_a["aircraft_id"].unique().tolist()
            + df_cur_1_b["aircraft_id"].unique().tolist()
            + df_cur_2["aircraft_id"].unique().tolist()
        )
        for x in set(aircraft_set):
            aircraft_set.add(f"{x.split()[0]} {x.split()[1]} до")
            aircraft_set.add(f"{x.split()[0]} {x.split()[1]} после")

        aircraft = sorted(
            aircraft_set,
            key=(lambda x: 2 * int(x.split()[1]) + (x.split()[2] == "после") * 1),
        )

        df_technical_service_combined = pd.concat(
            [self._df_new_technical_service, self._df_current_technical_service],
            ignore_index=True,
        )

        self._add_df(df_new_1, ax, "yellow", aircraft)
        self._add_df(df_new_2, ax, "skyblue", aircraft)
        self._add_df(df_cur_1_a, ax, "#fe5e21", aircraft)
        self._add_df(df_cur_1_b, ax, "#fad725", aircraft)
        self._add_df(df_cur_2, ax, "skyblue", aircraft)
        self._add_df(df_technical_service_combined, ax, "lightgreen", aircraft)

        for i, row in enumerate(aircraft):
            if i % 4 < 2:  # Чередуем цвета для каждой строки
                ax.fill_betweenx(
                    [i + 0.5, i + 1.5],
                    0,
                    max_minutes,
                    color="#BFBFBF",
                    alpha=0.5,
                    zorder=0,
                )

        self._add_days_lines(ax, self._min_time, self._max_time)
        self._add_opt_period_lines(ax, self._df_constant, self._min_time)

        ax.set_xlabel("Время", fontweight="bold")
        ax.set_title(
            "Диаграмма Ганта назначений (цепочки АК Россия)", fontweight="bold"
        )

        xticks = self.get_x_ticks(self._min_time, self._max_time)

        xtick_labels = [
            (self._min_time + pd.Timedelta(minutes=minute)).strftime("%d.%m")
            for minute in xticks
        ]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45, fontweight="bold")

        # Установка видимых границ оси Y и исправление пустой строки
        ax.set_yticks(range(1, len(aircraft) + 1))  # Индексы с 1 до len(aircraft)
        ax.set_yticklabels(
            [f"{i}" for i in aircraft], fontweight="bold"
        )  # Метки оси Y с aircraft_id
        ax.invert_yaxis()  # Инвертируем ось Y, чтобы они шли сверху вниз

        plt.tight_layout()
        plt.savefig(os.path.join(self._path, self._pdf_name), format="pdf")


if __name__ == "__main__":
    scen = "r_bnb/2_4"
    try:
        v = Visualizer(
            f"../out/{scen}", pdf_name=f"Картинка.pdf", format_="%Y-%m-%d %H:%M:%S"
        )
        v.visualise()
    except:
        v = Visualizer(f"../out/{scen}", pdf_name=f"Картинка.pdf")
        v.visualise()
