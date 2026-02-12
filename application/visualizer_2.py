import os.path

import matplotlib.pyplot as plt
import pandas as pd

from src.py.settings import app_param


class Visualizer:
    def __init__(self, path: str, pdf_name: str):
        self._time_step = 20
        self._path = path
        self._pdf_name = pdf_name
        self._import_data()
        self._init_min_max_time()

        self._modify_data()

    def _import_data(self):
        self._import_previous_solution()
        self._import_current_technical_service()
        self._import_df_constant()

    def _modify_data(self):
        self._df_previous_solution = self._modify_solution_or_ts(
            self._df_previous_solution
        )
        self._df_current_technical_service = self._modify_solution_or_ts(
            self._df_current_technical_service
        )

    def _init_min_max_time(self):
        min_minutes = [
            self._df_previous_solution["start"].min(),
            self._df_current_technical_service["start"].min(),
        ]

        max_minutes = [
            self._df_previous_solution["end"].max(),
            self._df_current_technical_service["end"].max(),
        ]

        self._min_time, self._max_time = min(min_minutes), max(max_minutes)
        self._min_time, self._max_time = pd.Timestamp(
            "2025-02-09 00:00:00"
        ), pd.Timestamp("2025-02-18 00:00:00")

    @staticmethod
    def _import_solution(path):
        df_solution = pd.read_csv(path, sep=";")
        df_solution["start"] = df_solution["departure_time"].apply(
            lambda x: pd.to_datetime(str(x), format="%d.%m.%Y %H:%M")
        )
        df_solution["end"] = df_solution["arrival_time"].apply(
            lambda x: pd.to_datetime(str(x), format="%d.%m.%Y %H:%M")
        )
        df_solution["aircraft_id"] = df_solution["aircraft_id"].apply(
            lambda x: f"BC {x}"
        )
        df_solution["previous_solution_id"] = df_solution["previous_solution_id"].apply(
            int
        )
        return df_solution

    @staticmethod
    def _import_technical_service(path):
        df_technical_service = pd.read_csv(path, sep=";")
        df_technical_service["start"] = df_technical_service["time_start"].apply(
            lambda x: pd.to_datetime(str(x), format="%d.%m.%Y %H:%M")
        )
        df_technical_service["end"] = df_technical_service["time_finish"].apply(
            lambda x: pd.to_datetime(str(x), format="%d.%m.%Y %H:%M")
        )
        df_technical_service["aircraft_id"] = df_technical_service["aircraft_id"].apply(
            lambda x: f"BC {x}"
        )
        return df_technical_service

    def _import_previous_solution(self):
        self._df_previous_solution = self._import_solution(
            os.path.join(self._path, "df_previous_solution.csv")
        )

    def _import_current_technical_service(self):
        self._df_current_technical_service = self._import_technical_service(
            os.path.join(self._path, "df_technical_service.csv")
        )

    def _import_df_constant(self):
        df_constant = pd.read_csv(os.path.join(self._path, "df_constant.csv"), sep=";")
        df_constant["start_time"] = df_constant["start_time"].apply(
            lambda x: pd.Timestamp(x)
        )
        df_constant["end_time"] = df_constant["end_time"].apply(
            lambda x: pd.Timestamp(x)
        )
        self._df_constant = df_constant

    def _modify_solution_or_ts(self, df) -> pd.DataFrame:
        df = df.copy()

        df["Start_minutes"] = (df["start"] - self._min_time).dt.total_seconds() / 60
        df["Duration_minutes"] = (df["end"] - df["start"]).dt.total_seconds() / 60
        return df

    def _add_df(self, df, ax, color, aircraft_order) -> None:
        delta = 5
        for i, row in df.iterrows():
            if row["aircraft_id"] not in aircraft_order:
                continue
            aircraft_index = aircraft_order.index(row["aircraft_id"]) + 1
            ax.barh(
                aircraft_index,
                row["Duration_minutes"] - delta,
                left=row["Start_minutes"],
                color=color,
            )

            if "flight_id" in row.index:
                if not app_param.debug:
                    text = f"{row['flight_id']}\n{row['departure_airport_code']}-{row['arrival_airport_code']}\n{row['start'].strftime('%H:%M')}-{row['end'].strftime('%H:%M')}"
                else:
                    text = f"{row['previous_solution_id']}\n{row['flight_id']}\n{row['start'].strftime('%d %H:%M')}"
                ax.text(
                    row["Start_minutes"] + row["Duration_minutes"] / 2,
                    aircraft_index,
                    text,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=5,
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
                    fontsize=5,
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
            df_constant["start_time"].iat[0],
            df_constant["end_time"].iat[0],
        ]:
            ax.axvline(
                (current_time - min_time).total_seconds() / 60,
                color="red",
                linestyle="solid",
                alpha=0.7,
                linewidth=3,
            )

    def visualise(self):
        max_minutes = round((self._max_time - self._min_time).total_seconds() / 60)
        fig, ax = plt.subplots(figsize=(80, 6 * 1.6))
        aircraft = sorted(
            set(self._df_previous_solution["aircraft_id"].unique().tolist()),
            key=(lambda x: 2 * int(x.split()[1])),
        )

        self._add_df(self._df_previous_solution, ax, "skyblue", aircraft)
        self._add_df(self._df_current_technical_service, ax, "lightgreen", aircraft)

        for i, row in enumerate(aircraft):
            if i % 2 == 0:  # Чередуем цвета для каждой строки
                ax.fill_betweenx(
                    [i + 0.5, i + 1.5],
                    0,
                    max_minutes,
                    color="#BFBFBF",
                    alpha=0.5,
                    zorder=0,
                )

        self._add_days_lines(ax, self._min_time, self._max_time)

        ax.set_xlabel("Время", fontweight="bold")
        ax.set_title("Диаграмма Ганта назначений ВС", fontweight="bold")

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
    scen = "scen_0"
    v = Visualizer(f"../out/13/{scen}", pdf_name=f"Картинка № 0.pdf")
    v.visualise()
