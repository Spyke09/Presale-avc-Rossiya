import os.path
import typing as tp

import pandas as pd

from src.py.data_processing import (
    DataGenerator,
    DataImporter,
    DataImporterForGeneration,
)
from src.py.settings import app_param
from src.py.utils import save_data


def main(
    path_in: str,
    path_out: str,
    seed: int = 10,
    number_equipment_problem: int = 0,
    number_shift_problem: int = 0,
    broken_aircraft: tp.Optional[
        tp.List[tp.Tuple[int, pd.Timestamp, pd.Timestamp]]
    ] = None,
    import_flight_equipments_q: bool = False,
) -> None:
    start_time = app_param.start_time
    end_time = app_param.start_time + app_param.size_horizont

    data_importer_gen = DataImporterForGeneration()

    df_previous_solution = data_importer_gen.import_previous_solution(
        os.path.join(path_in, "test solution.xlsx")
    )

    data_generator = DataGenerator(seed)

    df_aircraft = data_generator.generate_aircraft(df_previous_solution)

    if import_flight_equipments_q:
        data_importer = DataImporter(path_out)
        df_flight_equipments = data_importer.import_flight_equipments()
    else:
        df_flight_equipments = data_generator.generate_flight_equipments(
            df_previous_solution, df_aircraft
        )

    base_airports = app_param.base_airports
    df_technical_service = data_generator.generate_technical_service(
        df_previous_solution, base_airports
    )

    df_problematic_equipment = data_generator.generate_problematic_flight_equipment(
        df_flight_equipments,
        df_previous_solution,
        number_equipment_problem,
        start_time,
        start_time + pd.Timedelta(days=1),
    )

    if broken_aircraft is not None:
        for i in broken_aircraft:
            df_problematic_equipment = (
                data_generator.generate_problematic_aircraft_equipment(
                    df_problematic_equipment,
                    df_flight_equipments,
                    df_previous_solution,
                    i[0],
                    i[1],
                    i[2],
                )
            )

    df_problematic_flight_shift = data_generator.generate_problematic_flight_shift(
        df_previous_solution, start_time, end_time, number_shift_problem
    )

    df_turnaround = data_generator.generate_df_turnaround(
        df_aircraft, df_previous_solution
    )

    save_data(
        path_out,
        # df_aircraft=df_aircraft,
        # df_constant=pd.DataFrame({"start_time": [start_time], "end_time": [end_time]}),
        # df_flight_equipments=df_flight_equipments,
        # df_previous_solution=df_previous_solution,
        df_problematic_aircraft_equipment=df_problematic_equipment,
        # df_problematic_flight_shift=df_problematic_flight_shift,
        # df_technical_service=df_technical_service,
        # df_turnaround=df_turnaround,
    )


if __name__ == "__main__":
    path_in_ = "../data/13"
    path_out_ = "../out/13/scen_3_5"
    broken_aircraft_ = [
        (
            89106,
            pd.Timestamp(year=2025, month=2, day=14, hour=0, minute=0),
            pd.Timestamp(year=2025, month=2, day=16, hour=12, minute=0),
        )
    ]
    main(
        path_in_,
        path_out_,
        seed=1234,
        number_equipment_problem=0,
        number_shift_problem=0,
        broken_aircraft=broken_aircraft_,
        import_flight_equipments_q=True,
    )
