import os

from src.py.data_processing import DataImporter, DataManipulator
from src.py.data_processing.manipulate_opt_data import DataOptManipulatorMIP
from src.py.settings import app_param
from src.py.solver import FlightDistributionModelMIP
from src.py.utils import save_data


def main(path: str, max_time_delay: int, format_="%d.%m.%Y %H:%M") -> None:
    data_importer = DataImporter(path, format_)
    df_aircraft = data_importer.import_aircraft()
    df_previous_solution = data_importer.import_previous_solution()
    df_problematic_flight_shift = data_importer.import_problematic_flight_shift()
    df_technical_service = data_importer.import_technical_service()
    df_flight_equipments = data_importer.import_flight_equipments()
    df_problematic_aircraft_equipment = (
        data_importer.import_problematic_aircraft_equipment()
    )
    df_constant = data_importer.import_constant()
    df_turnaround = data_importer.import_turnaround()

    data_manipulator = DataManipulator()

    start_time, end_time = data_manipulator.get_start_end_time(df_constant)

    df_current_solution = data_manipulator.get_current_solution(
        df_previous_solution, df_problematic_flight_shift, start_time, end_time
    )


    df_current_technical_service = data_manipulator.get_current_technical_service(
        df_technical_service, start_time, end_time
    )
    df_current_flight_equipments = data_manipulator.get_current_flight_equipments(
        df_flight_equipments, df_problematic_aircraft_equipment
    )

    save_data(
        path,
        format_,
        df_current_solution=df_current_solution,
        df_current_technical_service=df_current_technical_service,
        df_turnaround=df_turnaround,
        df_current_flight_equipments=df_current_flight_equipments,
    )

    data_manipulator = DataOptManipulatorMIP()
    problem = data_manipulator.get_data(
        start_time,
        end_time,
        df_current_solution,
        df_aircraft,
        df_current_flight_equipments,
        df_current_technical_service,
        df_turnaround,
        app_param.base_airports,
        max_time_delay
    )

    print("Aircraft number", len(problem.aircraft.tasks))
    print("Flight number", len(problem.flight_to_nodes))
    print("TS number", len(problem.ts_to_nodes))

    solver = FlightDistributionModelMIP(problem, workers=1, timelimit=300)
    solver.create_problem()
    solver.solve()

    df_new_solution, df_new_technical_service = data_manipulator.get_results(solver)

    save_data(
        path,
        format_,
        df_new_solution=df_new_solution,
        df_new_technical_service=df_new_technical_service,
    )


if __name__ == "__main__":
    import cProfile

    max_time_delay = 5
    # for i in ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4", "2_5", "2_6",
    #           "3_1", "3_2", "3_3", "3_4", "3_5", "3_6", "3_7"]:
    # for i in ["3_1"]:
    for i in ["1_3", "1_4"]:
    # for i in ["2_1", "2_2", "2_3", "2_4", "2_5", "2_6",
    #           "3_1", "3_2", "3_3", "3_4", "3_5", "3_6", "3_7"]:
        profiler = cProfile.Profile()
        profiler.enable()
        path = f"../out/r_mip/{i}"
        print(i)
        if i in ("1_3", "1_4", "1_1", "1_2"):
            main(path, max_time_delay)
        else:
            main(path, max_time_delay, format_="%Y-%m-%d %H:%M:%S")

        profiler.disable()
        profiler.dump_stats(os.path.join(path, "profile_results.prof"))
