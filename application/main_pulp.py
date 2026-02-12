import os

import pulp
from pulp import PULP_CBC_CMD, SCIP_CMD, HiGHS_CMD

from src.py.data_processing import DataImporter, DataManipulator
from src.py.data_processing.manipulate_opt_data import DataOptManipulatorPulp
from src.py.settings import app_param
from src.py.solver import FlightDistributionModelPulp
from src.py.utils import save_data


def main(path: str, format_="%d.%m.%Y %H:%M") -> None:
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

    data_manipulator = DataOptManipulatorPulp()
    problem = data_manipulator.get_data(
        start_time,
        end_time,
        df_current_solution,
        df_aircraft,
        df_current_flight_equipments,
        df_current_technical_service,
        df_turnaround,
        app_param.base_airports,
    )

    c_solver = lambda *args, **kwargs: pulp.HiGHS_CMD(
        path=r"D:\Program Files\HiGHS\bin\highs.exe", timeLimit=120, msg=True
    )
    # c_solver = PULP_CBC_CMD
    c_solver = c_solver = lambda *args, **kwargs: pulp.SCIP_CMD(
        path=r"D:\Program Files\SCIPOptSuite 9.2.2\bin\scip.exe",
        timeLimit=120,
        msg=True,
    )
    solver = FlightDistributionModelPulp(problem, solver=c_solver, timelimit=120)
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

    profiler = cProfile.Profile()
    profiler.enable()

    path = "../out/r_scip/3_7"
    # main(path)
    main(path, format_="%Y-%m-%d %H:%M:%S")

    profiler.disable()
    profiler.dump_stats(os.path.join(path, "profile_results.prof"))
