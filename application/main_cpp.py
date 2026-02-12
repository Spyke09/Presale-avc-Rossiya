import os.path

import pandas as pd

from src.py.data_processing import DataImporter, DataManipulator
from src.py.data_processing.manipulate_opt_data import (
    DataOptManipulatorCPP,
    DataOptManipulatorMIP,
)
from src.py.settings import app_param
from src.py.solver.solver_cpp import FlightDistributionModelCPP
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

    data_manipulator_mip = DataOptManipulatorMIP()
    mip_problem = data_manipulator_mip.get_data(
        start_time,
        end_time,
        df_current_solution,
        df_aircraft,
        df_current_flight_equipments,
        df_current_technical_service,
        df_turnaround,
        app_param.base_airports,
        0
    )

    data_manipulator_cpp = DataOptManipulatorCPP()
    problem = data_manipulator_cpp.get_data(mip_problem)
    print("Aircraft number", len(mip_problem.aircraft.tasks))
    print("Flight number", len(mip_problem.flight_to_nodes))
    print("TS number", len(mip_problem.ts_to_nodes))
    solver = FlightDistributionModelCPP(
        problem,
        workers=1,
        clean_tmp_folder=True,
        timelimit=40,
        return_first=False,
        fast_comp_feasible=True,
        fast_comp_opt=False,
    )
    solver.create_problem()
    solver.solve()

    df_new_solution, df_new_technical_service = data_manipulator_cpp.get_results(
        solver, mip_problem
    )

    save_data(
        path,
        format_,
        df_new_solution=df_new_solution,
        df_new_technical_service=df_new_technical_service,
    )


if __name__ == "__main__":
    import cProfile

    for i in ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4", "2_5", "2_6",
              "3_1", "3_2", "3_3", "3_4", "3_5", "3_6", "3_7"]:
    # for i in ["2_4"]:
    # for i in ["1_1", "1_2", "1_3", "1_4"]:
    # for i in ["2_1", "2_2", "2_3", "2_4", "2_5", "2_6",
    #           "3_1", "3_2", "3_3", "3_4", "3_5", "3_6", "3_7"]:
        profiler = cProfile.Profile()
        profiler.enable()
        path = f"../out/r_bnb/{i}"
        print(i)
        if i in ("1_3", "1_4", "1_1", "1_2"):
            main(path)
        else:
            main(path, format_="%Y-%m-%d %H:%M:%S")

        profiler.disable()
        profiler.dump_stats(os.path.join(path, "profile_results.prof"))
