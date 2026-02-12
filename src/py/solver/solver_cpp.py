import logging
import os.path
import subprocess
import typing as tp
from datetime import datetime

import orjson

from src.py.problem_structures import FlightDistributionProblemCPP
from src.py.settings import solver_param, app_param
from src.py.solver.base_solver import BaseFlightDistributionModel


class FlightDistributionModelCPP(BaseFlightDistributionModel):
    def __init__(
        self,
        data: FlightDistributionProblemCPP,
        timelimit: int = solver_param.timelimit,
        workers: int = solver_param.workers,
        log_verbosity: str = solver_param.log_verbosity,
        path_to_cpp_solver: str = app_param.path_to_cpp_solver,
        tmp_for_cpp_solver: str = app_param.tmp_for_cpp_solver,
        clean_tmp_folder: bool = False,
        return_first: bool = False,
        fast_comp_feasible=False,
        fast_comp_opt=True,
    ):
        super().__init__(data, timelimit, workers, log_verbosity)
        self._return_first: bool = return_first
        self._fast_comp_feasible: bool = fast_comp_feasible
        self._fast_comp_opt: bool = fast_comp_opt

        self.data: FlightDistributionProblemCPP = data
        self._path_to_cpp_solver = path_to_cpp_solver
        self._tmp_for_cpp_solver = tmp_for_cpp_solver

        self._tmp_file_name_in = f"{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}.json"
        self._tmp_file_name_out = f"out_{self._tmp_file_name_in}"
        self._tmp_file_name_in = f"in_{self._tmp_file_name_in}"

        self._clean_tmp_folder = clean_tmp_folder

        self.solution: tp.Optional[tp.List[int]] = None

        self._logger = logging.getLogger("FlightDistributionModelCPP")

    def solve(self) -> None:
        self.solution = None

        os.makedirs(os.path.dirname(self._tmp_for_cpp_solver), exist_ok=True)
        path_in = os.path.join(self._tmp_for_cpp_solver, self._tmp_file_name_in)
        path_out = os.path.join(self._tmp_for_cpp_solver, self._tmp_file_name_out)

        with open(path_in, "wb") as f:
            data = self.data.get_data_for_save()
            f.write(orjson.dumps(data))

        cmd = [
            self._path_to_cpp_solver,
            path_in,
            path_out,
            str(self.timelimit),
            str(int(self._return_first)),
            str(int(self._fast_comp_feasible)),
            str(int(self._fast_comp_opt)),
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._logger.info("Solver output:")
        for line in process.stdout:
            self._logger.info(line.rstrip().replace(" [console] [info]", ""))

        return_code = process.wait()
        self._logger.info(f"Return code: {return_code}")

        if return_code == 0 and os.path.exists(path_out):
            with open(path_out, "rb") as f:
                d = orjson.loads(f.read())
                self.solution = d["solution"]
                self._logger.info(f"Solution status: {d['status']}")
                self._logger.info(f"Objective: {d['objective']}")

            if self._clean_tmp_folder:
                os.remove(path_out)

        if self._clean_tmp_folder:
            os.remove(path_in)

    def create_model(self) -> None:
        pass

    def create_problem(self) -> None:
        pass

    def add_variables(self) -> None:
        pass

    def add_expressions(self) -> None:
        pass

    def add_obj_function(self) -> None:
        pass

    def add_constraints(self) -> None:
        pass

    def add_warm_start(self) -> None:
        pass
