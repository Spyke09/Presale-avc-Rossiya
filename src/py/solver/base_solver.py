import typing as tp
from abc import ABC, abstractmethod

import docplex.cp.model as cp_model
import docplex.cp.parameters as cp_params
import docplex.cp.solver.solver as cp_solver
import docplex.mp.model as mp_model
import pulp

from src.py.problem_structures import (
    FlightDistributionProblemBase,
    FlightDistributionProblemCP,
    FlightDistributionProblemMIP,
)


class BaseFlightDistributionModel(ABC):
    def __init__(
        self,
        data: FlightDistributionProblemBase,
        timelimit: int,
        workers: int,
        log_verbosity: str,
    ):
        super().__init__()
        self.data = data
        self.timelimit = timelimit
        self.workers = workers
        self.log_verbosity = log_verbosity

        self.model = None
        self.solver = None
        self.solution = None

    def create_problem(self) -> None:
        self.create_model()
        self.add_variables()
        self.add_expressions()
        self.add_constraints()
        self.add_obj_function()
        self.add_warm_start()

    @abstractmethod
    def create_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_variables(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_expressions(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_obj_function(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_constraints(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_warm_start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> None:
        raise NotImplementedError


class BaseFlightDistributionModelCP(BaseFlightDistributionModel, ABC):
    def __init__(
        self,
        data: FlightDistributionProblemCP,
        timelimit: int,
        workers: int,
        log_verbosity: str,
        search_type: str,
        temporal_relaxation: str,
    ):
        super().__init__(data, timelimit, workers, log_verbosity)
        self.search_type = search_type
        self.temporal_relaxation = temporal_relaxation

        self.model = None
        self.solver = None
        self.solution = None

    def create_model(self) -> None:
        self.model = cp_model.CpoModel(f"Flight distribution model")

    def solve(self) -> None:
        # CPO parameters
        params = cp_params.CpoParameters()
        params.TimeLimit = self.timelimit
        params.Workers = self.workers
        params.LogVerbosity = self.log_verbosity
        params.SearchType = self.search_type
        params.TemporalRelaxation = self.temporal_relaxation

        self.solver = cp_solver.CpoSolver(self.model, params=params)
        self.solution = self.solver.solve()


class BaseFlightDistributionModelMIP(BaseFlightDistributionModel, ABC):
    def __init__(
        self,
        data: FlightDistributionProblemMIP,
        timelimit: int,
        workers: int,
        log_verbosity: str,
    ):
        super().__init__(data, timelimit, workers, log_verbosity)

        self.model = None
        self.solver = None
        self.solution = None

    def create_model(self) -> None:
        self.model = mp_model.Model(name="Flight distribution MIP model")

    def solve(self) -> None:
        params = self.model.parameters
        params.timelimit = self.timelimit
        params.threads = self.workers
        params.mip.display = self._convert_log_verbosity(self.log_verbosity)

        self.solution = self.model.solve(log_output=True)

        if self.solution is None:
            raise ValueError("No feasible solution found")

    @staticmethod
    def _convert_log_verbosity(verbosity: str) -> int:
        verbosity_map = {"off": 0, "low": 1, "normal": 2, "high": 3, "verbose": 4}
        return verbosity_map.get(verbosity.lower(), 1)


class BaseFlightDistributionModelPulp(BaseFlightDistributionModel, ABC):
    def __init__(
        self,
        data: FlightDistributionProblemMIP,
        timelimit: int,
        workers: int,
        log_verbosity: str,
    ):
        super().__init__(data, timelimit, workers, log_verbosity)

        self.model = None
        self.solver = None
        self.solution = None

    def create_model(self) -> None:
        self.model = pulp.LpProblem("Flight_Distribution", pulp.LpMinimize)

    def solve(self) -> None:
        solver = self.solver(msg=1, timeLimit=self.timelimit)
        result_status = self.model.solve(solver)
        print(result_status)
        print(pulp.LpStatus[self.model.status])

        if result_status is None:
            raise ValueError("No feasible solution found")
