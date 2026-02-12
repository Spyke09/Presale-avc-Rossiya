import logging
import os.path
import subprocess
import typing as tp
from collections import defaultdict
from datetime import datetime

import orjson

from src.py.problem_structures import FlightDistributionProblemCPP
from src.py.settings import solver_param, app_param
from src.py.solver.base_solver import BaseFlightDistributionModel


class FlightDistributionModelMaxSAT(BaseFlightDistributionModel):
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
    ):
        super().__init__(data, timelimit, workers, log_verbosity)
        self._return_first: bool = return_first
        self.data: FlightDistributionProblemCPP = data
        self._path_to_cpp_solver = path_to_cpp_solver
        self._tmp_for_cpp_solver = tmp_for_cpp_solver

        self._tmp_file_name_in = f"{path_to_cpp_solver}.wcnf"
        self._tmp_file_name_in = f"{self._tmp_file_name_in}"

        self._clean_tmp_folder = clean_tmp_folder

        self.solution: tp.Optional[tp.List[int]] = None

        self._logger = logging.getLogger("FlightDistributionModelCPP")

    def solve(self) -> None:
        self.solution = None

        os.makedirs(os.path.dirname(self._tmp_for_cpp_solver), exist_ok=True)
        path_in = os.path.join(self._tmp_for_cpp_solver, self._tmp_file_name_in)

        with open(path_in, "w") as f:
            f.write(self._wcnf)

    def pairwise_exactly_one(self, literals, var_counter, m):
        n = len(literals)
        clauses = []

        if n == 1:
            return [f"{m} {literals[0]} 0"], var_counter

        # AtLeastOne
        clauses.append(" ".join(map(str, [m] + literals)) + " 0")

        # AtMostOne — pairwise
        for i in range(n):
            for j in range(i + 1, n):
                clauses.append(f"{m} {-literals[i]} {-literals[j]} 0")

        return clauses, var_counter

    def sequential_exactly_one(self, literals, var_counter, m):
        n = len(literals)
        clauses = []

        if n == 1:
            return [f"{m} {literals[0]} 0"], var_counter

        clauses.append(" ".join(map(str, [m] + literals)) + " 0")

        # 2. Sequential encoding for at most one True
        s_vars = []
        for i in range(n - 1):
            s_vars.append(var_counter)
            var_counter += 1

        clauses.append(f"{m} -{literals[0]} {s_vars[0]} 0")

        for i in range(1, n - 1):
            clauses.append(f"{m} {-literals[i]} {s_vars[i]} 0")
            clauses.append(f"{m} {-s_vars[i - 1]} {s_vars[i]} 0")
            clauses.append(f"{m} {-literals[i]} {-s_vars[i - 1]} 0")
        clauses.append(f"{m} {-literals[-1]} {-s_vars[-1]} 0")

        return clauses, var_counter

    def create_model(self) -> None:
        sources_sinks = self.data.find_sources_and_sinks()

        m = max(self.data.node_cost.values()) + 1

        node_cpp_to_cnf = {j: i + 1 for i, j in enumerate(self.data.node_cost.keys())}
        edge_cpp_to_cnf = {
            e: len(node_cpp_to_cnf) + i + 1 for i, e in enumerate(self.data.edges)
        }

        clauses = []
        var_counter = (
            len(node_cpp_to_cnf) + len(edge_cpp_to_cnf) + 1
        )  # for auxiliary variables

        for nodes in self.data.alt_nodes:
            literals = [node_cpp_to_cnf[i] for i in nodes]
            ex_one, var_counter = self.pairwise_exactly_one(literals, var_counter, m)
            clauses.extend(ex_one)

        for node, cost in self.data.node_cost.items():
            clause = [cost, -node_cpp_to_cnf[node], 0]
            clauses.append(" ".join(map(str, clause)))

        in_edges_map = defaultdict(list)
        out_edges_map = defaultdict(list)

        for u, v in self.data.edges:
            edge_id = edge_cpp_to_cnf[(u, v)]
            out_edges_map[u].append(edge_id)
            in_edges_map[v].append(edge_id)

        for aircraft_id, nodes in self.data.aircraft_nodes.items():
            source, sink = sources_sinks[aircraft_id]
            for node in nodes:
                in_edges = in_edges_map[node]
                out_edges = out_edges_map[node]

                if node == source:
                    ex_one, var_counter = self.pairwise_exactly_one(
                        out_edges, var_counter, m
                    )
                    clauses.extend(ex_one)

                elif node == sink:
                    ex_one, var_counter = self.pairwise_exactly_one(
                        in_edges, var_counter, m
                    )
                    clauses.extend(ex_one)
                else:
                    ex_one, var_counter = self.pairwise_exactly_one(
                        out_edges + [-node_cpp_to_cnf[node]], var_counter, m
                    )
                    clauses.extend(ex_one)
                    ex_one, var_counter = self.pairwise_exactly_one(
                        in_edges + [-node_cpp_to_cnf[node]], var_counter, m
                    )
                    clauses.extend(ex_one)

        wcnf_list = [f"p wcnf {var_counter} {len(clauses)} {m}"] + clauses
        self._wcnf = "\n".join(wcnf_list)

    def create_problem(self) -> None:
        self.create_model()

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
