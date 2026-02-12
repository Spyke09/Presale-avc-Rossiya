import pulp

from src.py.problem_structures import FlightDistributionProblemMIP, NodeType
from src.py.settings import solver_param
from src.py.solver.base_solver import BaseFlightDistributionModelPulp


class FlightDistributionModelPulp(BaseFlightDistributionModelPulp):
    def __init__(
        self,
        data: FlightDistributionProblemMIP,
        solver,
        workers: int = solver_param.workers,
        log_verbosity: str = solver_param.log_verbosity,
        timelimit: int = solver_param.timelimit,
    ):
        super().__init__(data, timelimit, workers, log_verbosity)
        self.data: FlightDistributionProblemMIP = data
        self.solver = solver

        self._timelimit = timelimit

    def _var_flow_bundle(self):
        self.flow_bundle = {}
        for aircraft in self.data.aircraft.tasks:
            aid = aircraft.aircraft_id
            g = self.data.graph[aid]

            self.flow_bundle[aid] = {
                (node1, node2.node_id): pulp.LpVariable(
                    f"flow_{aid}_{node1}_{node2.node_id}", 0, 1, pulp.LpContinuous
                )
                for node1, nexts in g.next_nodes.items()
                for node2 in nexts
            }

    def _var_flight_assignment(self) -> None:
        self.flight_assignment = {}
        for aircraft in self.data.aircraft.tasks:
            aid = aircraft.aircraft_id
            g = self.data.graph[aid]

            self.flight_assignment[aid] = {
                node.node_id: pulp.LpVariable(
                    f"x_{aid}_{node.node_id}", 0, 1, pulp.LpBinary
                )
                for node in g.nodes
                if node.node_type in {NodeType.FLIGHT, NodeType.TS}
            }

    def add_variables(self) -> None:
        self._var_flow_bundle()
        self._var_flight_assignment()

    def _expr_flight_similarity_obj_min(self):
        obj_similarity = []

        for aircraft in self.data.aircraft.tasks:
            aid = aircraft.aircraft_id
            g = self.data.graph[aid]

            for node in g.nodes:
                if (
                    node.node_type == NodeType.FLIGHT
                    and node.entity.old_aircraft_id != node.entity.alt_aircraft_id
                ):
                    obj_similarity.append(self.flight_assignment[aid][node.node_id])

        self.flight_similarity_obj_min = pulp.lpSum(obj_similarity)

    def _expr_technical_service_difference_obj_min(self):
        obj_ts_shift = []

        for aircraft in self.data.aircraft.tasks:
            aid = aircraft.aircraft_id
            g = self.data.graph[aid]

            for node in g.nodes:
                if node.node_type == NodeType.TS:
                    delay = abs(node.entity.start - node.entity.old_start)
                    obj_ts_shift.append(
                        self.flight_assignment[aid][node.node_id] * delay
                    )

        self.technical_service_difference_obj_min = pulp.lpSum(obj_ts_shift)

    def add_expressions(self) -> None:
        self._expr_flight_similarity_obj_min()
        self._expr_technical_service_difference_obj_min()

    def add_obj_function(self) -> None:
        m = (
            self.data.parameters.end_with_delta_opt
            - self.data.parameters.start_with_delta_opt
        )
        obj_function_components = (
            self.flight_similarity_obj_min * (m + 1)
            + self.technical_service_difference_obj_min
        )

        self.model += obj_function_components

    def _constr_flow_conservation(self):
        for aircraft in self.data.aircraft.tasks:
            aid = aircraft.aircraft_id
            g = self.data.graph[aid]

            for node in g.nodes:
                in_nodes = g.prev_nodes[node.node_id]
                out_nodes = g.next_nodes[node.node_id]

                inflow = pulp.lpSum(
                    self.flow_bundle[aid][(prev.node_id, node.node_id)]
                    for prev in in_nodes
                )
                outflow = pulp.lpSum(
                    self.flow_bundle[aid][(node.node_id, nxt.node_id)]
                    for nxt in out_nodes
                )

                if in_nodes and out_nodes:
                    self.model += inflow == outflow
                elif not in_nodes:
                    self.model += outflow == 1  # источник
                elif not out_nodes:
                    self.model += inflow == 1  # сток

    def _constr_unique_assignment_constraints(self):
        for nodes in self.data.flight_to_nodes.values():
            self.model += (
                pulp.lpSum(
                    self.flight_assignment[n.aircraft_id][n.node_id] for n in nodes
                )
                == 1
            )

        for nodes in self.data.ts_to_nodes.values():
            self.model += (
                pulp.lpSum(
                    self.flight_assignment[n.aircraft_id][n.node_id] for n in nodes
                )
                == 1
            )

    def _constr_flight_assignment(self):
        for aircraft in self.data.aircraft.tasks:
            aid = aircraft.aircraft_id
            g = self.data.graph[aid]

            for node in g.nodes:
                if node.node_type not in {NodeType.FLIGHT, NodeType.TS}:
                    continue

                in_nodes = g.prev_nodes[node.node_id]
                out_nodes = g.next_nodes[node.node_id]

                if out_nodes:
                    flow = pulp.lpSum(
                        self.flow_bundle[aid][(node.node_id, nxt.node_id)]
                        for nxt in out_nodes
                    )
                elif in_nodes:
                    flow = pulp.lpSum(
                        self.flow_bundle[aid][(prev.node_id, node.node_id)]
                        for prev in in_nodes
                    )
                else:
                    continue  # isolated node?

                self.model += self.flight_assignment[aid][node.node_id] == flow

    def add_constraints(self):
        self._constr_flow_conservation()
        self._constr_flight_assignment()
        self._constr_unique_assignment_constraints()

    def add_warm_start(self):
        for aircraft in self.data.aircraft.tasks:
            aid = aircraft.aircraft_id
            g = self.data.graph[aid]

            for node in g.nodes:
                if node.node_type not in {NodeType.FLIGHT, NodeType.TS}:
                    continue

                var = self.flight_assignment[aid][node.node_id]

                if node.node_type == NodeType.FLIGHT:
                    value = int(node.aircraft_id == node.entity.old_aircraft_id)
                else:  # TS
                    value = int(
                        (node.start == node.entity.old_start)
                        and (node.start_airport == node.entity.base_airport)
                    )

                var.setInitialValue(value)
