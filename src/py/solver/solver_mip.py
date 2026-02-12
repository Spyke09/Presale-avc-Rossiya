from src.py.problem_structures import FlightDistributionProblemMIP, NodeType
from src.py.settings import solver_param
from src.py.solver.base_solver import BaseFlightDistributionModelMIP


class FlightDistributionModelMIP(BaseFlightDistributionModelMIP):
    def __init__(
        self,
        data: FlightDistributionProblemMIP,
        timelimit: int = solver_param.timelimit,
        workers: int = solver_param.workers,
        log_verbosity: str = solver_param.log_verbosity,
    ):
        super().__init__(data, timelimit, workers, log_verbosity)
        self.data: FlightDistributionProblemMIP = data

    def _var_flow_bundle(self):
        self.flow_bundle = {}

        for aircraft in self.data.aircraft.tasks:
            aircraft_id = aircraft.aircraft_id
            graph = self.data.graph[aircraft_id]

            var_keys = [
                (node_1, node_2.node_id)
                for node_1, node_2_list in graph.next_nodes.items()
                for node_2 in node_2_list
            ]

            vars_dict = self.model.continuous_var_dict(var_keys, lb=0, ub=1, name=None)

            self.flow_bundle[aircraft_id] = vars_dict

    def _var_flight_assignment(self) -> None:
        self.flight_assignment = {
            aircraft.aircraft_id: {
                node.node_id: self.model.binary_var(
                    name=None,
                )
                for node in self.data.graph[aircraft.aircraft_id].nodes
                if (node.node_type == NodeType.FLIGHT)
                or (node.node_type == NodeType.TS)
            }
            for aircraft in self.data.aircraft.tasks
        }

    def add_variables(self) -> None:
        self._var_flow_bundle()
        self._var_flight_assignment()

    def _expr_flight_delay_obj_min(self):
        matching_assignments = [
            self.flight_assignment[aircraft.aircraft_id][node.node_id]
            for aircraft in self.data.aircraft.tasks
            for node in self.data.graph[aircraft.aircraft_id].nodes
            if node.node_type == NodeType.FLIGHT
            and node.entity.delayed
        ]

        total_matching = len(matching_assignments)
        if total_matching == 0:
            self.flight_similarity_obj_min = self.model.continuous_var(
                lb=0, ub=0, name="flight_similarity_obj_max_empty"
            )
        else:
            self.flight_delay_obj_min = self.model.sum(matching_assignments)

    def _expr_flight_similarity_obj_min(self):
        matching_assignments = [
            self.flight_assignment[aircraft.aircraft_id][node.node_id]
            for aircraft in self.data.aircraft.tasks
            for node in self.data.graph[aircraft.aircraft_id].nodes
            if node.node_type == NodeType.FLIGHT
            and node.entity.old_aircraft_id != node.entity.alt_aircraft_id
        ]

        total_matching = len(matching_assignments)
        if total_matching == 0:
            self.flight_similarity_obj_min = self.model.continuous_var(
                lb=0, ub=0, name="flight_similarity_obj_max_empty"
            )
        else:
            self.flight_similarity_obj_min = self.model.sum(matching_assignments)

    def _expr_technical_service_difference_obj_min(self):
        matching_assignments = [
            self.flight_assignment[aircraft.aircraft_id][node.node_id]
            * abs(node.entity.start - node.entity.old_start)
            for aircraft in self.data.aircraft.tasks
            for node in self.data.graph[aircraft.aircraft_id].nodes
            if node.node_type == NodeType.TS
        ]

        total_matching = len(matching_assignments)
        if total_matching == 0:
            self.technical_service_difference_obj_min = self.model.continuous_var(
                lb=0, ub=0, name="technical_service_difference_obj_min_empty"
            )
        else:
            self.technical_service_difference_obj_min = self.model.sum(
                matching_assignments
            )

    def add_expressions(self) -> None:
        self._expr_flight_similarity_obj_min()
        self._expr_technical_service_difference_obj_min()
        self._expr_flight_delay_obj_min()

    def add_obj_function(self) -> None:
        m = (
            self.data.parameters.end_with_delta_opt
            - self.data.parameters.start_with_delta_opt
        )
        obj_function_components = (
            self.flight_delay_obj_min * (m + 1) * (len(self.data.flight.tasks) + 1)
            + self.flight_similarity_obj_min * (m + 1)
            + self.technical_service_difference_obj_min
        )

        self.model.minimize(obj_function_components)

    def _constr_flow_conservation(self):
        self._mem_flow_in_out = {}

        for aircraft in self.data.aircraft.tasks:
            g = self.data.graph[aircraft.aircraft_id]
            constraints = []

            for node in g.nodes:
                in_nodes = g.prev_nodes[node.node_id]
                out_nodes = g.next_nodes[node.node_id]

                sum_in = self.model.sum(
                    self.flow_bundle[aircraft.aircraft_id][prev.node_id, node.node_id]
                    for prev in in_nodes
                )
                sum_out = self.model.sum(
                    self.flow_bundle[aircraft.aircraft_id][node.node_id, nxt.node_id]
                    for nxt in out_nodes
                )
                self._mem_flow_in_out[aircraft.aircraft_id, node.node_id] = (
                    sum_in,
                    sum_out,
                )

                if in_nodes and out_nodes:
                    constraints.append(sum_in == sum_out)
                elif not in_nodes:
                    constraints.append(sum_out == 1)
                elif not out_nodes:
                    constraints.append(sum_in == 1)

            self.model.add_constraints(constraints)

    def _constr_flight_use_one_aircraft(self):
        constraints = []
        for nodes in self.data.flight_to_nodes.values():
            vars_sum = self.model.sum(
                self.flight_assignment[node.aircraft_id][node.node_id] for node in nodes
            )
            constraints.append(vars_sum == 1)

        self.model.add_constraints(constraints)

    def _constr_technical_service_use_one(self):
        constraints = []
        for nodes in self.data.ts_to_nodes.values():
            vars_sum = self.model.sum(
                self.flight_assignment[node.aircraft_id][node.node_id] for node in nodes
            )
            constraints.append(vars_sum == 1)

        self.model.add_constraints(constraints)

    def _constr_flight_assignment(self):
        for aircraft in self.data.aircraft.tasks:
            g = self.data.graph[aircraft.aircraft_id]
            constraints = []

            for node in g.nodes:
                if node.node_type not in {NodeType.FLIGHT, NodeType.TS}:
                    continue

                in_nodes = g.prev_nodes[node.node_id]
                out_nodes = g.next_nodes[node.node_id]

                if out_nodes:
                    flow = self._mem_flow_in_out[aircraft.aircraft_id, node.node_id][1]
                elif in_nodes:
                    flow = self._mem_flow_in_out[aircraft.aircraft_id, node.node_id][0]
                else:
                    raise ValueError("Empty node")

                constraints.append(
                    self.flight_assignment[aircraft.aircraft_id][node.node_id] == flow
                )

            self.model.add_constraints(constraints)

    def add_constraints(self) -> None:
        self._constr_flow_conservation()
        self._constr_flight_assignment()
        self._constr_flight_use_one_aircraft()
        self._constr_technical_service_use_one()

    def add_warm_start(self):
        stp = self.model.new_solution()
        for aircraft in self.data.aircraft.tasks:
            for node in self.data.graph[aircraft.aircraft_id].nodes:
                if (node.node_type == NodeType.FLIGHT) or (
                    node.node_type == NodeType.TS
                ):
                    var = self.flight_assignment[aircraft.aircraft_id][node.node_id]
                    if node.node_type == NodeType.FLIGHT:
                        value = int(node.aircraft_id == node.entity.old_aircraft_id)
                        stp.add_var_value(var, value)
                    if node.node_type == NodeType.TS:
                        value = int(
                            (node.start == node.entity.old_start)
                            and (node.start_airport == node.entity.base_airport)
                        )
                        stp.add_var_value(var, value)

        self.model.add_mip_start(stp)
