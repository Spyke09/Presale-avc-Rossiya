import typing as tp
from collections import defaultdict
import pandas as pd

from src.py.problem_structures import (
    Aircraft,
    AircraftTask,
    Airports,
    AlternativeFlight,
    AlternativeFlightTask,
    AlternativeTechnicalService,
    AlternativeTechnicalServiceTask,
    Flight,
    FlightDistributionProblemCP,
    FlightDistributionProblemMIP,
    FlightDistributionProblemCPP,
    FlightTask,
    FrozenChain,
    Graph,
    Node,
    NodeType,
    Parameters,
    TechnicalService,
    TechnicalServiceTask,
)
from src.py.settings import app_param
from src.py.solver import (
    FlightDistributionModelCP,
    FlightDistributionModelMIP,
    FlightDistributionModelCPP,
)
from src.py.solver.solver_pulp_mip import FlightDistributionModelPulp
from src.py.utils import opt_delta_time_to_date, opt_time_to_date


class DataOptManipulatorBase:
    def get_data(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        df_current_solution: pd.DataFrame,
        df_current_aircraft: pd.DataFrame,
        df_current_flight_equipments: pd.DataFrame,
        df_current_technical_service: pd.DataFrame,
        df_turnaround: pd.DataFrame,
        base_airports: tp.Tuple[str, ...],
        max_time_delay: int,
    ) -> tp.Union[FlightDistributionProblemCP, FlightDistributionProblemMIP]:
        raise NotImplementedError

    @staticmethod
    def _to_opt_time(time: pd.Timedelta) -> int:
        return round(
            time.total_seconds() * app_param.time_sec_to_min * app_param.time_conversion
        )

    @staticmethod
    def _get_aircraft(df_current_aircraft: pd.DataFrame) -> Aircraft:
        return Aircraft(
            [
                AircraftTask(
                    aircraft_id=row["aircraft_id"],
                    reserve_q=row["reserve_q"],
                )
                for i, row in df_current_aircraft.iterrows()
            ]
        )

    @staticmethod
    def _get_base_airports(
        df_current_solution: pd.DataFrame, base_airports: tp.Tuple[str, ...]
    ) -> Airports:
        airports = set(df_current_solution["departure_airport_code"].unique()) | set(
            df_current_solution["arrival_airport_code"].unique()
        )
        return Airports(airports=list(airports), base_airports=list(base_airports))

    def _get_parameters(
        self, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> Parameters:
        start_with_delta = start_time - app_param.delta_horizont
        end_with_delta = end_time + app_param.delta_horizont
        start_with_delta_opt = 0
        end_with_delta_opt = self._to_opt_time(end_with_delta - start_with_delta)
        start_horizon_opt = self._to_opt_time(start_time - start_with_delta)
        end_horizon_opt = self._to_opt_time(end_time - start_with_delta)
        return Parameters(
            start_horizon=start_time,
            end_horizon=end_time,
            start_horizon_opt=start_horizon_opt,
            end_horizon_opt=end_horizon_opt,
            start_with_delta=start_with_delta,
            end_with_delta=end_with_delta,
            start_with_delta_opt=start_with_delta_opt,
            end_with_delta_opt=end_with_delta_opt,
        )

    def _get_flight(
        self,
        df_current_solution: pd.DataFrame,
        parameters: Parameters,
    ) -> Flight:
        tasks = [
            FlightTask(
                opt_flight_task_id=row["previous_solution_id"],
                old_aircraft_id=row["aircraft_id"],
                flight_id=row["flight_id"],
                start=(
                    self._to_opt_time(
                        row["departure_time"] - parameters.start_with_delta
                    )
                ),
                end=(
                    self._to_opt_time(row["arrival_time"] - parameters.start_with_delta)
                ),
                is_fixed=row["is_fixed"],
                departure_airport_code=row["departure_airport_code"],
                arrival_airport_code=row["arrival_airport_code"],
            )
            for i, (_, row) in enumerate(df_current_solution.iterrows())
        ]

        opt_flight_task_id_for_turnaround = []
        for task in tasks:
            if (task.start >= parameters.start_with_delta_opt) and (
                task.end <= parameters.end_with_delta_opt
            ):
                opt_flight_task_id_for_turnaround.append(task.opt_flight_task_id)

        task_by_id = {task.opt_flight_task_id: task for task in tasks}

        return Flight(tasks, opt_flight_task_id_for_turnaround, task_by_id)

    def _get_technical_service(
        self,
        df_current_technical_service: pd.DataFrame,
        parameters: Parameters,
        aircraft: Aircraft,
    ) -> TechnicalService:
        aircraft_ids = set(i.aircraft_id for i in aircraft.tasks)
        tasks = [
            TechnicalServiceTask(
                technical_service_task_id=row["technical_service_id"],
                aircraft_id=row["aircraft_id"],
                old_start=self._to_opt_time(
                    row["time_start"] - parameters.start_with_delta
                ),
                old_end=self._to_opt_time(
                    row["time_finish"] - parameters.start_with_delta
                ),
                is_fixed=row["is_fixed"],
            )
            for _, row in df_current_technical_service.iterrows()
            if row["aircraft_id"] in aircraft_ids
        ]
        return TechnicalService(tasks)

    def _get_alt_flight(
        self,
        flight: Flight,
        df_current_flight_equipments: pd.DataFrame,
        df_turnaround: pd.DataFrame,
        max_time_delay: int
    ) -> AlternativeFlight:
        dict_turnaround = df_turnaround.set_index(["aircraft_id", "airport_code"])[
            "turnaround_time"
        ].to_dict()
        dict_equipments = df_current_flight_equipments.set_index(
            "previous_solution_id"
        )["equipment_ids"].to_dict()
        tasks = []
        idx = 0
        for task in flight.tasks:
            if task.is_fixed:
                equipments = [task.old_aircraft_id]
            else:
                equipments = dict_equipments[task.opt_flight_task_id]
            for equipment in equipments:
                for delay in range(max_time_delay + 1):
                    turnaround = dict_turnaround[(equipment, task.departure_airport_code)]
                    tasks.append(
                        AlternativeFlightTask(
                            opt_alt_flight_task_id=idx,
                            opt_flight_task_id=task.opt_flight_task_id,
                            old_aircraft_id=task.old_aircraft_id,
                            alt_aircraft_id=equipment,
                            start=task.start + delay,
                            end=task.end + delay,
                            turnaround_time=self._to_opt_time(turnaround),
                            departure_airport_code=task.departure_airport_code,
                            arrival_airport_code=task.arrival_airport_code,
                            is_fixed=task.is_fixed,
                            flight_id=task.flight_id,
                            delayed=(delay > 0)
                        )
                    )
                    idx += 1
                    if task.is_fixed:
                        break


        return AlternativeFlight(tasks)

    @classmethod
    def get_results(
        cls,
        problem: FlightDistributionModelCP,
    ) -> tp.Tuple[pd.DataFrame, ...]:
        raise NotImplementedError


class DataOptManipulatorCP(DataOptManipulatorBase):
    def get_data(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        df_current_solution: pd.DataFrame,
        df_current_aircraft: pd.DataFrame,
        df_current_flight_equipments: pd.DataFrame,
        df_current_technical_service: pd.DataFrame,
        df_turnaround: pd.DataFrame,
        base_airports: tp.Tuple[str, ...],
        max_time_delay: int,
    ) -> FlightDistributionProblemCP:

        parameters = self._get_parameters(start_time, end_time)

        flight = self._get_flight(df_current_solution, parameters)

        alt_flight = self._get_alt_flight(
            flight, df_current_flight_equipments, df_turnaround, max_time_delay
        )

        aircraft = self._get_aircraft(df_current_aircraft)

        technical_service = self._get_technical_service(
            df_current_technical_service, parameters, aircraft
        )

        airports = self._get_base_airports(df_current_solution, base_airports)
        return FlightDistributionProblemCP(
            aircraft=aircraft,
            flight=flight,
            alt_flight=alt_flight,
            technical_service=technical_service,
            parameters=parameters,
            airports=airports,
        )

    @classmethod
    def get_results(
        cls,
        problem: FlightDistributionModelCP,
    ) -> tp.Tuple[pd.DataFrame, ...]:
        if not problem or not problem.solution or not problem.solution.is_solution():
            df_new_solution = pd.DataFrame(
                columns=[
                    "departure_time",
                    "arrival_time",
                    "flight_id",
                    "departure_airport_code",
                    "arrival_airport_code",
                    "aircraft_id",
                    "previous_solution_id",
                ]
            )
            df_new_technical_service = pd.DataFrame(
                columns=[
                    "technical_service_id",
                    "aircraft_id",
                    "time_size",
                    "time_start",
                    "time_finish",
                ]
            )
            return df_new_solution, df_new_technical_service

        start_time = problem.data.parameters.start_with_delta
        dict_new_solution = defaultdict(list)

        aircraft_ids_dict = defaultdict(list)
        for task_af in problem.data.alt_flight.tasks:
            task_var = problem.alt_flight_tasks[task_af.opt_alt_flight_task_id]
            if problem.solution.get_var_solution(task_var).is_present():
                aircraft_ids_dict[task_af.opt_flight_task_id].append(
                    task_af.alt_aircraft_id
                )

        for task_f in problem.data.flight.tasks:
            task_f_itv = problem.solution.get_var_solution(
                problem.flight_tasks[task_f.opt_flight_task_id]
            )
            if not task_f_itv.is_present():
                continue

            aircraft_ids = aircraft_ids_dict[task_f.opt_flight_task_id]

            assert len(aircraft_ids) == 1
            dict_new_solution["departure_time"].append(
                opt_time_to_date(start_time, task_f_itv.get_start())
            )
            dict_new_solution["arrival_time"].append(
                opt_time_to_date(start_time, task_f_itv.get_end())
            )
            dict_new_solution["flight_id"].append(task_f.flight_id)
            dict_new_solution["departure_airport_code"].append(
                task_f.departure_airport_code
            )
            dict_new_solution["arrival_airport_code"].append(
                task_f.arrival_airport_code
            )
            dict_new_solution["aircraft_id"].append(aircraft_ids[0])
            dict_new_solution["previous_solution_id"].append(task_f.opt_flight_task_id)

        dict_new_technical_service = defaultdict(list)
        for task_tc in problem.data.technical_service.tasks:
            task_tc_itv = problem.solution.get_var_solution(
                problem.technical_service_tasks[task_tc.technical_service_task_id]
            )
            if not task_tc_itv.is_present():
                continue

            dict_new_technical_service["technical_service_id"].append(
                task_tc.technical_service_task_id
            )
            dict_new_technical_service["aircraft_id"].append(task_tc.aircraft_id)
            dict_new_technical_service["time_start"].append(
                opt_time_to_date(start_time, task_tc_itv.get_start())
            )
            dict_new_technical_service["time_finish"].append(
                opt_time_to_date(start_time, task_tc_itv.get_end())
            )
            dict_new_technical_service["time_size"].append(
                dict_new_technical_service["time_finish"][-1]
                - dict_new_technical_service["time_start"][-1]
            )

        df_new_solution = pd.DataFrame(dict_new_solution)
        df_new_technical_service = pd.DataFrame(dict_new_technical_service)
        return df_new_solution, df_new_technical_service


class DataOptManipulatorMIP(DataOptManipulatorBase):
    def get_data(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        df_current_solution: pd.DataFrame,
        df_current_aircraft: pd.DataFrame,
        df_current_flight_equipments: pd.DataFrame,
        df_current_technical_service: pd.DataFrame,
        df_turnaround: pd.DataFrame,
        base_airports: tp.Tuple[str, ...],
        max_time_delay: int = 0,
    ) -> FlightDistributionProblemMIP:

        parameters = self._get_parameters(start_time, end_time)

        flight = self._get_flight(df_current_solution, parameters)
        alt_flight = self._get_alt_flight(
            flight, df_current_flight_equipments, df_turnaround, max_time_delay
        )

        aircraft = self._get_aircraft(df_current_aircraft)

        technical_service = self._get_technical_service(
            df_current_technical_service, parameters, aircraft
        )

        frozen_chain = self._get_frozen_chain(
            flight, technical_service, aircraft, parameters
        )

        alt_technical_service = self._get_alternative_technical_service(
            technical_service, alt_flight, flight, parameters, frozen_chain
        )

        graph = self._get_graph(
            alt_flight, alt_technical_service, aircraft, frozen_chain
        )

        ts_to_nodes = self._get_ts_to_nodes(aircraft, graph, technical_service)
        flight_to_nodes = self._get_flight_to_nodes(aircraft, graph, flight)
        return FlightDistributionProblemMIP(
            aircraft=aircraft,
            graph=graph,
            ts_to_nodes=ts_to_nodes,
            flight_to_nodes=flight_to_nodes,
            parameters=parameters,
            technical_service=technical_service,
            flight=flight,
        )

    @staticmethod
    def _get_flight_to_nodes(
        aircraft: Aircraft, graph: tp.Dict[int, Graph], flight: Flight
    ) -> tp.Dict[int, tp.List[Node]]:
        flight_to_nodes = {i.opt_flight_task_id: [] for i in flight.tasks}

        for aircraft in aircraft.tasks:
            for node in graph[aircraft.aircraft_id].nodes:
                if node.node_type == NodeType.FLIGHT:
                    flight_to_nodes[node.entity.opt_flight_task_id].append(node)

        return flight_to_nodes

    @staticmethod
    def _get_ts_to_nodes(
        aircraft: Aircraft,
        graph: tp.Dict[int, Graph],
        technical_service: TechnicalService,
    ) -> tp.Dict[int, tp.List[Node]]:
        ts_to_nodes = {i.technical_service_task_id: [] for i in technical_service.tasks}

        for aircraft in aircraft.tasks:
            for node in graph[aircraft.aircraft_id].nodes:
                if node.node_type == NodeType.TS:
                    ts_to_nodes[node.entity.technical_service_task_id].append(node)

        return ts_to_nodes

    @staticmethod
    def _get_frozen_chain(
        flight: Flight,
        technical_service: TechnicalService,
        aircraft: Aircraft,
        parameters: Parameters,
    ) -> tp.List[FrozenChain]:
        result = []
        first_flight_not_frozen: tp.Dict[int, tp.Optional[FlightTask]] = {
            i.aircraft_id: None for i in aircraft.tasks
        }
        for f in flight.tasks:
            if (not f.is_fixed) and (
                (first_flight_not_frozen[f.old_aircraft_id] is None)
                or (first_flight_not_frozen[f.old_aircraft_id].start > f.start)
            ):
                first_flight_not_frozen[f.old_aircraft_id] = f

        last_flight_not_frozen: tp.Dict[int, tp.Optional[FlightTask]] = {
            i.aircraft_id: None for i in aircraft.tasks
        }
        for f in flight.tasks:
            if (not f.is_fixed) and (
                (last_flight_not_frozen[f.old_aircraft_id] is None)
                or (last_flight_not_frozen[f.old_aircraft_id].start < f.start)
            ):
                last_flight_not_frozen[f.old_aircraft_id] = f

        frozen_flight = [
            Node(
                node_id=i,
                node_type=NodeType.FLIGHT,
                start=t.start,
                end=t.end,
                start_airport=t.departure_airport_code,
                end_airport=t.arrival_airport_code,
                aircraft_id=t.old_aircraft_id,
                entity=t,
            )
            for i, t in enumerate(flight.tasks)
            if t.is_fixed
        ]
        frozen_ts = [
            Node(
                node_id=i + len(frozen_flight),
                node_type=NodeType.TS,
                start=t.old_start,
                end=t.old_end,
                start_airport="",
                end_airport="",
                aircraft_id=t.aircraft_id,
                entity=t,
            )
            for i, t in enumerate(technical_service.tasks)
            if t.is_fixed
        ]
        frozen_nodes = frozen_flight + frozen_ts
        dict_frozen: tp.Dict[int, tp.List[Node]] = defaultdict(list)
        for task in frozen_nodes:
            dict_frozen[task.aircraft_id].append(task)

        for cur_aircraft in aircraft.tasks:
            aircraft_id = cur_aircraft.aircraft_id
            frozen_all = dict_frozen[aircraft_id]
            sorted_frozen = sorted(frozen_all, key=(lambda x: x.start))
            assert all(
                sorted_frozen[i].end <= sorted_frozen[i + 1].start
                for i in range(len(sorted_frozen) - 1)
            )

            sorted_left = [
                i for i in sorted_frozen if i.start < parameters.start_horizon_opt
            ]
            sorted_right = [
                i for i in sorted_frozen if i.start > parameters.start_horizon_opt
            ]
            airport_for_ts = {}
            stack = []

            for j, sorted_tmp in enumerate([sorted_left, sorted_right]):
                next_airport = None
                for node in sorted_tmp:
                    if node.node_type == NodeType.FLIGHT:
                        next_airport = node.end_airport
                        for prev_node in stack:
                            airport_for_ts[prev_node] = node.start_airport
                        stack.clear()
                    else:
                        stack.append(node)
                if len(stack) > 0:
                    if (j == 0) and (next_airport is None):
                        if first_flight_not_frozen[aircraft_id] is None:
                            next_airport = app_param.base_airports[0]
                        else:
                            next_airport = first_flight_not_frozen[
                                aircraft_id
                            ].departure_airport_code
                    if (j == 1) and (next_airport is None):
                        if last_flight_not_frozen[aircraft_id] is None:
                            next_airport = app_param.base_airports[0]
                        else:
                            next_airport = last_flight_not_frozen[
                                aircraft_id
                            ].arrival_airport_code
                    for prev_node in stack:
                        airport_for_ts[prev_node] = next_airport
                    stack.clear()

            sorted_left = [
                (
                    Node(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        start=node.start,
                        end=node.end,
                        start_airport=airport_for_ts[node],
                        end_airport=airport_for_ts[node],
                        aircraft_id=node.aircraft_id,
                        entity=node.entity,
                    )
                    if (node.node_type == NodeType.TS)
                    else node
                )
                for i, node in enumerate(sorted_left)
            ]

            sorted_right = [
                (
                    Node(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        start=node.start,
                        end=node.end,
                        start_airport=airport_for_ts[node],
                        end_airport=airport_for_ts[node],
                        aircraft_id=node.aircraft_id,
                        entity=node.entity,
                    )
                    if (node.node_type == NodeType.TS)
                    else node
                )
                for i, node in enumerate(sorted_right)
            ]

            assert len(sorted_left) + len(sorted_right) == len(sorted_frozen)

            result.append(FrozenChain(aircraft_id, sorted_left, sorted_right))

        return result

    def _get_alternative_technical_service(
        self,
        technical_service: TechnicalService,
        alt_flight: AlternativeFlight,
        flight: Flight,
        parameters: Parameters,
        frozen_chain: tp.List[FrozenChain],
    ) -> AlternativeTechnicalService:
        alt_tasks = []
        task_ids = 0

        alt_flight_to_flight = {}
        _flight_id_to_flight = {i.opt_flight_task_id: i for i in flight.tasks}
        for cur_alt_flight in alt_flight.tasks:
            alt_flight_to_flight[cur_alt_flight] = _flight_id_to_flight[
                cur_alt_flight.opt_flight_task_id
            ]

        aircraft_to_chain_left = {i.aircraft_id: i.chain_left for i in frozen_chain}
        aircraft_to_chain_right = {i.aircraft_id: i.chain_right for i in frozen_chain}
        entity_to_chain_node = {}
        for aircraft_to_chain in [aircraft_to_chain_left, aircraft_to_chain_right]:
            for _, chain in aircraft_to_chain.items():
                for node in chain:
                    entity_to_chain_node[node.entity] = node

        next_node = {}
        for f_chain in frozen_chain:
            for chain in [f_chain.chain_left, f_chain.chain_right]:
                for i in range(len(chain) - 1):
                    next_node[chain[i]] = chain[i + 1]

        for technical_service_task in technical_service.tasks:
            if technical_service_task.is_fixed:
                if technical_service_task not in entity_to_chain_node:
                    for base_airport in app_param.base_airports:
                        task = AlternativeTechnicalServiceTask(
                            alt_technical_service_task_id=task_ids,
                            technical_service_task_id=technical_service_task.technical_service_task_id,
                            aircraft_id=technical_service_task.aircraft_id,
                            start=technical_service_task.old_start,
                            end=technical_service_task.old_end,
                            old_start=technical_service_task.old_start,
                            old_end=technical_service_task.old_end,
                            base_airport=base_airport,
                            is_fixed=technical_service_task.is_fixed,
                        )
                        alt_tasks.append(task)
                        task_ids += 1
                else:
                    base_airport = entity_to_chain_node[
                        technical_service_task
                    ].start_airport
                    task = AlternativeTechnicalServiceTask(
                        alt_technical_service_task_id=task_ids,
                        technical_service_task_id=technical_service_task.technical_service_task_id,
                        aircraft_id=technical_service_task.aircraft_id,
                        start=technical_service_task.old_start,
                        end=technical_service_task.old_end,
                        old_start=technical_service_task.old_start,
                        old_end=technical_service_task.old_end,
                        base_airport=base_airport,
                        is_fixed=technical_service_task.is_fixed,
                    )
                    alt_tasks.append(task)
                    task_ids += 1
                continue

            for airport in app_param.base_airports:
                task = AlternativeTechnicalServiceTask(
                    alt_technical_service_task_id=task_ids,
                    technical_service_task_id=technical_service_task.technical_service_task_id,
                    aircraft_id=technical_service_task.aircraft_id,
                    start=technical_service_task.old_start,
                    end=technical_service_task.old_end,
                    old_start=technical_service_task.old_start,
                    old_end=technical_service_task.old_end,
                    base_airport=airport,
                    is_fixed=technical_service_task.is_fixed,
                )
                alt_tasks.append(task)
                task_ids += 1
                # size = technical_service_task.old_end - technical_service_task.old_start
                # task = AlternativeTechnicalServiceTask(
                #     alt_technical_service_task_id=task_ids,
                #     technical_service_task_id=technical_service_task.technical_service_task_id,
                #     aircraft_id=technical_service_task.aircraft_id,
                #     start=parameters.start_horizon_opt,
                #     end=parameters.start_horizon_opt + size,
                #     old_start=technical_service_task.old_start,
                #     old_end=technical_service_task.old_end,
                #     base_airport=airport,
                #     is_fixed=technical_service_task.is_fixed,
                # )
                # alt_tasks.append(task)
                # task_ids += 1

            for alt_flight_task in alt_flight.tasks:
                if (
                    technical_service_task.aircraft_id
                    != alt_flight_task.alt_aircraft_id
                ):
                    continue
                flight_task = alt_flight_to_flight[alt_flight_task]
                if flight_task in entity_to_chain_node:
                    node = entity_to_chain_node[flight_task]
                    end_ts = node.end + (
                        technical_service_task.old_end
                        - technical_service_task.old_start
                    )
                    if (node in next_node) and (end_ts <= next_node[node].start):
                        task = AlternativeTechnicalServiceTask(
                            alt_technical_service_task_id=task_ids,
                            technical_service_task_id=technical_service_task.technical_service_task_id,
                            aircraft_id=technical_service_task.aircraft_id,
                            start=node.end,
                            end=end_ts,
                            old_start=technical_service_task.old_start,
                            old_end=technical_service_task.old_end,
                            base_airport=alt_flight_task.arrival_airport_code,
                            is_fixed=technical_service_task.is_fixed,
                        )
                        alt_tasks.append(task)
                        task_ids += 1
                        continue
                    else:
                        continue
                else:
                    if alt_flight_task.end < parameters.start_horizon_opt:
                        continue
                    end_ts = alt_flight_task.end + (
                        technical_service_task.old_end
                        - technical_service_task.old_start
                    )
                    task = AlternativeTechnicalServiceTask(
                        alt_technical_service_task_id=task_ids,
                        technical_service_task_id=technical_service_task.technical_service_task_id,
                        aircraft_id=technical_service_task.aircraft_id,
                        start=alt_flight_task.end,
                        end=end_ts,
                        old_start=technical_service_task.old_start,
                        old_end=technical_service_task.old_end,
                        base_airport=alt_flight_task.arrival_airport_code,
                        is_fixed=technical_service_task.is_fixed,
                    )
                    alt_tasks.append(task)
                    task_ids += 1

        return AlternativeTechnicalService(alt_tasks)

    def _get_graph(
        self,
        alt_flight: AlternativeFlight,
        alt_technical_service: AlternativeTechnicalService,
        aircraft: Aircraft,
        frozen_chain: tp.List[FrozenChain],
    ) -> tp.Dict[int, Graph]:
        next_node_by_chain_ts = {}
        next_node_by_chain_flight = {}
        for fc in frozen_chain:
            for i, node in enumerate(fc.chain_left[:-1]):
                if node.node_type == NodeType.FLIGHT:
                    next_node_by_chain_flight[
                        fc.chain_left[i].entity.opt_flight_task_id
                    ] = fc.chain_left[i + 1]
                if node.node_type == NodeType.TS:
                    next_node_by_chain_ts[
                        fc.chain_left[i].entity.technical_service_task_id
                    ] = fc.chain_left[i + 1]

        nodes_flight = [
            Node(
                node_id=i,
                node_type=NodeType.FLIGHT,
                start=alt_flight_task.start,
                end=alt_flight_task.end,
                start_airport=alt_flight_task.departure_airport_code,
                end_airport=alt_flight_task.arrival_airport_code,
                aircraft_id=alt_flight_task.alt_aircraft_id,
                entity=alt_flight_task,
            )
            for i, alt_flight_task in enumerate(alt_flight.tasks)
        ]

        nodes_ts = [
            Node(
                node_id=i + len(nodes_flight),
                node_type=NodeType.TS,
                start=alt_ts_task.start,
                end=alt_ts_task.end,
                start_airport=alt_ts_task.base_airport,
                end_airport=alt_ts_task.base_airport,
                aircraft_id=alt_ts_task.aircraft_id,
                entity=alt_ts_task,
            )
            for i, alt_ts_task in enumerate(alt_technical_service.tasks)
        ]

        flight_to_node = {}
        for node in nodes_flight:
            if node.entity.is_fixed:
                if node.entity.opt_flight_task_id in flight_to_node:
                    raise ValueError
                flight_to_node[node.entity.opt_flight_task_id] = node

        ts_to_node = {}
        for node in nodes_ts:
            if node.entity.is_fixed:
                if node.entity.technical_service_task_id in ts_to_node:
                    raise ValueError
                ts_to_node[node.entity.technical_service_task_id] = node

        nodes = nodes_flight + nodes_ts
        edges = []

        nodes_by_start_airport_and_aircraft = defaultdict(list)
        for node in nodes:
            nodes_by_start_airport_and_aircraft[
                (node.start_airport, node.aircraft_id)
            ].append(node)

        for node_1 in nodes:
            if (node_1.node_type == NodeType.FLIGHT) and (
                node_1.entity.opt_flight_task_id in next_node_by_chain_flight
            ):
                old_node = next_node_by_chain_flight[node_1.entity.opt_flight_task_id]
                if old_node.node_type == NodeType.FLIGHT:
                    edges.append(
                        (node_1, flight_to_node[old_node.entity.opt_flight_task_id])
                    )
                elif old_node.node_type == NodeType.TS:
                    edges.append(
                        (node_1, ts_to_node[old_node.entity.technical_service_task_id])
                    )
                continue
            key = (node_1.end_airport, node_1.aircraft_id)
            for node_2 in nodes_by_start_airport_and_aircraft.get(key, []):
                if node_1.node_id == node_2.node_id:
                    continue
                if (node_1.node_type == NodeType.FLIGHT) and (
                    node_2.node_type == NodeType.FLIGHT
                ):
                    turnaround = node_1.entity.turnaround_time
                    if node_1.end + turnaround <= node_2.start:
                        edges.append((node_1, node_2))
                    else:
                        continue
                else:
                    if node_1.end <= node_2.start:
                        edges.append((node_1, node_2))
                    else:
                        continue

        nodes_ids = len(nodes_flight) + len(nodes_ts)
        aircraft_to_source = {
            t.aircraft_id: Node(
                node_id=nodes_ids + i,
                node_type=NodeType.SOURCE,
                start=0,
                end=0,
                start_airport="",
                end_airport="",
                aircraft_id=t.aircraft_id,
                entity=None,
            )
            for i, t in enumerate(aircraft.tasks)
        }
        nodes_ids += len(aircraft.tasks)
        aircraft_to_sink = {
            t.aircraft_id: Node(
                node_id=nodes_ids + i,
                node_type=NodeType.SINK,
                start=0,
                end=0,
                start_airport="",
                end_airport="",
                aircraft_id=t.aircraft_id,
                entity=None,
            )
            for i, t in enumerate(aircraft.tasks)
        }

        result = {}
        nodes_by_aircraft_id = defaultdict(list)
        for node in nodes:
            nodes_by_aircraft_id[node.aircraft_id].append(node)

        nodes_by_nodes_id = {i.node_id: i for i in nodes}
        for i in aircraft_to_source.values():
            nodes_by_nodes_id[i.node_id] = i
        for i in aircraft_to_sink.values():
            nodes_by_nodes_id[i.node_id] = i

        for ai in aircraft.tasks:
            cur_nodes = nodes_by_aircraft_id[ai.aircraft_id]
            cur_source, cur_sink = (
                aircraft_to_source[ai.aircraft_id],
                aircraft_to_sink[ai.aircraft_id],
            )
            prev_nodes_dict = {i.node_id: [] for i in cur_nodes}
            for node_1, node_2 in edges:
                if node_2.node_id in prev_nodes_dict:
                    prev_nodes_dict[node_2.node_id].append(node_1)

            for node in cur_nodes:
                if (len(prev_nodes_dict[node.node_id]) == 0) or (
                    (node.entity is not None) and (not node.entity.is_fixed)
                ):
                    prev_nodes_dict[node.node_id].append(cur_source)
            prev_nodes_dict[cur_source.node_id] = []
            cur_nodes.append(cur_source)

            next_nodes_dict = {i.node_id: [] for i in cur_nodes}
            for node_2, prev_nodes in prev_nodes_dict.items():
                for node_1 in prev_nodes:
                    next_nodes_dict[node_1.node_id].append(nodes_by_nodes_id[node_2])

            for node in cur_nodes:
                if (len(next_nodes_dict[node.node_id]) == 0) or (
                    (node.entity is not None) and (not node.entity.is_fixed)
                ):
                    next_nodes_dict[node.node_id].append(cur_sink)
            next_nodes_dict[cur_sink.node_id] = []
            cur_nodes.append(cur_sink)

            prev_nodes_dict = {i.node_id: [] for i in cur_nodes}
            for node_1, next_nodes in next_nodes_dict.items():
                for node_2 in next_nodes:
                    prev_nodes_dict[node_2.node_id].append(nodes_by_nodes_id[node_1])

            result[ai.aircraft_id] = Graph(cur_nodes, next_nodes_dict, prev_nodes_dict)

        return result

    @classmethod
    def get_results(
        cls,
        problem: FlightDistributionModelMIP,
    ) -> tp.Tuple[pd.DataFrame, ...]:
        if problem.solution is None:
            df_new_solution = pd.DataFrame(
                columns=[
                    "departure_time",
                    "arrival_time",
                    "flight_id",
                    "departure_airport_code",
                    "arrival_airport_code",
                    "aircraft_id",
                    "previous_solution_id",
                ]
            )
            df_new_technical_service = pd.DataFrame(
                columns=[
                    "technical_service_id",
                    "aircraft_id",
                    "time_size",
                    "time_start",
                    "time_finish",
                ]
            )
            return df_new_solution, df_new_technical_service

        start_time = problem.data.parameters.start_with_delta
        dict_new_solution = defaultdict(list)

        flight_to_aircraft = dict()
        ts_to_time = dict()
        for aircraft in problem.data.aircraft.tasks:
            for node in problem.data.graph[aircraft.aircraft_id].nodes:
                if (node.node_type == NodeType.FLIGHT) or (
                    node.node_type == NodeType.TS
                ):
                    val = round(
                        problem.flight_assignment[aircraft.aircraft_id][
                            node.node_id
                        ].solution_value,
                        5,
                    )
                    assert (val == 1) or (val == 0)
                    if (node.node_type == NodeType.FLIGHT) and (val == 1):
                        flight_to_aircraft[node.entity.opt_flight_task_id] = (
                            aircraft.aircraft_id
                        )

                    if (node.node_type == NodeType.TS) and (val == 1):
                        ts_to_time[node.entity.technical_service_task_id] = node.start

        for task_f in problem.data.flight.tasks:
            dict_new_solution["departure_time"].append(
                opt_time_to_date(start_time, task_f.start)
            )
            dict_new_solution["arrival_time"].append(
                opt_time_to_date(start_time, task_f.end)
            )
            dict_new_solution["flight_id"].append(task_f.flight_id)
            dict_new_solution["departure_airport_code"].append(
                task_f.departure_airport_code
            )
            dict_new_solution["arrival_airport_code"].append(
                task_f.arrival_airport_code
            )
            dict_new_solution["aircraft_id"].append(
                flight_to_aircraft.get(task_f.opt_flight_task_id, 0)
            )
            dict_new_solution["previous_solution_id"].append(task_f.opt_flight_task_id)

        dict_new_technical_service = defaultdict(list)
        for task_tc in problem.data.technical_service.tasks:
            if task_tc.technical_service_task_id not in ts_to_time:
                continue
            tc_start_time = ts_to_time[task_tc.technical_service_task_id]
            time_size = task_tc.old_end - task_tc.old_start
            dict_new_technical_service["technical_service_id"].append(
                task_tc.technical_service_task_id
            )
            dict_new_technical_service["aircraft_id"].append(task_tc.aircraft_id)
            dict_new_technical_service["time_size"].append(
                opt_delta_time_to_date(time_size)
            )
            dict_new_technical_service["time_start"].append(
                opt_time_to_date(start_time, tc_start_time)
            )
            dict_new_technical_service["time_finish"].append(
                opt_time_to_date(start_time, tc_start_time + time_size)
            )

        df_new_solution = pd.DataFrame(dict_new_solution)
        df_new_technical_service = pd.DataFrame(dict_new_technical_service)
        return df_new_solution, df_new_technical_service


class DataOptManipulatorCPP:
    @staticmethod
    def get_data(
        mip_problem: FlightDistributionProblemMIP,
    ) -> FlightDistributionProblemCPP:
        node_from_id = {
            node.node_id: node
            for aircraft_id, graph in mip_problem.graph.items()
            for node in graph.nodes
        }

        node_cpp_id_to_opt_id = {i: j for i, j in enumerate(node_from_id.keys())}
        node_opt_id_to_cpp_id = {i: j for j, i in node_cpp_id_to_opt_id.items()}

        aircraft_cpp_id_to_opt_id = {
            i: j.aircraft_id for i, j in enumerate(mip_problem.aircraft.tasks)
        }
        aircraft_opt_id_to_cpp_id = {i: j for j, i in aircraft_cpp_id_to_opt_id.items()}

        aircraft_nodes = {
            aircraft_id: [] for aircraft_id in aircraft_cpp_id_to_opt_id.keys()
        }
        for i, j in node_cpp_id_to_opt_id.items():
            aircraft_id = aircraft_opt_id_to_cpp_id[node_from_id[j].aircraft_id]
            aircraft_nodes[aircraft_id].append(i)

        alt_nodes = (
            [
                [node_opt_id_to_cpp_id[node.node_id] for node in nodes]
                for nodes in mip_problem.ts_to_nodes.values()
            ]
            + [
                [node_opt_id_to_cpp_id[node.node_id] for node in nodes]
                for nodes in mip_problem.flight_to_nodes.values()
            ]
            + [
                [node_opt_id_to_cpp_id[node_id]]
                for node_id, node in node_from_id.items()
                if (node.node_type == NodeType.SINK)
                or (node.node_type == NodeType.SOURCE)
            ]
        )

        edges = []
        for aircraft in mip_problem.aircraft.tasks:
            g = mip_problem.graph[aircraft.aircraft_id]
            for node in g.nodes:
                for next_node in g.next_nodes[node.node_id]:
                    edges.append(
                        (
                            node_opt_id_to_cpp_id[node.node_id],
                            node_opt_id_to_cpp_id[next_node.node_id],
                        )
                    )

        m = (
            mip_problem.parameters.end_with_delta_opt
            - mip_problem.parameters.start_with_delta_opt
        ) + 1
        mm = (len(mip_problem.flight.tasks) + 1) * m
        node_cost = {i: 0 for i in node_cpp_id_to_opt_id.keys()}
        start_point = []
        for cpp_id, opt_id in node_cpp_id_to_opt_id.items():
            node = node_from_id[opt_id]
            if node.node_type == NodeType.FLIGHT:
                cost = int(
                    node.entity.old_aircraft_id != node.entity.alt_aircraft_id
                ) * m + int(node.entity.delayed) * mm
                node_cost[cpp_id] = cost
                if cost == 0:
                    start_point.append(cpp_id)
            if node.node_type == NodeType.TS:
                cost = abs(node.entity.old_start - node.entity.start)
                node_cost[cpp_id] = cost
                if (cost == 0) and (node.start_airport == node.entity.base_airport):
                    start_point.append(cpp_id)
            if (node.node_type == NodeType.SOURCE) or (node.node_type == NodeType.SINK):
                start_point.append(cpp_id)

        problem = FlightDistributionProblemCPP(
            aircraft_nodes=aircraft_nodes,
            alt_nodes=alt_nodes,
            node_cost=node_cost,
            edges=edges,
            aircraft_cpp_id_to_opt_id=aircraft_cpp_id_to_opt_id,
            node_cpp_id_to_opt_id=node_cpp_id_to_opt_id,
            start_point=start_point,
        )

        return problem

    @classmethod
    def get_results(
        cls,
        problem: FlightDistributionModelCPP,
        data_mip: FlightDistributionProblemMIP,
    ) -> tp.Tuple[pd.DataFrame, ...]:
        if problem.solution is None:
            df_new_solution = pd.DataFrame(
                columns=[
                    "departure_time",
                    "arrival_time",
                    "flight_id",
                    "departure_airport_code",
                    "arrival_airport_code",
                    "aircraft_id",
                    "previous_solution_id",
                ]
            )
            df_new_technical_service = pd.DataFrame(
                columns=[
                    "technical_service_id",
                    "aircraft_id",
                    "time_size",
                    "time_start",
                    "time_finish",
                ]
            )
            return df_new_solution, df_new_technical_service

        solution = problem.solution
        node_from_id = {
            node.node_id: node
            for aircraft_id, graph in data_mip.graph.items()
            for node in graph.nodes
        }

        solution_node = [
            node_from_id[problem.data.node_cpp_id_to_opt_id[i]] for i in solution
        ]

        flight_to_aircraft = {
            i.entity.opt_flight_task_id: i.aircraft_id
            for i in solution_node
            if i.node_type == NodeType.FLIGHT
        }

        flight_to_time = {
            i.entity.opt_flight_task_id: (i.start, i.end)
            for i in solution_node
            if i.node_type == NodeType.FLIGHT
        }

        ts_to_time = {
            i.entity.technical_service_task_id: i.entity.start
            for i in solution_node
            if i.node_type == NodeType.TS
        }
        start_time = data_mip.parameters.start_with_delta
        dict_new_solution = defaultdict(list)

        for task_f in data_mip.flight.tasks:
            dict_new_solution["departure_time"].append(
                opt_time_to_date(start_time, flight_to_time.get(task_f.opt_flight_task_id, (task_f.start, task_f.end))[0])
            )
            dict_new_solution["arrival_time"].append(
                opt_time_to_date(start_time, flight_to_time.get(task_f.opt_flight_task_id, (task_f.start, task_f.end))[1])
            )
            dict_new_solution["flight_id"].append(task_f.flight_id)
            dict_new_solution["departure_airport_code"].append(
                task_f.departure_airport_code
            )
            dict_new_solution["arrival_airport_code"].append(
                task_f.arrival_airport_code
            )
            dict_new_solution["aircraft_id"].append(
                flight_to_aircraft.get(task_f.opt_flight_task_id, 0)
            )
            dict_new_solution["previous_solution_id"].append(task_f.opt_flight_task_id)

        dict_new_technical_service = defaultdict(list)
        for task_tc in data_mip.technical_service.tasks:
            if task_tc.technical_service_task_id not in ts_to_time:
                continue
            tc_start_time = ts_to_time[task_tc.technical_service_task_id]
            time_size = task_tc.old_end - task_tc.old_start
            dict_new_technical_service["technical_service_id"].append(
                task_tc.technical_service_task_id
            )
            dict_new_technical_service["aircraft_id"].append(task_tc.aircraft_id)
            dict_new_technical_service["time_size"].append(
                opt_delta_time_to_date(time_size)
            )
            dict_new_technical_service["time_start"].append(
                opt_time_to_date(start_time, tc_start_time)
            )
            dict_new_technical_service["time_finish"].append(
                opt_time_to_date(start_time, tc_start_time + time_size)
            )

        df_new_solution = pd.DataFrame(dict_new_solution)
        df_new_technical_service = pd.DataFrame(dict_new_technical_service)
        return df_new_solution, df_new_technical_service


class DataOptManipulatorPulp(DataOptManipulatorMIP):
    @classmethod
    def get_results(
        cls,
        problem: FlightDistributionModelPulp,  # или BaseFlightDistributionModelPulp
    ) -> tp.Tuple[pd.DataFrame, ...]:
        if problem.flight_assignment is None:
            df_new_solution = pd.DataFrame(
                columns=[
                    "departure_time",
                    "arrival_time",
                    "flight_id",
                    "departure_airport_code",
                    "arrival_airport_code",
                    "aircraft_id",
                    "previous_solution_id",
                ]
            )
            df_new_technical_service = pd.DataFrame(
                columns=[
                    "technical_service_id",
                    "aircraft_id",
                    "time_size",
                    "time_start",
                    "time_finish",
                ]
            )
            return df_new_solution, df_new_technical_service

        start_time = problem.data.parameters.start_with_delta
        dict_new_solution = defaultdict(list)

        flight_to_aircraft = dict()
        ts_to_time = dict()
        for aircraft in problem.data.aircraft.tasks:
            aid = aircraft.aircraft_id
            for node in problem.data.graph[aid].nodes:
                if node.node_type in {NodeType.FLIGHT, NodeType.TS}:
                    var = problem.flight_assignment[aid][node.node_id]
                    val = (
                        round(var, 5)
                        if isinstance(var, (int, float))
                        else round(var.varValue or 0.0, 5)
                    )
                    assert val in {0, 1}
                    if node.node_type == NodeType.FLIGHT and val == 1:
                        flight_to_aircraft[node.entity.opt_flight_task_id] = aid
                    elif node.node_type == NodeType.TS and val == 1:
                        ts_to_time[node.entity.technical_service_task_id] = node.start

        for task_f in problem.data.flight.tasks:
            dict_new_solution["departure_time"].append(
                opt_time_to_date(start_time, task_f.start)
            )
            dict_new_solution["arrival_time"].append(
                opt_time_to_date(start_time, task_f.end)
            )
            dict_new_solution["flight_id"].append(task_f.flight_id)
            dict_new_solution["departure_airport_code"].append(
                task_f.departure_airport_code
            )
            dict_new_solution["arrival_airport_code"].append(
                task_f.arrival_airport_code
            )
            dict_new_solution["aircraft_id"].append(
                flight_to_aircraft.get(task_f.opt_flight_task_id, 0)
            )
            dict_new_solution["previous_solution_id"].append(task_f.opt_flight_task_id)

        dict_new_technical_service = defaultdict(list)
        for task_tc in problem.data.technical_service.tasks:
            if task_tc.technical_service_task_id not in ts_to_time:
                continue
            tc_start_time = ts_to_time[task_tc.technical_service_task_id]
            time_size = task_tc.old_end - task_tc.old_start
            dict_new_technical_service["technical_service_id"].append(
                task_tc.technical_service_task_id
            )
            dict_new_technical_service["aircraft_id"].append(task_tc.aircraft_id)
            dict_new_technical_service["time_size"].append(
                opt_delta_time_to_date(time_size)
            )
            dict_new_technical_service["time_start"].append(
                opt_time_to_date(start_time, tc_start_time)
            )
            dict_new_technical_service["time_finish"].append(
                opt_time_to_date(start_time, tc_start_time + time_size)
            )

        df_new_solution = pd.DataFrame(dict_new_solution)
        df_new_technical_service = pd.DataFrame(dict_new_technical_service)
        return df_new_solution, df_new_technical_service
