import enum
import typing as tp
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Parameters:
    start_horizon: pd.Timestamp
    end_horizon: pd.Timestamp
    start_horizon_opt: int
    end_horizon_opt: int
    start_with_delta: pd.Timestamp
    end_with_delta: pd.Timestamp
    start_with_delta_opt: int
    end_with_delta_opt: int


@dataclass(frozen=True)
class FlightTask:
    opt_flight_task_id: int
    flight_id: int
    departure_airport_code: str
    arrival_airport_code: str
    old_aircraft_id: int
    start: int
    end: int
    is_fixed: bool


@dataclass(frozen=True)
class Flight:
    tasks: tp.List[FlightTask]
    opt_flight_task_id_for_turnaround: tp.List[int]
    task_by_id: tp.Dict[int, FlightTask]


@dataclass(frozen=True)
class AlternativeFlightTask(FlightTask):
    opt_alt_flight_task_id: int
    opt_flight_task_id: int
    old_aircraft_id: int
    alt_aircraft_id: int
    start: int
    end: int
    turnaround_time: int
    departure_airport_code: str
    arrival_airport_code: str
    is_fixed: bool
    delayed: bool


@dataclass(frozen=True)
class AlternativeFlight:
    tasks: tp.List[AlternativeFlightTask]


@dataclass(frozen=True)
class AircraftTask:
    aircraft_id: int
    reserve_q: bool


@dataclass(frozen=True)
class Aircraft:
    tasks: tp.List[AircraftTask]


@dataclass(frozen=True)
class TechnicalServiceTask:
    technical_service_task_id: int
    aircraft_id: int
    old_start: int
    old_end: int
    is_fixed: bool


@dataclass(frozen=True)
class AlternativeTechnicalServiceTask(TechnicalServiceTask):
    alt_technical_service_task_id: int
    technical_service_task_id: int
    aircraft_id: int
    start: int
    end: int
    old_start: int
    old_end: int
    base_airport: str
    is_fixed: bool


@dataclass(frozen=True)
class TechnicalService:
    tasks: tp.List[TechnicalServiceTask]


@dataclass(frozen=True)
class AlternativeTechnicalService:
    tasks: tp.List[AlternativeTechnicalServiceTask]


@dataclass(frozen=True)
class Airports:
    airports: tp.List[str]
    base_airports: tp.List[str]


class NodeType(enum.Enum):
    FLIGHT = enum.auto()
    TS = enum.auto()
    SINK = enum.auto()
    SOURCE = enum.auto()


@dataclass(frozen=True)
class Node:
    node_id: int
    node_type: NodeType
    start: int
    end: int
    start_airport: str
    end_airport: str
    aircraft_id: int

    entity: tp.Union[
        None,
        FlightTask,
        TechnicalServiceTask,
        AlternativeFlightTask,
        AlternativeTechnicalServiceTask,
    ]

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.node_id == other.node_id


@dataclass(frozen=True)
class FrozenChain:
    aircraft_id: int
    chain_left: tp.List[Node]
    chain_right: tp.List[Node]


@dataclass(frozen=True)
class Graph:
    nodes: tp.List[Node]
    next_nodes: tp.Dict[int, tp.List[Node]]
    prev_nodes: tp.Dict[int, tp.List[Node]]


@dataclass(frozen=True)
class FlightDistributionProblemBase:
    pass


@dataclass(frozen=True)
class FlightDistributionProblemCP(FlightDistributionProblemBase):
    aircraft: Aircraft
    airports: Airports
    flight: Flight
    alt_flight: AlternativeFlight
    technical_service: TechnicalService
    parameters: Parameters


@dataclass(frozen=True)
class FlightDistributionProblemMIP(FlightDistributionProblemBase):
    aircraft: Aircraft
    flight_to_nodes: tp.Dict[int, tp.List[Node]]
    graph: tp.Dict[int, Graph]
    ts_to_nodes: tp.Dict[int, tp.List[Node]]
    parameters: Parameters
    technical_service: TechnicalService
    flight: Flight


@dataclass(frozen=True)
class FlightDistributionProblemCPP(FlightDistributionProblemBase):
    aircraft_nodes: tp.Dict[int, tp.List[int]]
    alt_nodes: tp.List[tp.List[int]]
    node_cost: tp.Dict[int, float]
    edges: tp.List[tp.Tuple[int, int]]

    aircraft_cpp_id_to_opt_id: tp.Dict[int, int]
    node_cpp_id_to_opt_id: tp.Dict[int, int]

    start_point: tp.List[int]

    def get_data_for_save(self):
        # self.check_aircraft_sources_and_sinks()
        data = {
            "aircraft_nodes": {str(i): j for i, j in self.aircraft_nodes.items()},
            "alt_nodes": self.alt_nodes,
            "node_cost": {str(i): j for i, j in self.node_cost.items()},
            "edges": self.edges,
            "start_point": self.start_point,
        }
        return data

    def find_sources_and_sinks(self) -> tp.Dict[int, tp.Tuple[int, int]]:
        result = {}
        # Для быстрого поиска принадлежности вершины к самолету — инвертируем dict
        node_to_aircraft = {}
        for ac_id, nodes in self.aircraft_nodes.items():
            for n in nodes:
                node_to_aircraft[n] = ac_id

        # Для каждого самолёта делаем множества входящих и исходящих вершин
        for ac_id, nodes in self.aircraft_nodes.items():
            nodes_set = set(nodes)
            incoming = {n: 0 for n in nodes}
            outgoing = {n: 0 for n in nodes}

            for u, v in self.edges:
                # Проверяем ребра внутри этого самолета
                if u in nodes_set and v in nodes_set:
                    outgoing[u] += 1
                    incoming[v] += 1

            # Исток — вершина с нулём входящих
            sources = [n for n in nodes if incoming[n] == 0]
            # Сток — вершина с нулём исходящих
            sinks = [n for n in nodes if outgoing[n] == 0]

            if len(sources) != 1 or len(sinks) != 1:
                raise ValueError(
                    f"Aircraft {ac_id} should have exactly one source and one sink, found {sources} and {sinks}"
                )

            result[ac_id] = (sources[0], sinks[0])

        return result
