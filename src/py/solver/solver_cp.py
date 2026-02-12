from collections import defaultdict

import docplex.cp.expression

from src.py.problem_structures import FlightDistributionProblemCP
from src.py.settings import app_param, solver_param
from src.py.solver.base_solver import BaseFlightDistributionModelCP


class FlightDistributionModelCP(BaseFlightDistributionModelCP):
    def __init__(
        self,
        data: FlightDistributionProblemCP,
        timelimit: int = solver_param.timelimit,
        workers: int = solver_param.workers,
        log_verbosity: str = solver_param.log_verbosity,
        search_type: str = solver_param.search_type,
        temporal_relaxation: str = solver_param.temporal_relaxation,
    ):
        super().__init__(
            data, timelimit, workers, log_verbosity, search_type, temporal_relaxation
        )
        self.data: FlightDistributionProblemCP = data

    def add_variables(self) -> None:
        # intervals
        self._var_flight_tasks()
        self._var_alt_flight_tasks()
        self._var_technical_service_tasks()
        self._var_aircraft_max_downtime_task()
        # sequences
        self._seq_aircraft_flights_seq()
        self._seq_aircraft_flights_technical_service_seq()

    def _var_flight_tasks(self) -> None:
        self.flight_tasks = {}
        for task_f in self.data.flight.tasks:
            self.flight_tasks[task_f.opt_flight_task_id] = self.model.interval_var(
                name=None,
                start=(task_f.start,) * 2,
                end=(task_f.end,) * 2,
                optional=False,
            )

    def _var_alt_flight_tasks(self) -> None:
        self.alt_flight_tasks = {}
        for task_af in self.data.alt_flight.tasks:
            self.alt_flight_tasks[task_af.opt_alt_flight_task_id] = (
                self.model.interval_var(
                    name=None,
                    start=(task_af.start,) * 2,
                    end=(task_af.end,) * 2,
                    optional=True,
                )
            )

    def _var_technical_service_tasks(self) -> None:
        self.technical_service_tasks = {}
        for task_ts in self.data.technical_service.tasks:
            if task_ts.is_fixed:
                self.technical_service_tasks[task_ts.technical_service_task_id] = (
                    self.model.interval_var(
                        name=None,
                        start=(task_ts.old_start, task_ts.old_start),
                        end=(task_ts.old_end, task_ts.old_end),
                        size=(task_ts.old_end - task_ts.old_start,) * 2,
                        optional=False,
                    )
                )
            else:
                self.technical_service_tasks[task_ts.technical_service_task_id] = (
                    self.model.interval_var(
                        name=None,
                        size=(task_ts.old_end - task_ts.old_start,) * 2,
                        start=(
                            self.data.parameters.start_horizon_opt,
                            docplex.cp.expression.INTERVAL_MAX,
                        ),
                        optional=False,
                    )
                )

    def _var_aircraft_max_downtime_task(self) -> None:
        self.aircraft_max_downtime_task = {}
        for task_a in self.data.aircraft.tasks:
            self.aircraft_max_downtime_task[task_a.aircraft_id] = (
                self.model.interval_var(
                    name=None,
                    start=(
                        self.data.parameters.start_horizon_opt,
                        self.data.parameters.end_horizon_opt,
                    ),
                    end=(
                        self.data.parameters.start_horizon_opt,
                        self.data.parameters.end_horizon_opt,
                    ),
                    optional=False,
                )
            )

    def _seq_aircraft_flights_seq(self) -> None:
        self.aircraft_flights_seq = {}
        for task_a in self.data.aircraft.tasks:
            self.aircraft_flights_seq[task_a.aircraft_id] = self.model.sequence_var(
                name=None,
                vars=[
                    self.alt_flight_tasks[task_af.opt_alt_flight_task_id]
                    for task_af in self.data.alt_flight.tasks
                    if (task_af.alt_aircraft_id == task_a.aircraft_id)
                    and task_af.opt_flight_task_id
                    in self.data.flight.opt_flight_task_id_for_turnaround
                ],
                types=[
                    task_af.opt_alt_flight_task_id
                    for task_af in self.data.alt_flight.tasks
                    if (task_af.alt_aircraft_id == task_a.aircraft_id)
                    and task_af.opt_flight_task_id
                    in self.data.flight.opt_flight_task_id_for_turnaround
                ],
            )

    def _seq_aircraft_flights_technical_service_seq(self):
        self.aircraft_flights_technical_service_seq = {}
        for task_a in self.data.aircraft.tasks:
            vars_ = [
                self.alt_flight_tasks[task_af.opt_alt_flight_task_id]
                for task_af in self.data.alt_flight.tasks
                if (task_af.alt_aircraft_id == task_a.aircraft_id)
            ] + [
                self.technical_service_tasks[task_ts.technical_service_task_id]
                for task_ts in self.data.technical_service.tasks
                if task_ts.aircraft_id == task_a.aircraft_id
            ]
            types = [
                int(task_af.arrival_airport_code in self.data.airports.base_airports)
                for task_af in self.data.alt_flight.tasks
                if (task_af.alt_aircraft_id == task_a.aircraft_id)
            ] + [
                2
                for task_ts in self.data.technical_service.tasks
                if task_ts.aircraft_id == task_a.aircraft_id
            ]
            self.aircraft_flights_technical_service_seq[task_a.aircraft_id] = (
                self.model.sequence_var(
                    name=None,
                    vars=vars_,
                    types=types,
                )
            )

    def add_expressions(self) -> None:
        self._expr_flight_similarity_obj_min()
        self._expr_technical_service_difference_obj_min()
        self._expr_aircraft_max_downtime_obj_max()
        self._expr_num_different_airports_of_aircraft_obj_min()

    def _expr_flight_similarity_obj_min(self) -> None:
        self.flight_similarity_obj_min = self.model.sum(
            self.model.presence_of(
                self.alt_flight_tasks[task_af.opt_alt_flight_task_id],
            )
            for task_af in self.data.alt_flight.tasks
            if (task_af.old_aircraft_id != task_af.alt_aircraft_id)
            and (task_af.start >= self.data.parameters.start_horizon_opt)
            and (task_af.start <= self.data.parameters.end_horizon_opt)
        )

    def _expr_technical_service_difference_obj_min(self) -> None:
        self.technical_service_difference_obj_min = self.model.sum(
            self.model.abs(
                self.model.start_of(
                    self.technical_service_tasks[task_ts.technical_service_task_id]
                )
                - task_ts.old_start
            )
            for task_ts in self.data.technical_service.tasks
        )

    def _expr_aircraft_max_downtime_obj_max(self):
        self.aircraft_max_downtime_task_obj_max = self.model.max(
            self.model.size_of(self.aircraft_max_downtime_task[task_a.aircraft_id])
            for task_a in self.data.aircraft.tasks
        )

    def _expr_num_different_airports_of_aircraft_obj_min(self):
        aa_dict = defaultdict(list)
        for af_task in self.data.alt_flight.tasks:
            if (
                af_task.start >= self.data.parameters.start_with_delta_opt
            ) and af_task.end <= self.data.parameters.end_with_delta_opt:
                aa_dict[
                    (af_task.alt_aircraft_id, af_task.departure_airport_code)
                ].append(af_task.opt_alt_flight_task_id)
                aa_dict[(af_task.alt_aircraft_id, af_task.arrival_airport_code)].append(
                    af_task.opt_alt_flight_task_id
                )

        self.num_different_airports_of_aircraft_obj_min = self.model.sum(
            self.model.sum(
                self.model.logical_and(
                    self.model.presence_of(
                        self.alt_flight_tasks[opt_alt_flight_task_id]
                    )
                    for opt_alt_flight_task_id in aa_dict[task_a.aircraft_id, airport]
                )
                for airport in self.data.airports.airports
            )
            for task_a in self.data.aircraft.tasks
        )

    def add_obj_function(self) -> None:
        obj_function_components = [
            self.flight_similarity_obj_min,
            self.technical_service_difference_obj_min,
        ]

        self.model.add(self.model.minimize_static_lex(obj_function_components))

    def add_constraints(self) -> None:
        self._constr_alternative_rule_on_flight_task()
        self._constr_no_overlap_rules()
        self._constr_if_then_rules_on_flights_turnaround()
        self._constr_if_then_rules_on_flights_bundle()
        self._constr_if_then_rules_on_flights_technical_services()

    def _constr_alternative_rule_on_flight_task(self) -> None:
        for task_f in self.data.flight.tasks:
            self.model.add(
                self.model.alternative(
                    interval=self.flight_tasks[task_f.opt_flight_task_id],
                    array=[
                        self.alt_flight_tasks[task_af.opt_alt_flight_task_id]
                        for task_af in self.data.alt_flight.tasks
                        if task_af.opt_flight_task_id == task_f.opt_flight_task_id
                    ],
                )
            )

    def _constr_no_overlap_rules(self) -> None:
        for task_a in self.data.aircraft.tasks:
            vars_list = [
                self.alt_flight_tasks[task_af.opt_alt_flight_task_id]
                for task_af in self.data.alt_flight.tasks
                if (task_af.alt_aircraft_id == task_a.aircraft_id)
                and task_af.opt_flight_task_id
                in self.data.flight.opt_flight_task_id_for_turnaround
            ] + [self.aircraft_max_downtime_task[task_a.aircraft_id]]

            if vars_list:
                self.model.add(self.model.no_overlap(vars_list))

            self.model.add(
                self.model.no_overlap(self.aircraft_flights_seq[task_a.aircraft_id])
            )
            self.model.add(
                self.model.no_overlap(
                    self.aircraft_flights_technical_service_seq[task_a.aircraft_id]
                )
            )

    def _constr_if_then_rules_on_flights_turnaround(self) -> None:
        for task_af_prev in self.data.alt_flight.tasks:
            for task_af_next in self.data.alt_flight.tasks:
                if (
                    task_af_prev != task_af_next
                    and task_af_prev.alt_aircraft_id == task_af_next.alt_aircraft_id
                    and task_af_prev.opt_flight_task_id
                    in self.data.flight.opt_flight_task_id_for_turnaround
                    and task_af_next.opt_flight_task_id
                    in self.data.flight.opt_flight_task_id_for_turnaround
                ):
                    turnaround_time = self.model.start_of(
                        self.alt_flight_tasks[task_af_next.opt_alt_flight_task_id]
                    ) - self.model.end_of(
                        self.alt_flight_tasks[task_af_prev.opt_alt_flight_task_id]
                    )
                    self.model.add(
                        self.model.if_then(
                            self.model.type_of_prev(
                                self.aircraft_flights_seq[task_af_prev.alt_aircraft_id],
                                self.alt_flight_tasks[
                                    task_af_next.opt_alt_flight_task_id
                                ],
                                app_param.absent_interval_var_position,
                                app_param.absent_interval_var_position,
                            )
                            == task_af_prev.opt_alt_flight_task_id,
                            turnaround_time >= task_af_prev.turnaround_time,
                        )
                    )

    def _constr_if_then_rules_on_flights_bundle(self) -> None:
        for task_af_prev in self.data.alt_flight.tasks:
            for task_af_next in self.data.alt_flight.tasks:
                if (
                    task_af_prev != task_af_next
                    and task_af_prev.alt_aircraft_id == task_af_next.alt_aircraft_id
                    and task_af_prev.opt_flight_task_id
                    in self.data.flight.opt_flight_task_id_for_turnaround
                    and task_af_next.opt_flight_task_id
                    in self.data.flight.opt_flight_task_id_for_turnaround
                ):
                    bundle_q = (
                        task_af_prev.arrival_airport_code
                        == task_af_next.departure_airport_code
                    )
                    if not bundle_q:
                        self.model.add(
                            self.model.type_of_prev(
                                self.aircraft_flights_seq[task_af_prev.alt_aircraft_id],
                                self.alt_flight_tasks[
                                    task_af_next.opt_alt_flight_task_id
                                ],
                                app_param.absent_interval_var_position,
                                app_param.absent_interval_var_position,
                            )
                            != task_af_prev.opt_alt_flight_task_id
                        )

    def _constr_if_then_rules_on_flights_technical_services(self) -> None:
        for task_ts in self.data.technical_service.tasks:
            if task_ts.aircraft_id in self.aircraft_flights_technical_service_seq:
                self.model.add(
                    self.model.type_of_prev(
                        self.aircraft_flights_technical_service_seq[
                            task_ts.aircraft_id
                        ],
                        self.technical_service_tasks[task_ts.technical_service_task_id],
                        app_param.absent_interval_var_position,
                        app_param.absent_interval_var_position,
                    )
                    != 0
                )

    def add_warm_start(self):
        stp = self.model.create_empty_solution()
        for task_af in self.data.alt_flight.tasks:
            if task_af.alt_aircraft_id == task_af.old_aircraft_id:
                stp[self.alt_flight_tasks[task_af.opt_alt_flight_task_id]] = (
                    task_af.start,
                    task_af.end,
                    task_af.end - task_af.start,
                )
        for task_ts in self.data.technical_service.tasks:
            stp[self.technical_service_tasks[task_ts.technical_service_task_id]] = (
                task_ts.old_start,
                task_ts.old_end,
                task_ts.old_end - task_ts.old_start,
            )
        self.model.set_starting_point(stp)
