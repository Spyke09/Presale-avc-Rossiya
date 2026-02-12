#pragma once

#include "core/fdp_types.hpp"

struct SolutionState
{
    DictInt<NodeList> aircraft_solution{};
    GroupSet group_history{};
    GroupList untaken_groups{};
    std::set<NodeId> node_history{};
    NodeId last_taken_node{ABSENT_NODE};
    Score score_untaken_groups;
    Score score_objective;
    GroupId best_group{};
    AircraftId best_aircraft{};
    GroupSet depth;
    Score taken_groups{};
    NodeSet blocked_nodes{};

    SolutionState(DictInt<NodeList>&& aircraft_solution_,
                  GroupSet&& group_history_,
                  GroupList&& untaken_groups_,
                  std::set<NodeId>&& node_history_,
                  NodeId last_taken_node_,
                  Score score_untaken_groups_,
                  Score score_objective_,
                  GroupId best_group_,
                  AircraftId best_aircraft_,
                  GroupSet&& depth_,
                  Score taken_groups_,
                  NodeSet&& blocked_nodes_) :
            aircraft_solution(std::move(aircraft_solution_)),
            group_history(std::move(group_history_)),
            untaken_groups(std::move(untaken_groups_)),
            node_history(std::move(node_history_)),
            last_taken_node(last_taken_node_),
            score_untaken_groups(score_untaken_groups_),
            score_objective(score_objective_),
            best_group(best_group_),
            best_aircraft(best_aircraft_),
            depth(std::move(depth_)),
            taken_groups(taken_groups_),
            blocked_nodes(std::move(blocked_nodes_))
    {
    }
};
