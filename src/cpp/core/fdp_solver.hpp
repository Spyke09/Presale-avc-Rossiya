#include <algorithm>
#include <cassert>
#include <chrono>
#include <queue>
#include <stack>

#include "app/spdlog_tmp.hpp"
#include "core/conflicts.hpp"
#include "core/fdp_types.hpp"
#include "core/instance.hpp"
#include "core/priority_queue.hpp"
#include "core/score_manager.hpp"
#include "core/solution_state.hpp"

class Solver
{
    using clock = std::chrono::steady_clock;

    DictInt<char> untaken_init_q_{};

    GroupList untaken_groups_init_{};
    DictInt<NodeList> aircraft_solution_{};
    DictInt<NodeList> aircraft_solution_init_{};
    Score return_objective_;
    bool feasible_q_{};
    bool solved_optimum_{};
    std::string return_status_;
    ConflictManager conflict_manager_{};

    ScoreManager score_manager_;

    void initStructures_(Instance const& inst)
    {
        aircraft_solution_init_.resize(inst.num_aircrafts);
        for (NodeId node_id : inst.start_point)
        {
            aircraft_solution_init_[inst.node_to_aircraft[node_id]].push_back(node_id);
        }
    }

    NodeList findMaxPath_(AircraftId aircraft_id, NodeList const& nodes, Instance const& inst)
    {
        return findMaxPath_(
            nodes, inst, inst.aircraft_to_source[aircraft_id], inst.aircraft_to_sink[aircraft_id], false);
    }

    NodeList findMaxPath_(
        NodeList const& nodes, Instance const& inst, NodeId start_node, NodeId end_node, bool inverse_q)
    {
        NodeSet nodes_set = NodeSet(nodes.begin(), nodes.end());
        return findMaxPath_(nodes_set, inst, start_node, end_node, inverse_q);
    }

    NodeList findMaxPath_(
        NodeSet const& nodes, Instance const& inst, NodeId start_node, NodeId end_node, bool inverse_q)
    {
        if (nodes.find(start_node) == nodes.end())
            return {};
        if (nodes.find(end_node) == nodes.end())
            return {};

        Dict<NodeId, int> dp_length;
        Dict<NodeId, Score> dp_cost;
        Dict<NodeId, int> indegree;
        Dict<NodeId, NodeId> parent;

        DictInt<NodeSet> const& next_nodes = inverse_q ? inst.prev_nodes : inst.next_nodes;
        std::unordered_map<NodeId, std::vector<NodeId>> local_next_nodes;
        for (NodeId node : nodes)
        {
            for (NodeId candidate : nodes)
            {
                if (next_nodes[node].find(candidate) != next_nodes[node].end())
                {
                    local_next_nodes[node].push_back(candidate);
                }
            }
        }

        for (NodeId node : nodes)
        {
            dp_length[node] = -1;
            dp_cost[node] = std::numeric_limits<int>::max();
            indegree[node] = 0;
            parent[node] = -1;
        }

        for (NodeId node : nodes)
        {
            for (NodeId next_node : local_next_nodes[node])
            {
                if (nodes.find(next_node) != nodes.end())
                {
                    indegree[next_node]++;
                }
            }
        }

        dp_length[start_node] = 0;
        dp_cost[start_node] = inst.node_cost[start_node];
        std::queue<NodeId> topo_q;
        topo_q.push(start_node);

        while (!topo_q.empty())
        {
            NodeId current = topo_q.front();
            topo_q.pop();

            for (NodeId next_node : local_next_nodes[current])
            {
                if (nodes.find(next_node) != nodes.end())
                {
                    int new_length = dp_length[current] + 1;
                    int new_cost = dp_cost[current] + inst.node_cost[next_node];

                    if (new_length > dp_length[next_node] ||
                        (new_length == dp_length[next_node] && new_cost < dp_cost[next_node]))
                    {
                        dp_length[next_node] = new_length;
                        dp_cost[next_node] = new_cost;
                        parent[next_node] = current;
                    }

                    indegree[next_node]--;
                    if (indegree[next_node] == 0)
                    {
                        topo_q.push(next_node);
                    }
                }
            }
        }

        NodeList path;

        if (dp_length[end_node] != -1)
        {
            for (NodeId cur = end_node; cur != -1; cur = parent[cur])
            {
                path.push_back(cur);
            }
            if (!inverse_q)
                std::reverse(path.begin(), path.end());
        }

        return path;
    }

    void addHorizontalGroups(GroupList const& untaken_groups,
                             Instance const& inst,
                             DictInt<NodeList>& aircraft_solution)
    {
        for (GroupId group : untaken_groups)
        {
            if (inst.group_to_horizontal_aircraft[group] == ABSENT_AIRCRAFT)
                continue;

            auto aircraft_id = inst.group_to_horizontal_aircraft[group];
            auto& alt_nodes = inst.alt_nodes[group];
            auto& solution = aircraft_solution[aircraft_id];

            for (auto alt_node : alt_nodes)
            {
                bool inserted = false;

                for (size_t i = 0; i < solution.size() - 1; ++i)
                {
                    auto cur = solution[i];
                    auto next = solution[i + 1];

                    if (inst.next_nodes[cur].count(alt_node) && inst.prev_nodes[next].count(alt_node))
                    {
                        solution.insert(solution.begin() + i + 1, alt_node);
                        inserted = true;
                        break;
                    }
                }

                if (inserted)
                    break;
            }
        }
    }

    void addHorizontalGroups(GroupList const& untaken_groups,
                             Instance const& inst,
                             NodeList& solution,
                             AircraftId aircraft_id)
    {
        for (GroupId group : untaken_groups)
        {
            if (inst.group_to_horizontal_aircraft[group] != aircraft_id)
                continue;

            auto& alt_nodes = inst.alt_nodes[group];

            for (auto alt_node : alt_nodes)
            {
                bool inserted = false;

                for (size_t i = 0; i < solution.size() - 1; ++i)
                {
                    auto cur = solution[i];
                    auto next = solution[i + 1];

                    if (inst.next_nodes[cur].count(alt_node) && inst.prev_nodes[next].count(alt_node))
                    {
                        solution.insert(solution.begin() + i + 1, alt_node);
                        inserted = true;
                        break;
                    }
                }

                if (inserted)
                    break;
            }
        }
    }

    GroupList untakenGroups_(DictInt<NodeList> const& aircraft_solution, Instance const& inst)
    {
        DictInt<char> group_take_status(inst.num_group, 0);

        for (AircraftId aircraft_id = 0; aircraft_id < inst.num_aircrafts; ++aircraft_id)
        {
            for (auto const& node : aircraft_solution[aircraft_id])
            {
                group_take_status[inst.node_to_group[node]] += 1;
            }
        }

        GroupList untaken_groups{};
        for (GroupId i = 0; i < inst.num_group; ++i)
        {
            if (group_take_status[i] == 0)
            {
                untaken_groups.push_back(i);
            }
            if (group_take_status[i] > 1)
            {
                logger_->error("Multiple nodes in group {} were used. This is invalid.", i);
                throw std::runtime_error("More than one element is included in a group.");
            }
        }

        return untaken_groups;
    }

    GroupList untakenGroups_(NodeList const& aircraft_solution, Instance const& inst)
    {
        DictInt<char> group_take_status(inst.num_group, 0);

        for (auto const& node : aircraft_solution)
        {
            group_take_status[inst.node_to_group[node]] += 1;
        }

        GroupList untaken_groups{};
        for (GroupId i = 0; i < inst.num_group; ++i)
        {
            if (group_take_status[i] == 0)
            {
                untaken_groups.push_back(i);
            }
            if (group_take_status[i] > 1)
            {
                logger_->error("Multiple nodes in group {} were used. This is invalid.", i);
                throw std::runtime_error("More than one element is included in a group.");
            }
        }

        return untaken_groups;
    }

    void repareInitSolution_(Instance const& inst)
    {
        logger_->info("Repairing initial solution...");
        for (AircraftId aircraft_id = 0; aircraft_id < inst.num_aircrafts; ++aircraft_id)
        {
            aircraft_solution_init_[aircraft_id] =
                findMaxPath_(aircraft_id, aircraft_solution_init_[aircraft_id], inst);
        }

        auto untaken_groups = untakenGroups_(aircraft_solution_init_, inst);
        addHorizontalGroups(untaken_groups, inst, aircraft_solution_init_);
        logger_->info("Initial solution repaired.");
    }

    NodeList reparedSolution_(Instance const& inst, AircraftId aircraft_id, NodeList nodes, NodeId required_node)
    {
        auto source = inst.aircraft_to_source[aircraft_id];
        auto sink = inst.aircraft_to_sink[aircraft_id];

        auto first_path = findMaxPath_(nodes, inst, source, required_node, false);
        if (first_path.empty())
        {
            return {};
        }
        auto second_path = findMaxPath_(nodes, inst, sink, required_node, true);
        if (second_path.empty())
        {
            return {};
        }

        NodeList solution;

        solution.reserve(first_path.size() + second_path.size() - 1);
        solution.insert(solution.end(), first_path.begin(), first_path.end() - 1);
        solution.insert(solution.end(), second_path.begin(), second_path.end());

        auto untaken_groups = untakenGroups_(solution, inst);
        addHorizontalGroups(untaken_groups, inst, solution, aircraft_id);
        return solution;
    }

    NodeList getAvailableNodes_(Instance const& inst,
                                GroupId group,
                                NodeId alt_node,
                                std::vector<GroupId> const& untaken_groups,
                                AircraftId aircraft_id,
                                NodeList const& base_path)
    {
        size_t estimate_size = base_path.size() + untaken_groups.size();
        NodeList result;
        result.reserve(estimate_size);

        result.insert(result.end(), base_path.begin(), base_path.end());

        for (GroupId g : untaken_groups)
        {
            if (g == group)
            {
                result.push_back(alt_node);
            }
            else
            {
                NodeId node = inst.group_aircraft_to_vertical_node[g][aircraft_id];
                if (node != ABSENT_NODE)
                {
                    result.push_back(node);
                }
            }
        }

        return result;
    }

    Score objective_(Instance const& inst, DictInt<NodeList> const& aircraft_solution)
    {
        Score score = 0;
        for (AircraftId aircraft_id = 0; aircraft_id < inst.num_aircrafts; ++aircraft_id)
        {
            for (auto& node : aircraft_solution[aircraft_id])
            {
                score += inst.node_cost[node];
            }
        }

        return score;
    }

    Score objective_(Instance const& inst, NodeList const& aircraft_solution)
    {
        Score score = 0;
        for (auto& node : aircraft_solution)
        {
            score += inst.node_cost[node];
        }

        return score;
    }

    DictInt<int> groupOrderByCostWithPriority_(Instance const& inst)
    {
        DictInt<Score> mean_cost(inst.num_group);
        for (GroupId g = 0; g < inst.num_group; ++g)
        {
            for (NodeId node : inst.alt_nodes[g])
            {
                mean_cost[g] += inst.node_cost[node];
            }
            mean_cost[g] /= (inst.alt_nodes[g].size() + 1);
        }

        std::vector<char> is_untaken(inst.num_group, 1);
        for (GroupId g : untaken_groups_init_)
        {
            is_untaken[g] = 0;
        }

        GroupList taken_groups;
        for (GroupId g = 0; g < inst.num_group; ++g)
        {
            if (is_untaken[g] == 0)
                continue;
            taken_groups.push_back(g);
        }

        std::sort(taken_groups.begin(),
                  taken_groups.end(),
                  [&](GroupId g1, GroupId g2) { return mean_cost[g1] > mean_cost[g2]; });

        GroupList sorted_groups;
        sorted_groups.reserve(inst.num_group);
        sorted_groups.insert(sorted_groups.end(), untaken_groups_init_.begin(), untaken_groups_init_.end());
        sorted_groups.insert(sorted_groups.end(), taken_groups.begin(), taken_groups.end());

        DictInt<int> group_rank(inst.num_group, -1);
        for (size_t i = 0; i < inst.num_group; ++i)
        {
            group_rank[sorted_groups[i]] = i;
        }

        return group_rank;
    }

    std::vector<int> vectorDifference_(const std::vector<int>& base, const std::vector<int>& target)
    {
        std::unordered_set<int> base_set(base.begin(), base.end());
        std::vector<int> result;

        for (int x : target)
        {
            if (base_set.find(x) == base_set.end())
            {
                result.push_back(x);
            }
        }

        return result;
    }

    bool allGroupInitUntaken_(GroupList const& untaken_groups)
    {
        for (GroupId group : untaken_groups)
        {
            if (!untaken_init_q_[group])
            {
                return false;
            }
        }
        return true;
    }

    void analyzeConflict(SolutionState const& solution_state, Conflict& conflict_nodes)
    {
        for (NodeId conflict_node : solution_state.node_history)
        {
            conflict_nodes.emplace(conflict_node);
        }
    }

    void analyzeConflictObjBounds(SolutionState const& solution_state, Instance const& inst)
    {
        Conflict conflict_nodes{};
        for (NodeId conflict_node : solution_state.node_history)
        {
            if (inst.node_cost[conflict_node] > 0)
                conflict_nodes.emplace(conflict_node);
        }
        conflict_manager_.add(conflict_nodes);
    }

    bool handleEmptyGroupsCase()
    {
        if (untaken_groups_init_.empty())
        {
            aircraft_solution_ = aircraft_solution_init_;
            return_objective_ = 0;
            feasible_q_ = true;
            return true;
        }
        return false;
    }

    void initializePriorityQueue(SwitchablePriorityQueue& pq, Instance const& inst)
    {
        auto copied_solution = aircraft_solution_init_;
        auto copied_untaken = untaken_groups_init_;

        pq.emplace(std::move(copied_solution),
                   GroupSet{},
                   std::move(copied_untaken),
                   std::set<NodeId>{},
                   ABSENT_NODE,
                   0,
                   static_cast<Score>(objective_(inst, aircraft_solution_init_)),
                   ABSENT_GROUP,
                   ABSENT_AIRCRAFT,
                   GroupSet{},
                   0,
                   NodeSet{});
    }

    bool timeLimitExceeded(size_t max_seconds,
                           std::chrono::_V2::steady_clock::time_point const& start_time,
                           SolutionState const& solution_state)
    {
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(clock::now() - start_time).count();

        if (elapsed_seconds > max_seconds)
        {
            aircraft_solution_ = solution_state.aircraft_solution;
            return true;
        }
        return false;
    }

    void logIteration(size_t iteration,
                      SolutionState const& solution_state,
                      SwitchablePriorityQueue& pq,
                      bool& switched_to_slow)
    {
        if (iteration % 100 == 0)
        {
            logger_->info(
                "Next iter {}: init groups left {}, depth {}, obj {}, conflict num {}, num cuts {}, "
                "mean conflicts size {}",
                iteration,
                untaken_groups_init_.size() - solution_state.taken_groups,
                solution_state.depth.size(),
                solution_state.score_objective,
                conflict_manager_.size(),
                conflict_manager_.numCuts(),
                conflict_manager_.averageConflictSize());

            pq.switch_comp(!switched_to_slow);
            switched_to_slow = !switched_to_slow;
        }
    }

    template <bool OptimizingQ>
    bool processNode(const Instance& inst,
                     const SolutionState& solution_state,
                     GroupId group,
                     NodeId alt_node,
                     std::vector<SolutionState>& pq_candidates,
                     bool& is_conflict,
                     Score& upper_bound)
    {
        auto aircraft_id = inst.node_to_aircraft[alt_node];

        NodeList available_nodes = getAvailableNodes_(inst,
                                                      group,
                                                      alt_node,
                                                      solution_state.untaken_groups,
                                                      aircraft_id,
                                                      solution_state.aircraft_solution[aircraft_id]);

        auto new_path = reparedSolution_(inst, aircraft_id, available_nodes, alt_node);

        if (new_path.empty())
        {
            score_manager_.increaseGroupAircraftsScore(group, aircraft_id);
            score_manager_.increaseGroupScore(group);
            return false;
        }

        Score cur_score = solution_state.score_objective -
                          objective_(inst, solution_state.aircraft_solution[aircraft_id]) + objective_(inst, new_path);

        DictInt<NodeList> new_aircraft_solution = solution_state.aircraft_solution;
        new_aircraft_solution[aircraft_id] = new_path;

        auto cur_untaken_groups = untakenGroups_(new_aircraft_solution, inst);
        for (auto cur_group : cur_untaken_groups)
        {
            if (solution_state.group_history.count(cur_group))
                return false;
        }

        if (solution_state.aircraft_solution[aircraft_id].size() < new_path.size())
            score_manager_.decreaseGroupAircraftsScore(group, aircraft_id);
        else
            score_manager_.increaseGroupAircraftsScore(group, aircraft_id);

        auto group_diff = vectorDifference_(solution_state.untaken_groups, cur_untaken_groups);
        if (group_diff.empty())
            score_manager_.decreaseGroupScore(group);
        else
            score_manager_.increaseGroupScore(group);

        if (cur_untaken_groups.empty())
        {
            is_conflict = false;
            if constexpr (OptimizingQ)
            {
                if (cur_score < upper_bound)
                {
                    logger_->info("Found new solution with objective {}", cur_score);
                    aircraft_solution_ = new_aircraft_solution;
                    return_objective_ = cur_score;
                    upper_bound = cur_score;
                }
                return false;
            }
            else
            {
                logger_->info("Found new solution with objective {}", cur_score);
                aircraft_solution_ = new_aircraft_solution;
                return_objective_ = cur_score;
                feasible_q_ = true;
                return true;
            }
        }

        auto new_group_history = solution_state.group_history;
        new_group_history.emplace(group);
        auto new_node_history = solution_state.node_history;
        new_node_history.emplace(alt_node);

        auto blocked = findAllowedAndBlockedNodesForAircraft(
            inst.aircraft_nodes[aircraft_id], inst, aircraft_id, new_aircraft_solution[aircraft_id], new_node_history);
        for (auto i : solution_state.blocked_nodes) blocked.insert(i);

        Score cur_lower_bound_copy = cur_score;
        if constexpr (OptimizingQ)
        {
            for (GroupId g : cur_untaken_groups)
            {
                Score best = 10000000;
                for (NodeId node : inst.alt_nodes[g])
                {
                    if (blocked.count(node))
                        continue;
                    best = std::min(best, inst.node_cost[node]);
                }
                cur_lower_bound_copy += best;
            }

            if (upper_bound <= cur_lower_bound_copy)
            {
                return false;
            }
        }

        if (conflict_manager_.contains(new_node_history))
            return false;

        auto depth = solution_state.depth;
        if (allGroupInitUntaken_(solution_state.untaken_groups))
            depth = {};
        for (auto ug : cur_untaken_groups)
        {
            if (!untaken_init_q_[ug])
                depth.emplace(ug);
        }

        GroupSet taken_groups_set = GroupSet(untaken_groups_init_.begin(), untaken_groups_init_.end());
        for (auto ug : cur_untaken_groups)
            if (untaken_init_q_[ug])
                taken_groups_set.erase(ug);

        is_conflict = false;

        pq_candidates.emplace_back(
            std::move(new_aircraft_solution),
            std::move(new_group_history),
            std::move(cur_untaken_groups),
            std::move(new_node_history),
            alt_node,
            static_cast<Score>(cur_untaken_groups.size()),
            cur_score,
            group_diff.size() > 0 ? group_diff[0] : ABSENT_GROUP,
            group_diff.size() > 0
                ? (solution_state.best_aircraft != ABSENT_AIRCRAFT ? solution_state.best_aircraft : aircraft_id)
                : solution_state.best_aircraft,
            std::move(depth),
            taken_groups_set.size(),
            std::move(blocked));

        return false;
    }

    template <bool OptimizingQ>
    bool processGroups(const Instance& inst,
                       const SolutionState& solution_state,
                       std::vector<SolutionState>& pq_candidates,
                       Score& upper_bound)
    {
        auto untaken_groups = solution_state.untaken_groups;
        bool all_init_groups = allGroupInitUntaken_(untaken_groups);
        for (GroupId g : untaken_groups)
        {
            if (untaken_init_q_[g])
                score_manager_.decreaseGroupScore(g);
        }

        if (conflict_manager_.contains(solution_state.node_history))
            return false;

        score_manager_.sortGroups(untaken_groups, solution_state.best_group);

        for (GroupId group : untaken_groups)
        {
            if ((!all_init_groups) && (untaken_init_q_[group]))
                continue;

            if (solution_state.best_aircraft != ABSENT_AIRCRAFT)
                score_manager_.resetGroupAircraftsScore(group, solution_state.best_aircraft);

            auto ltn = solution_state.last_taken_node;
            auto sorted_nodes = score_manager_.sortedNodes(inst.alt_nodes[group], ltn, inst);

            bool is_conflict = true;
            for (NodeId alt_node : sorted_nodes)
            {
                auto success = processNode<OptimizingQ>(
                    inst, solution_state, group, alt_node, pq_candidates, is_conflict, upper_bound);
                if constexpr (!OptimizingQ)
                {
                    if (success)
                        return true;
                }
            }

            if (is_conflict)
            {
                analyzeConflictObjBounds(solution_state, inst);
                break;
            }
        }

        return false;
    }

    template <bool OptimizingQ>
    bool mainLoop_(const Instance& inst,
                   size_t max_seconds,
                   const std::chrono::_V2::steady_clock::time_point& start_time,
                   bool fast_comp)
    {
        SwitchablePriorityQueue pq(fast_comp);
        size_t iteration = 0;
        bool switched_to_slow = fast_comp;

        if constexpr (!OptimizingQ)
        {
            if (handleEmptyGroupsCase())
                return true;
        }

        initializePriorityQueue(pq, inst);
        Score upper_bound = return_objective_;

        while (!pq.empty())
        {
            auto solution_state = pq.top();
            pq.pop();

            if (timeLimitExceeded(max_seconds, start_time, solution_state))
                return false;

            ++iteration;
            logIteration(iteration, solution_state, pq, switched_to_slow);

            if constexpr (OptimizingQ)
            {
                Score cur_lower_bound = solution_state.score_objective;
                if (upper_bound <= cur_lower_bound)
                {
                    analyzeConflictObjBounds(solution_state, inst);
                    continue;
                }
            }

            std::vector<SolutionState> pq_candidates;

            auto sucess = processGroups<OptimizingQ>(inst, solution_state, pq_candidates, upper_bound);
            if constexpr (!OptimizingQ)
            {
                if (sucess)
                    return true;
            }

            for (auto& pq_candidate : pq_candidates) pq.push(std::move(pq_candidate));
        }

        if constexpr (!OptimizingQ)
        {
            feasible_q_ = false;
        }

        return true;
    }

    void initUntakenGroups(Instance const& inst)
    {
        untaken_groups_init_ = untakenGroups_(aircraft_solution_init_, inst);
        untaken_init_q_ = DictInt<char>(inst.num_group, false);
        for (auto g : untaken_groups_init_) untaken_init_q_[g] = true;
    }

   public:
    void setup(Instance const& inst)
    {
        logger_->info("Running setup...");
        clear();

        initStructures_(inst);

        score_manager_ = ScoreManager();
        score_manager_.setup(inst, 100, 2);
        conflict_manager_ = ConflictManager(2000);
        conflict_manager_.setup();
    }

    void clear()
    {
        untaken_init_q_ = {};
        untaken_groups_init_ = {};
        aircraft_solution_ = {};
        return_objective_ = 0;
        feasible_q_ = 0;
        solved_optimum_ = 0;
        return_status_ = "UNDEFINED";
        score_manager_ = {};
        conflict_manager_ = {};
    }

    void solve(Instance const& inst, size_t max_seconds, bool return_first, bool fast_comp_feasible, bool fast_comp_opt)
    {
        logger_->info("Solving instance...");
        repareInitSolution_(inst);
        initUntakenGroups(inst);

        auto start_time = clock::now();

        bool solved_loop_feasible = mainLoop_<false>(inst, max_seconds, start_time, fast_comp_feasible);
        if (!return_first && solved_loop_feasible && feasible_q_)
        {
            logger_->info("Starting optimizing...");
            solved_optimum_ = mainLoop_<true>(inst, max_seconds, start_time, fast_comp_opt);
        }

        if (!solved_loop_feasible)
            return_status_ = "UNDEFINED";
        else if (!feasible_q_)
            return_status_ = "INFEASIBLE";
        else if (!solved_optimum_)
            return_status_ = "NON-OPTIMAL";
        else
            return_status_ = "OPTIMAL";

        logger_->info(
            "Obj {}, conflict num {}, num cuts {}, "
            "mean conflicts size {}",
            return_objective_,
            conflict_manager_.size(),
            conflict_manager_.numCuts(),
            conflict_manager_.averageConflictSize());

        logger_->info("Solving completed.");
    }

    NodeList getSolution(Instance const& inst)
    {
        NodeList solution;
        for (AircraftId aircraft_id = 0; aircraft_id < inst.num_aircrafts; ++aircraft_id)
        {
            for (auto& node : aircraft_solution_[aircraft_id])
            {
                solution.push_back(node);
            }
        }
        logger_->info("Returning solution with {} total nodes.", solution.size());
        return solution;
    }

    Score getObjective() { return return_objective_; }
    std::string getStatus() { return return_status_; }

    void setLogger(std::shared_ptr<spdlog::logger> logger) { logger_ = std::move(logger); }

    NodeSet findAllowedAndBlockedNodesForAircraft(NodeList const& allowed_nodes,
                                                  Instance const& inst,
                                                  AircraftId a,
                                                  NodeList const& aircraft_path,  // solution_state.aircraft_solution[a]
                                                  std::set<NodeId> const& node_history)  // обязательные вершины
    {
        DictInt<NodeSet> const& next_nodes = inst.next_nodes;
        DictInt<NodeSet> const& prev_nodes = inst.prev_nodes;

        NodeSet allowed_set(allowed_nodes.begin(), allowed_nodes.end());

        // --------- восстановим порядок обязательных вершин ---------
        NodeList checkpoints;
        checkpoints.push_back(inst.aircraft_to_source[a]);
        for (NodeId v : aircraft_path)
            if (node_history.count(v))
                checkpoints.push_back(v);
        checkpoints.push_back(inst.aircraft_to_sink[a]);

        NodeSet globally_allowed;

        // --------- обрабатываем каждый сегмент ---------
        for (size_t i = 0; i + 1 < checkpoints.size(); ++i)
        {
            NodeId start = checkpoints[i];
            NodeId end = checkpoints[i + 1];

            // ---- reachable_from_start ----
            NodeSet reachable_from_start;
            {
                std::stack<NodeId> st;
                st.push(start);
                reachable_from_start.insert(start);

                while (!st.empty())
                {
                    NodeId u = st.top();
                    st.pop();

                    for (NodeId v : next_nodes[u])
                    {
                        if (!allowed_set.count(v))
                            continue;
                        if (!reachable_from_start.count(v))
                        {
                            reachable_from_start.insert(v);
                            st.push(v);
                        }
                    }
                }
            }

            // ---- can_reach_end ----
            NodeSet can_reach_end;
            {
                std::stack<NodeId> st;
                if (allowed_set.count(end))
                {
                    st.push(end);
                    can_reach_end.insert(end);
                }

                while (!st.empty())
                {
                    NodeId u = st.top();
                    st.pop();

                    for (NodeId v : prev_nodes[u])
                    {
                        if (!allowed_set.count(v))
                            continue;
                        if (!can_reach_end.count(v))
                        {
                            can_reach_end.insert(v);
                            st.push(v);
                        }
                    }
                }
            }

            // ---- пересечение ----
            for (NodeId v : reachable_from_start)
                if (can_reach_end.count(v))
                    globally_allowed.insert(v);
        }

        // --------- blocked = allowed_nodes \ globally_allowed ---------
        NodeSet blocked;
        for (NodeId v : allowed_set)
            if (!globally_allowed.count(v))
                blocked.insert(v);

        return blocked;
    }
};
