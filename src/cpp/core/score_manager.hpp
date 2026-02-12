#pragma once

#include <queue>

#include "core/fdp_types.hpp"
#include "core/instance.hpp"

class ScoreManager
{
    DictInt<DictInt<Score>> group_aircrafts_score{};
    DictInt<NodeId> ranks_{};
    DictInt<Score> group_score_{};
    Score increase_value{};
    Score decrease_value{};

   public:
    void clear()
    {
        group_aircrafts_score = {};
        ranks_ = {};
        group_score_ = {};
        increase_value = {};
        decrease_value = {};
    }

    void setup(Instance const& inst, Score increase_value, Score decrease_value)
    {
        clear();
        ranks_ = computeTopologicalRanks(inst);
        group_aircrafts_score = DictInt<DictInt<Score>>(inst.num_group, DictInt<Score>(inst.num_aircrafts, 0));
        group_score_ = DictInt<Score>(inst.num_group, 0);
    }

    void increaseGroupAircraftsScore(GroupId group_id, AircraftId aircraft_id)
    {
        group_aircrafts_score[group_id][aircraft_id] += increase_value;
    }

    void decreaseGroupAircraftsScore(GroupId group_id, AircraftId aircraft_id)
    {
        group_aircrafts_score[group_id][aircraft_id] /= decrease_value;
    }

    void resetGroupAircraftsScore(GroupId group_id, AircraftId aircraft_id)
    {
        group_aircrafts_score[group_id][aircraft_id] = 0;
    }

    void increaseGroupScore(GroupId group_id) { group_score_[group_id] += increase_value; }

    void decreaseGroupScore(GroupId group_id) { group_score_[group_id] /= decrease_value; }

    void sortGroups(GroupList& groups, GroupId target_group = ABSENT_GROUP)
    {
        std::sort(groups.begin(),
                  groups.end(),
                  [target_group, this](GroupId a, GroupId b)
                  {
                      if (a == target_group)
                          return true;
                      if (b == target_group)
                          return false;
                      return group_score_[a] < group_score_[b];
                  });
    }

    NodeList sortedNodes(NodeList const& nodes, NodeId target_node, Instance const& inst)
    {
        NodeList nodes_ = nodes;
        std::sort(nodes_.begin(),
                  nodes_.end(),
                  [this, target_node, &inst](GroupId a, GroupId b)
                  {
                      auto a1 = std::abs(ranks_[target_node] - ranks_[a]);
                      auto a2 = std::abs(ranks_[target_node] - ranks_[b]);
                      auto l1 = (a1 < a2);
                      auto l2 = (group_aircrafts_score[inst.node_to_group[a]][inst.node_to_aircraft[a]] <
                                 group_aircrafts_score[inst.node_to_group[b]][inst.node_to_aircraft[b]]);
                      auto e = (a1 == a2);
                      return l1 || (e && l2);
                  });
        return nodes_;
    }

   private:
    DictInt<NodeId> computeTopologicalRanks(const Instance& instance)
    {
        const NodeId num_nodes = instance.num_nodes;

        DictInt<NodeId> in_degree(num_nodes, 0);
        for (NodeId u = 0; u < instance.next_nodes.size(); ++u)
        {
            for (NodeId v : instance.next_nodes[u])
            {
                ++in_degree[v];
            }
        }

        std::queue<NodeId> q;
        for (NodeId i = 0; i < num_nodes; ++i)
        {
            if (in_degree[i] == 0)
            {
                q.push(i);
            }
        }

        DictInt<NodeId> topological_rank(num_nodes, -1);
        int rank = 0;

        while (!q.empty())
        {
            NodeId u = q.front();
            q.pop();
            topological_rank[u] = rank++;

            for (NodeId v : instance.next_nodes[u])
            {
                if (instance.isVerticalNode(v))
                {
                    NodeId first_alt_v = instance.alt_nodes[instance.node_to_group[v]][0];
                    if (--in_degree[first_alt_v] == 0)
                    {
                        q.push(first_alt_v);
                    }
                }
                else
                {
                    if (--in_degree[v] == 0)
                    {
                        q.push(v);
                    }
                }
            }
        }

        for (NodeId v = 0; v < num_nodes; ++v)
        {
            if (instance.isVerticalNode(v))
            {
                NodeId first_alt_v = instance.alt_nodes[instance.node_to_group[v]][0];
                topological_rank[v] = topological_rank[first_alt_v];
            }
        }

        return topological_rank;
    }
};
