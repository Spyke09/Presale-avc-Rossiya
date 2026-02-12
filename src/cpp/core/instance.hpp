#pragma once
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "core/fdp_types.hpp"

class Instance
{
   public:
    const NodeId num_aircrafts;
    const AircraftId num_nodes;
    const DictInt<NodeList> aircraft_nodes;
    const DictInt<double> node_cost;
    const DictInt<NodeSet> next_nodes;
    const DictInt<NodeSet> prev_nodes;
    const NodeList start_point;
    std::vector<NodeList> alt_nodes;

    DictInt<NodeId> node_to_aircraft{};
    DictInt<AircraftId> aircraft_to_source{};
    DictInt<AircraftId> aircraft_to_sink{};
    DictInt<NodeId> node_to_group{};
    DictInt<AircraftId> group_to_horizontal_aircraft;
    DictInt<DictInt<NodeId>> group_aircraft_to_vertical_node;

    GroupId num_group{};

    Instance(NodeId num_aircrafts,
             AircraftId num_nodes,
             DictInt<NodeList>&& aircraft_nodes,
             std::vector<NodeList>&& alt_nodes,
             DictInt<double>&& node_cost,
             DictInt<NodeSet>&& next_nodes,
             DictInt<NodeSet>&& prev_nodes,
             NodeList&& start_point) :
            num_aircrafts(num_aircrafts),
            num_nodes(num_nodes),
            aircraft_nodes(std::move(aircraft_nodes)),
            alt_nodes(std::move(alt_nodes)),
            node_cost(std::move(node_cost)),
            next_nodes(std::move(next_nodes)),
            prev_nodes(std::move(prev_nodes)),
            start_point(std::move(start_point))
    {
        initBaseFields_();
        initGroupVerticalNode_();
        initGroupHorizontalNode_();
        sortHorizontalGroups_();
    }

    bool isVerticalNode(NodeId v) const
    {
        auto group = node_to_group[v];
        auto aid = node_to_aircraft[v];
        // TODO
        return group_aircraft_to_vertical_node[group][aid] != ABSENT_NODE;
    }

   private:
    void initBaseFields_()
    {
        node_to_aircraft.resize(num_nodes);
        aircraft_to_source.resize(num_aircrafts);
        aircraft_to_sink.resize(num_aircrafts);

        for (AircraftId aircraft_id = 0; aircraft_id < num_aircrafts; ++aircraft_id)
        {
            NodeList sources, sinks;
            for (NodeId node_id : aircraft_nodes[aircraft_id])
            {
                node_to_aircraft[node_id] = aircraft_id;
                if (next_nodes[node_id].empty())
                    sinks.push_back(node_id);
                if (prev_nodes[node_id].empty())
                    sources.push_back(node_id);
            }
            if ((sources.size() != 1) || (sinks.size() != 1))
            {
                logger_->error("Aircraft {} must have exactly one source and one sink.", aircraft_id);
                throw std::runtime_error("Expected exactly one source and one sink.");
            }
            aircraft_to_source[aircraft_id] = sources[0];
            aircraft_to_sink[aircraft_id] = sinks[0];
        }

        num_group = alt_nodes.size();
        node_to_group.resize(num_nodes, -1);

        for (GroupId i = 0; i < num_group; ++i)
        {
            for (NodeId node : alt_nodes[i])
            {
                node_to_group[node] = i;
            }
        }
    }

    void initGroupVerticalNode_()
    {
        auto UNVIZITED_NODE = ABSENT_NODE - 1;
        group_aircraft_to_vertical_node =
            std::vector<DictInt<NodeId>>(num_group, DictInt<NodeId>(num_aircrafts, UNVIZITED_NODE));

        for (GroupId group = 0; group < num_group; ++group)
        {
            for (NodeId node : alt_nodes[group])
            {
                AircraftId aid = node_to_aircraft[node];
                if (group_aircraft_to_vertical_node[group][aid] == UNVIZITED_NODE)
                {
                    group_aircraft_to_vertical_node[group][aid] = node;
                }
                else
                {
                    group_aircraft_to_vertical_node[group][aid] = ABSENT_NODE;
                }
            }
            for (NodeId& node : group_aircraft_to_vertical_node[group])
            {
                if (node == UNVIZITED_NODE)
                    node = ABSENT_NODE;
            }
        }
    }

    void initGroupHorizontalNode_()
    {
        auto UNVIZITED_NODE = ABSENT_AIRCRAFT - 1;
        group_to_horizontal_aircraft = DictInt<int>(num_group, UNVIZITED_NODE);

        for (GroupId group = 0; group < num_group; ++group)
        {
            for (NodeId node : alt_nodes[group])
            {
                AircraftId aid = node_to_aircraft[node];
                if (group_to_horizontal_aircraft[group] == UNVIZITED_NODE)
                {
                    group_to_horizontal_aircraft[group] = aid;
                }
                else if (group_to_horizontal_aircraft[group] != aid)
                {
                    group_to_horizontal_aircraft[group] = ABSENT_AIRCRAFT;
                }
            }
        }
    }

    void sortHorizontalGroups_()
    {
        for (GroupId i = 0; i < num_group; ++i)
        {
            if (group_to_horizontal_aircraft[i] != ABSENT_AIRCRAFT)
                std::sort(alt_nodes[i].begin(),
                          alt_nodes[i].end(),
                          [this](NodeId g1, NodeId g2) { return node_cost[g1] < node_cost[g2]; });
        }
    }
};
