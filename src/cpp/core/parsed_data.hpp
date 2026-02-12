#pragma once
#include <utility>
#include <vector>

#include "core/fdp_types.hpp"
#include "core/instance.hpp"

class FlightDistributionData
{
   public:
    DictInt<NodeList> aircraft_nodes;
    std::vector<NodeList> alt_nodes;
    DictInt<double> node_cost;
    std::vector<std::pair<int, int>> edges;

    NodeList start_point;

    Instance getInstance()
    {
        DictInt<NodeSet> next_edge_map(node_cost.size()), prev_edge_map(node_cost.size());
        for (auto const& [from, to] : edges)
        {
            next_edge_map[from].emplace(to);
            prev_edge_map[to].emplace(from);
        }

        int num_aircrafts = static_cast<int>(aircraft_nodes.size());
        int num_nodes = static_cast<int>(node_cost.size());

        return Instance(num_aircrafts,
                        num_nodes,
                        std::move(aircraft_nodes),
                        std::move(alt_nodes),
                        std::move(node_cost),
                        std::move(next_edge_map),
                        std::move(prev_edge_map),
                        std::move(start_point));
    }
};
