#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

using AircraftId = int;
using NodeId = int;
using GroupId = int;
using NodeList = std::vector<NodeId>;
using NodeSet = std::unordered_set<NodeId>;
using GroupList = std::vector<GroupId>;
using GroupSet = std::unordered_set<GroupId>;
using AircraftList = std::vector<AircraftId>;
using Score = double;

constexpr int ABSENT_NODE = -1;
constexpr int ABSENT_INT = -1;
constexpr int ABSENT_GROUP = -1;
constexpr int ABSENT_AIRCRAFT = -1;

template <typename K, typename V>
using Dict = std::unordered_map<K, V>;

template <typename V>
using DictInt = std::vector<V>;

using Conflict = std::set<NodeId>;
