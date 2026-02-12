#pragma once

#include <algorithm>
#include <cstddef>
#include <set>
#include <unordered_map>

#include "core/fdp_types.hpp"

class ConflictManager
{
   public:
    ConflictManager() = default;
    explicit ConflictManager(size_t capacity) : max_size(capacity) {}

    void clear()
    {
        cache.clear();
        num_cuts = 0;
        epoch = 0;
    }

    void setup() { clear(); }

    bool contains(const Conflict& conflict)
    {
        for (auto& [_, data] : cache)
        {
            if (isSubset(data.conflict, conflict))
            {
                data.usage_count++;
                data.last_used = epoch++;
                num_cuts++;
                return true;
            }
        }
        return false;
    }

    void add(const Conflict& conflict)
    {
        size_t usage = 1;
        for (auto it = cache.begin(); it != cache.end();)
        {
            if (isSubset(conflict, it->second.conflict))
            {
                total_conflict_size -= it->second.conflict.size();
                usage += it->second.usage_count;
                it = cache.erase(it);
            }
            else if (isSubset(it->second.conflict, conflict))
            {
                return;
            }
            else
            {
                ++it;
            }
        }

        if (cache.size() >= max_size)
        {
            evictLeastUsed();
        }

        size_t h = hashConflict(conflict);
        cache[h] = ConflictData{conflict, usage, epoch++};
        total_conflict_size += conflict.size();
    }

    size_t size() const { return cache.size(); }
    size_t numCuts() const { return num_cuts; }

    double averageConflictSize() const
    {
        if (cache.empty())
            return 0.0;
        return double(100 * total_conflict_size / cache.size()) / 100;
    }

   private:
    struct ConflictData
    {
        Conflict conflict;
        size_t usage_count;
        size_t last_used;
    };

    size_t max_size{};
    std::unordered_map<size_t, ConflictData> cache{};
    size_t num_cuts{};
    size_t epoch{0};

    size_t total_conflict_size{0};

    bool isSubset(const Conflict& a, const Conflict& b) const
    {
        return std::includes(b.begin(), b.end(), a.begin(), a.end());
    }

    size_t hashConflict(const Conflict& c) const
    {
        size_t seed = c.size();
        for (NodeId id : c)
        {
            seed ^= std::hash<NodeId>{}(id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }

    void evictLeastUsed()
    {
        double min_score = 1e9;
        size_t best_h = 0;

        for (auto& [h, data] : cache)
        {
            double score = data.usage_count / double(epoch - data.last_used + 1);
            if (score < min_score)
            {
                min_score = score;
                best_h = h;
            }
        }

        total_conflict_size -= cache[best_h].conflict.size();
        cache.erase(best_h);
    }
};
