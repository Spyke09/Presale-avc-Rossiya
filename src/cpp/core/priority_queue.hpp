#include <functional>
#include <queue>

#include "core/solution_state.hpp"

class SwitchablePriorityQueue
{
    std::function<bool(const SolutionState&, const SolutionState&)> comp;
    std::priority_queue<SolutionState, std::vector<SolutionState>, decltype(comp)> pq;

   public:
    SwitchablePriorityQueue(bool use_fast = false) : comp(use_fast ? fast_comp : slow_comp), pq(comp) {}

    void switch_comp(bool use_fast)
    {
        comp = use_fast ? fast_comp : slow_comp;
        rebuild_queue();
    }

    void push(SolutionState s) { pq.push(std::move(s)); }

    template <typename... Args>
    void emplace(Args&&... args)
    {
        pq.emplace(std::forward<Args>(args)...);
    }

    void pop() { pq.pop(); }

    const SolutionState& top() const { return pq.top(); }

    bool empty() const { return pq.empty(); }

    size_t size() const { return pq.size(); }

   private:
    static bool slow_comp(const SolutionState& t1, const SolutionState& t2)
    {
        bool me1 = (t1.depth.size() - t1.taken_groups > t2.depth.size() - t2.taken_groups);
        bool me2 = (t1.taken_groups < t2.taken_groups);
        bool me3 = (t1.depth.size() > t2.depth.size());
        bool me4 = (t1.score_untaken_groups > t2.score_untaken_groups);
        bool me5 = (t1.score_objective > t2.score_objective);
        bool e1 = (t1.depth.size() - t1.taken_groups == t2.depth.size() - t2.taken_groups);
        bool e2 = (t1.taken_groups == t2.taken_groups);
        bool e3 = (t1.depth == t2.depth);
        bool e4 = (t1.score_untaken_groups == t2.score_untaken_groups);
        return (me1) || (e1 && me2) || (e1 && e2 && me3) || (e1 && e2 && e3 && me4) || (e1 && e2 && e3 && e4 && me5);
    }

    static bool fast_comp(const SolutionState& t1, const SolutionState& t2)
    {
        bool me1 = (t1.score_untaken_groups > t2.score_untaken_groups);
        bool me2 = (t1.score_objective > t2.score_objective);
        bool e1 = (t1.score_untaken_groups == t2.score_untaken_groups);
        return (me1) || (e1 && me2);
    }

    void rebuild_queue()
    {
        std::priority_queue<SolutionState, std::vector<SolutionState>, decltype(comp)> new_pq(comp);
        while (!pq.empty())
        {
            new_pq.push(pq.top());
            pq.pop();
        }
        pq = std::move(new_pq);
    }
};
