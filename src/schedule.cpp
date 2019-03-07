#include <migraphx/schedule.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <deque>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

auto get_inputs()
{
    return [](auto i) { return i->inputs(); };
}

auto get_outputs()
{
    return [](auto i) { return i->outputs(); };
}

struct stream_info
{
    std::unordered_map<instruction_ref, std::size_t> ins2stream;
    std::unordered_map<instruction_ref, std::size_t> weights;
    std::unordered_map<instruction_ref, std::size_t> iweights;

    void accumulate_weights(instruction_ref last, const schedule_model& model)
    {
        fix<std::size_t>([&](auto self, auto ins) -> std::size_t {
            if(weights.count(ins) == 0)
            {
                std::size_t weight = 0;
                auto&& op          = ins->get_operator();
                if(not is_context_free(op) and op.name()[0] != '@')
                    weight = model.weight(op);
                iweights[ins] = weight;
                weights[ins] =
                    std::accumulate(ins->inputs().begin(),
                                    ins->inputs().end(),
                                    weight,
                                    [&](std::size_t w, instruction_ref i) { return w + self(i); });
            }
            return weights[ins];
        })(last);
    }

    struct partition
    {
        std::size_t weight = 0;
        std::vector<instruction_ref> instructions{};

        void add(instruction_ref ins, std::size_t w)
        {
            weight += w;
            instructions.push_back(ins);
        }
    };

    void assign_streams(program& p, std::size_t n)
    {
        const std::size_t min_partition_threshold = 2;
        partition critical;
        std::unordered_map<instruction_ref, std::deque<partition>> partitions;
        fix([&](auto self, auto ins, auto& part) {
            // If weight is zero then stop
            if(this->weights[ins] == 0)
                return;
            part.add(ins, this->iweights[ins]);

            auto max_it = std::max_element(ins->inputs().begin(),
                                           ins->inputs().end(),
                                           by(std::less<>{}, index_of(this->weights)));
            for(auto i : ins->inputs())
            {
                const auto weight = this->weights[i];
                if(i == *max_it or weight <= min_partition_threshold)
                {
                    self(i, part);
                }
                else
                {
                    partitions[ins].emplace_back();
                    self(i, partitions[ins].back());
                }
            }
        })(std::prev(p.end()), critical);

        // Set the critical partition to stream 0
        set_stream(critical, 0);
        std::vector<std::size_t> streams(n - 1);
        // Assign streams for the other partitions
        for(auto&& ins_part : partitions)
        {
            std::sort(
                ins_part.second.begin(), ins_part.second.end(), by(std::greater<>{}, [](auto&& x) {
                    return std::make_tuple(x.weight, x.instructions.size());
                }));
            for(auto&& part : ins_part.second)
            {
                auto stream = std::min_element(streams.begin(), streams.end()) - streams.begin();
                set_stream(part, stream + 1);
                streams[stream] += part.weight;
            }
        }
    }

    void set_stream(const partition& p, std::size_t n)
    {
        for(auto ins : p.instructions)
            if(iweights[ins] > 0)
                set_stream(ins, n);
    }

    void set_stream(instruction_ref ins, std::size_t n)
    {
        assert(iweights[ins] > 0);
        ins2stream[ins] = n;
    }

    std::size_t get_stream(instruction_ref ins) const { return ins2stream.at(ins); }

    bool has_stream(instruction_ref ins) const { return ins2stream.count(ins) > 0; }

    bool different(const std::vector<std::size_t>& v) const
    {
        if(v.size() < 2)
            return false;
        return not std::all_of(v.begin(), v.end(), [&](std::size_t x) { return x == v.front(); });
    }

    template <class F>
    bool different(F f, std::size_t stream) const
    {
        bool result = false;
        f([&](auto s) {
            if(s != stream)
            {
                result = true;
                return false;
            }
            stream = s;
            return true;
        });
        return result;
    }

    template <class F>
    bool different(F f) const
    {
        bool result = false;
        f([&](auto s) {
            result = different(f, s);
            return false;
        });
        return result;
    }

    template <class Selector>
    auto get_streams(instruction_ref start, Selector select) const
    {
        return [=](auto f) {
            return fix<bool>([&](auto self, auto ins) {
                for(auto i : select(ins))
                {
                    if(iweights.at(i) == 0)
                    {
                        if(not self(i))
                            return false;
                    }
                    else
                    {
                        if(not f(get_stream(i)))
                            return false;
                    }
                }
                return true;
            })(start);
        };
    }

    template <class... Ts>
    bool is_merge_point(instruction_ref ins, Ts... xs) const
    {
        return different(get_streams(ins, get_inputs()), xs...);
    }

    template <class... Ts>
    bool is_split_point(instruction_ref ins, Ts... xs) const
    {
        return different(get_streams(ins, get_outputs()), xs...);
    }

    std::vector<instruction_ref> get_recorded_instructions(instruction_ref start)
    {
        std::vector<instruction_ref> result;
        std::unordered_map<std::size_t, instruction_ref> m;
        fix([&](auto self, auto ins) {
            for(auto i : ins->inputs())
            {
                if(iweights.at(i) == 0)
                {
                    self(i);
                    continue;
                }
                auto stream = get_stream(i);
                if(m.count(stream) == 0)
                    m[stream] = i;
                else
                    m[stream] = std::min(m[stream], i, by(std::less<>{}, [&](auto x) {
                                             return std::distance(x, start);
                                         }));
            }
        })(start);
        std::transform(
            m.begin(), m.end(), std::back_inserter(result), [](auto&& p) { return p.second; });
        return result;
    }

    std::vector<std::size_t> wait_for(instruction_ref ins) const
    {
        std::vector<std::size_t> result;
        get_streams(ins, get_inputs())([&](auto s) {
            result.push_back(s);
            return true;
        });
        // Remove duplicates
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        // Remove the merged stream
        auto it = std::find(result.begin(), result.end(), get_stream(ins));
        if(it != result.end())
            result.erase(it);
        return result;
    }

    std::unordered_map<instruction_ref, std::vector<std::vector<instruction_ref>>>
    find_concurrent_instructions(program& p)
    {
        std::unordered_map<instruction_ref, std::vector<std::vector<instruction_ref>>> result;
        std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>> split_from;
        for(auto ins : iterator_for(p))
        {
            if(iweights[ins] == 0)
                continue;
            for(auto&& arg : ins->inputs())
            {
                if(is_split_point(arg))
                    split_from[ins].insert(arg);
                split_from[ins].insert(split_from[arg].begin(), split_from[arg].end());
            }

            auto stream = get_stream(ins);
            // if (is_merge_point(ins))
            // {
            //     // post-dominator kills split point.
            //     for(auto& split : split_from[ins])
            //     {
            //         if(strictly_post_dominates(ins, split))
            //             split_from[ins].erase(split);
            //     }
            // }

            // Collect concur instructions for each split point.
            for(auto& split : split_from[ins])
            {
                if(result[split].size() <= stream)
                    result[split].resize(stream + 1);
                result[split][stream].push_back(ins);
            }
        }
        return result;
    }
};

void schedule::apply(program& p) const
{
    stream_info si;
    auto last = std::prev(p.end());
    si.accumulate_weights(last, model);
    si.assign_streams(p, model.concurrency());

    // Topo sort
    fix([&](auto self, auto ins) {
        auto args = ins->inputs();
        std::sort(args.begin(), args.end(), by(std::less<>{}, [&](auto x) {
                      return std::make_tuple(si.weights[x], x->inputs().size());
                  }));
        for(auto i : args)
        {
            p.move_instruction(i, p.begin());
            self(i);
        }
    })(last);

    if(enabled(MIGRAPHX_TRACE_COMPILE{}))
    {
        p.annotate(std::cout, [&](auto ins) {
            std::cout << ":";
            std::cout << " weight=" << si.weights.at(ins);
            std::cout << " input={";
            si.get_streams(ins, get_inputs())([&](auto s) {
                std::cout << s << ",";
                return true;
            });
            std::cout << "}";
            if(si.has_stream(ins))
                std::cout << " stream=" << si.get_stream(ins);
        });
        std::cout << std::endl;
    }

    // Schedule instructions
    std::unordered_map<instruction_ref, std::size_t> ins2wait;
    std::size_t wait_id = 0;
    for(auto ins : iterator_for(p))
    {
        // Only schedule instructions that have a stream
        if(not si.has_stream(ins))
            continue;
        assert(si.weights[ins] > 0);
        // Schedule instruction on the stream
        auto stream = si.get_stream(ins);
        assert(stream < model.concurrency());
        model.sched(p, ins, stream);
        // Insert wait instructions
        if(si.is_merge_point(ins, stream))
        {
            for(auto i : si.get_recorded_instructions(ins))
            {
                if(not si.has_stream(i))
                    continue;
                if(stream == si.get_stream(i))
                    continue;
                // Create a new event if it hasn't been recorded
                if(ins2wait.count(i) == 0)
                {
                    ins2wait[i] = wait_id;
                    model.record(p, i, wait_id);
                    wait_id++;
                }
                model.wait(p, ins, ins2wait.at(i));
            }
        }
    }

    // Add memory conflicts
    auto concur_ins = si.find_concurrent_instructions(p);
    for(auto&& split : concur_ins)
    {
        dfor(split.second.size(), split.second.size())([&](auto i, auto j) {
            if(i == j)
                return;
            for(auto ins1 : split.second[i])
            {
                auto args = split.second[j];
                args.insert(args.begin(), ins1);

                auto point = std::max_element(args.begin(), args.end(), [&](auto x, auto y) {
                    return std::distance(split.first, x) < std::distance(split.first, y);
                });
                p.insert_instruction(std::next(*point), op::identity{}, args);
            }
        });
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
