#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_horizdots : op_parser<parse_horizdots>
{
    std::vector<op_desc> operators() const { return {{"HorizDots"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto s = shape{shape::half_type, {768, 768}};
        std::vector<size_t> bcl{2, 768, 768};
        std::vector<float> ones(768 * 768, 1);
        std::vector<float> twos(768 * 768, 2);
        std::vector<float> threes(768 * 768, 3);
        std::vector<float> fours(768, 4);
        auto l1 = info.add_literal(literal{s, ones});
        auto l2 = info.add_literal(literal{s, twos});
        auto l3 = info.add_literal(literal{s, threes});
        auto l1b = info.add_instruction(make_op("multibroadcast", {{"out_lens", bcl}}), l1);
        auto l2b = info.add_instruction(make_op("multibroadcast", {{"out_lens", bcl}}), l2);
        auto l3b = info.add_instruction(make_op("multibroadcast", {{"out_lens", bcl}}), l3);

        auto l4 = info.add_literal(literal{shape{shape::half_type, {768}}, fours});

        auto add = info.add_broadcastable_binary_op("add", args[0], args[1]);
        auto mm1 = info.add_instruction(make_op("dot"), add, l1b);
        auto mm2 = info.add_instruction(make_op("dot"), add, l2b);
        auto mm3 = info.add_instruction(make_op("dot"), add, l3b);

        auto add1 = info.add_broadcastable_binary_op("add", l4, mm1);
        auto add2 = info.add_broadcastable_binary_op("add", l4, mm2);
        auto add3 = info.add_broadcastable_binary_op("add", l4, mm3);

        auto mul1 = info.add_broadcastable_binary_op("mul", add1, add2);
        return info.add_broadcastable_binary_op("mul", mul1, add3);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
