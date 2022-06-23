#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_gelutanh : op_parser<parse_gelutanh>
{
    std::vector<op_desc> operators() const { return {{"GeluTanh"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto input = args.front();
        if(args.size() == 2)
        {
            auto bias = args[1];
            input     = info.add_broadcastable_binary_op("add", input, bias);
        }

        auto l0 = info.add_literal(literal{shape{input->get_shape().type(), {1}}, {2.0}});
        auto l1 = info.add_literal(literal{shape{input->get_shape().type(), {1}}, {1.0}});
        auto l2 = info.add_literal(literal{shape{input->get_shape().type(), {1}}, {0.5}});
        auto l3 = info.add_literal(
            literal{shape{input->get_shape().type(), {1}}, {0.035677408136300125}});
        auto l4 = info.add_literal(
            literal{shape{input->get_shape().type(), {1}}, {0.79788456080286535588}});

        auto u   = info.add_broadcastable_binary_op("mul", input, input);
        u        = info.add_broadcastable_binary_op("mul", u, l3);
        u        = info.add_broadcastable_binary_op("add", u, l4);
        u        = info.add_broadcastable_binary_op("mul", u, input);
        u        = info.add_broadcastable_binary_op("mul", u, l0);
        u        = info.add_instruction(make_op("neg"), u);
        auto emu = info.add_instruction(make_op("exp"), u);
        auto cdf = info.add_broadcastable_binary_op("add", emu, l1);
        cdf      = info.add_broadcastable_binary_op("div", l0, cdf);
        cdf      = info.add_broadcastable_binary_op("sub", cdf, l1);
        cdf      = info.add_broadcastable_binary_op("mul", cdf, l2);
        cdf      = info.add_broadcastable_binary_op("add", cdf, l2);

        return info.add_broadcastable_binary_op("mul", input, cdf);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
