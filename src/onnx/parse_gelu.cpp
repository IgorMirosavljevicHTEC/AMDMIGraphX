#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_gelu : op_parser<parse_gelu>
{
    std::vector<op_desc> operators() const { return {{"Gelu"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto input = args.front();
        if (args.size() == 2)
        {
            auto bias = args[1];
            input = info.add_broadcastable_binary_op("add", input, bias);
        }

        auto sqrt2 = info.add_literal(literal{shape{input->get_shape().type(), {1}}, {1.4140625}});
        auto div = info.add_broadcastable_binary_op("div", input, sqrt2);
        auto erf = info.add_instruction(make_op("erf"), div);
        auto l1  = info.add_literal(literal{shape{input->get_shape().type(), {1}}, {1.0}});
        auto add = info.add_broadcastable_binary_op("add", erf, l1);
        input = info.add_broadcastable_binary_op("mul", input, add);
        auto l2  = info.add_literal(literal{shape{input->get_shape().type(), {1}}, {0.5}});
        return info.add_broadcastable_binary_op("mul", input, l2);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
