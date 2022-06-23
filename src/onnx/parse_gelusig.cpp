#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_gelusig : op_parser<parse_gelusig>
{
    std::vector<op_desc> operators() const { return {{"GeluSig"}}; }

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

        auto l1    = info.add_literal(literal{shape{input->get_shape().type(), {1}}, {1.702f}});
        auto inner   = info.add_broadcastable_binary_op("mul", input, l1);
        inner      = info.add_instruction(make_op("sigmoid"), inner);
        return info.add_broadcastable_binary_op("mul", input, inner);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
