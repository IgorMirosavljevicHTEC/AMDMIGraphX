#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_fastgelu : op_parser<parse_fastgelu>
{
    std::vector<op_desc> operators() const { return {{"FastGelu"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        if (args.size() == 2)
            return info.add_instruction(migraphx::make_op("bias_fast_gelu"), args[0], args[1]);
        else
            return info.add_instruction(migraphx::make_op("fast_gelu"), args[0]);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
