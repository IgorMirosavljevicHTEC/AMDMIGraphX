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
        if(args.size() == 2)
            return info.add_instruction(migraphx::make_op("bias_fast_gelu"), args);
        else
        {
            if(args[0]->name() == "add")
            {
                auto new_args = args[0]->inputs();
                //std::reverse(new_args.begin(), new_args.end());
                auto temp = new_args[0];
                new_args[0] = new_args[1];
                new_args[1] = temp->inputs().front();
                return info.add_instruction(migraphx::make_op("bias_fast_gelu"), new_args);
            }

            return info.add_instruction(migraphx::make_op("fast_gelu"), args);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
