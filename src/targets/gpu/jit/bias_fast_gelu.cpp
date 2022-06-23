#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <migraphx/cpp_generator.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(FASTGELU_ALGO2)

static const char* const bias_fast_gelu_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/bias_fast_gelu.hpp>
#include <args.hpp>
namespace migraphx {
extern "C" {
__global__ void bias_fast_gelu_kernel(void* input_p, void* bias_p, void* output_p) 
{
    make_tensors()(input_p, bias_p, output_p)([](auto input, auto bias, auto output) {
        bias_fast_gelu(input, bias, output);
    });
}
    
}
} // namespace migraphx
)__migraphx__";

static const char* const bias_fast_gelu_half2_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <migraphx/kernels/bias_fast_gelu.hpp>
#include <args.hpp>
namespace migraphx {
extern "C" {

__global__ void bias_fast_gelu_half2_kernel(void* input_p, void* bias_p, void* output_p) 
{
    auto settings = make_biasfastgelu_settings(MIGRAPHX_MAKE_CONSTANT(size_t{ELEMENTS}), MIGRAPHX_MAKE_CONSTANT(size_t{BIAS_DIM}));
    bias_fast_gelu_half2(input_p, bias_p, output_p, settings);
}
    
}
} // namespace migraphx
)__migraphx__";

static const char* const bias_fast_gelu_half4_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <migraphx/kernels/bias_fast_gelu.hpp>
#include <args.hpp>
namespace migraphx {
extern "C" {

__global__ void bias_fast_gelu_half4_kernel(void* input_p, void* bias_p, void* output_p) 
{
    auto settings = make_biasfastgelu_settings(MIGRAPHX_MAKE_CONSTANT(size_t{ELEMENTS}), MIGRAPHX_MAKE_CONSTANT(size_t{BIAS_DIM}));
    bias_fast_gelu_half4(input_p, bias_p, output_p, settings);
}
    
}
} // namespace migraphx
)__migraphx__";

struct bias_fast_gelu_compiler : compiler<bias_fast_gelu_compiler>
{
    std::vector<std::string> names() const { return {"bias_fast_gelu"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        std::size_t local = 1024;
        if(inputs.front().type() == migraphx::shape::half_type)
        {
            options.set_launch_params(
                v, compute_global_for(ctx, inputs.back().elements() / 4, 256), local);
            options.output = inputs.back();
            options.inputs = inputs;
            if(enabled(FASTGELU_ALGO2{}))
            {
                std::cout << "algo2" << std::endl;
                options.kernel_name = "bias_fast_gelu_half4_kernel";
            }
            else
            {
                std::cout << "algo1" << std::endl;
                options.kernel_name = "bias_fast_gelu_half2_kernel";
            }
            options.params += " -DELEMENTS=" + std::to_string(inputs.back().elements() / 4);
            options.params += " -DBIAS_DIM=" + std::to_string(inputs.at(1).elements() / 4);
            if(enabled(FASTGELU_ALGO2{}))
            {
                return compile_hip_code_object(bias_fast_gelu_half4_kernel, options);
            }
            else
            {
                return compile_hip_code_object(bias_fast_gelu_half2_kernel, options);
            }
        }
        options.set_launch_params(
            v, compute_global_for(ctx, inputs.back().elements(), local), local);
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "bias_fast_gelu_kernel";
        return compile_hip_code_object(bias_fast_gelu_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
