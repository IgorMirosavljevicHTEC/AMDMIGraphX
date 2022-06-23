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

MIGRAPHX_DECLARE_ENV_VAR(FASTGELU_ALGO)

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

static const char* const bias_fast_gelu_half2_tanh_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <migraphx/kernels/bias_fast_gelu.hpp>
#include <args.hpp>
namespace migraphx {
extern "C" {

__global__ void bias_fast_gelu_half2_tanh_kernel(void* input_p, void* bias_p, void* output_p) 
{
    auto settings = make_biasfastgelu_settings(MIGRAPHX_MAKE_CONSTANT(size_t{ELEMENTS}), MIGRAPHX_MAKE_CONSTANT(size_t{BIAS_DIM}));
    bias_fast_gelu_half2_tanh(input_p, bias_p, output_p, settings);
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
            auto algo_num = value_of(FASTGELU_ALGO{});
            auto vec_div  = algo_num == 1 ? 4 : 2;
            options.set_launch_params(
                v, compute_global_for(ctx, inputs.back().elements() / vec_div, 256), local);
            options.output      = inputs.back();
            options.inputs      = inputs;
            auto algo           = bias_fast_gelu_half2_kernel;
            options.kernel_name = "bias_fast_gelu_half2_kernel";
            if(algo_num == 1)
            {
                std::cout << "tanh4" << std::endl;
                options.kernel_name = "bias_fast_gelu_half4_kernel";
                algo                = bias_fast_gelu_half4_kernel;
            }
            else if(algo_num == 2)
            {
                std::cout << "tanh" << std::endl;
                options.kernel_name = "bias_fast_gelu_half4_kernel";
                algo                = bias_fast_gelu_half4_kernel;
            }
            else
            {
                std::cout << "sigmoid" << std::endl;
            }
            options.params += " -DELEMENTS=" + std::to_string(inputs.back().elements() / vec_div);
            options.params += " -DBIAS_DIM=" + std::to_string(inputs.at(1).elements() / vec_div);
            return compile_hip_code_object(algo, options);
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
