#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.h>
#include <migraphx/verify.hpp>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

#define MIGRAPHX_HIP_ASSERT(x) (EXPECT((x) == hipSuccess))
struct simple_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "simple_custom_op"; }
    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override
    {
        // sets first half size_bytes of the input 0, and rest of the half bytes are copied.
        float* d_output;
        auto* h_output   = reinterpret_cast<float*>(inputs[0].data());
        auto input_bytes = inputs[0].get_shape().bytes();
        auto copy_bytes  = input_bytes / 2;
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        MIGRAPHX_HIP_ASSERT(hipMalloc(&d_output, input_bytes));
        MIGRAPHX_HIP_ASSERT(hipMemcpyAsync(
            d_output, h_output, input_bytes, hipMemcpyHostToDevice, ctx.get_queue<hipStream_t>()));
        MIGRAPHX_HIP_ASSERT(hipMemset(d_output, 2, copy_bytes));
        MIGRAPHX_HIP_ASSERT(hipMemcpy(h_output, d_output, input_bytes, hipMemcpyDeviceToHost));
        MIGRAPHX_HIP_ASSERT(hipFree(d_output));
        return inputs[0];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        CHECK(inputs.size() == 1);
        return inputs.back();
    }

    virtual bool runs_on_offload_target() const override { return true; }
};

TEST_CASE(run_simple_custom_op)
{
    simple_custom_op simple_op;
    migraphx::register_experimental_custom_op(simple_op);
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {4, 3}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto relu          = m.add_instruction(migraphx::operation("relu"), {x});
    auto custom_kernel = m.add_instruction(migraphx::operation("simple_custom_op"), {relu});
    auto neg           = m.add_instruction(migraphx::operation("neg"), {custom_kernel});
    m.add_return({neg});
    migraphx::compile_options options;
    options.set_offload_copy(true);
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    std::vector<float> x_data(12, 1);
    std::vector<float> ret_data(12, -1);
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto result = p.eval(pp)[0];
    std::vector<float> expected_result(12, 0);
    std::fill(expected_result.begin() + 6, expected_result.end(), -1);
    auto result_vec = result.as_vector<float>();
    EXPECT(migraphx::verify_range(result_vec, expected_result));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
