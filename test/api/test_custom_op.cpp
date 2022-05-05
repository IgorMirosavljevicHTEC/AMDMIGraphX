#include <algorithm>
#include <cmath>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

struct sigmoid_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "sigmoid_custom_op"; }
    virtual migraphx::argument compute(migraphx::context,
                                       migraphx::shape output_shape,
                                       migraphx::arguments inputs) const override
    {
        size_t input_len = inputs[0].get_shape().lengths()[0];
        std::vector<float> output(input_len);
        auto input_vec = inputs[0].as_vector<float>();
        std::transform(input_vec.begin(), input_vec.end(), output.begin(), [](auto x) {
            return 1.f / (1.f + std::exp(-x));
        });
        auto ret = migraphx::argument(output_shape, output.data());
        return ret;
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        CHECK(inputs.size() == 1);
        CHECK(inputs[0].lengths().size() == 1);
        CHECK(inputs[0].type() == migraphx_shape_float_type);
        return inputs.front();
    }
};

TEST_CASE(register_custom_op)
{
    sigmoid_custom_op sigmoid_op;
    migraphx::register_experimental_custom_op(sigmoid_op);
    auto op = migraphx::operation("sigmoid_custom_op");
    EXPECT(op.name() == "sigmoid_custom_op");
}

TEST_CASE(run_sigmoid_custom_op)
{
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {12}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto custom_kernel = m.add_instruction(migraphx::operation("sigmoid_custom_op"), {x});
    p.compile(migraphx::target("ref"));
    // run program
    migraphx::program_parameters pp;
    auto param_shapes            = p.get_parameter_shapes();
    migraphx::argument input_arg = migraphx::argument::generate(param_shapes["x"]);
    pp.add("x", input_arg);
    auto results   = p.eval(pp);
    auto result    = results[0];
    auto input_vec = input_arg.as_vector<float>();
    std::transform(input_vec.begin(), input_vec.end(), input_vec.begin(), [](auto y) {
        return 1.f / (1.f + std::exp(-y));
    });
    EXPECT(bool{result == migraphx::argument(s, input_vec.data())});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
