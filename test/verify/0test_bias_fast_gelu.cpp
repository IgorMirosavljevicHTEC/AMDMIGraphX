#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_biasfastgelu : verify_program<test_biasfastgelu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<std::size_t> dims{1, 384, 3072};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::half_type, dims});
        auto b = mm->add_parameter("b", migraphx::shape{migraphx::shape::half_type, {dims.back()}});
        mm->add_instruction(migraphx::make_op("bias_fast_gelu"), x, b);

        return p;
    }
};
