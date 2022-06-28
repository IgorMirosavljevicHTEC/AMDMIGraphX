#include <migraphx/rewrite_fastgelu.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/bias_fast_gelu.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/match/gelu_erf.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/dead_code_elimination.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct find_add_gelu_erf
{
    static auto match_mul1()
    {
        return match::name("mul")(
            match::args(match::name("add").bind("add"), match::is_constant()));
    }

    static auto match_erf() { return match::name("erf")(match::arg(0)(match_mul1())); }

    static auto match_add2()
    {
        return match::name("add")(match::args(match_erf(), match::has_value(1.0f)));
    }

    static auto match_add1()
    {
        return match::name("add")(match::args(match::name("dot"), match::is_constant()));
    }

    static auto match_mul2() { return match::name("mul")(match::args(match_add1(), match_add2())); }

    auto matcher() const
    {
        return match::name("mul")(match::args(match_mul2(), match::has_value(0.5f)));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto add_ins = r.instructions["add"];
        auto args    = add_ins->inputs();

        m.replace_instruction(ins, make_op("bias_fast_gelu"), args);
    }
};

void rewrite_fastgelu::apply(module& m) const
{
    match::find_matches(m, find_add_gelu_erf{});
    dead_code_elimination{}.apply(m);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
