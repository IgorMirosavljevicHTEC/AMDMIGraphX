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

/* void move_standard_front(std::vector<instruction_ref>& args)
{
    // Ensure the first arguments is the standard one
    auto last = std::prev(args.end());
    auto it =
        std::find_if(args.begin(), last, [](auto arg) { return arg->get_shape().standard(); });
    if(it != last)
        std::swap(*it, args.front());
}

void move_broadcasted_back(std::vector<instruction_ref>& args)
{
    // Ensure the last arguments is the broadcasted one
    auto last = std::prev(args.end());
    auto it =
        std::find_if(args.begin(), last, [](auto arg) { return arg->get_shape().broadcasted(); });
    if(it != last)
        std::swap(*it, *std::prev(last));
}

struct find_add_gelu
{
    auto matcher() const
    {
        return match::gelu_erf()(match::arg(0)(match::name("gpu::add").bind("add")));
    }

    void apply(module& p, match::matcher_result r) const
    {
        auto add_ins = r.instructions["add"];
        auto ins     = r.result;
        auto args    = add_ins->inputs();
        move_standard_front(args);
        move_broadcasted_back(args);

        args.back() = ins->inputs().back();
        p.replace_instruction(ins, make_op("bias_fast_gelu"), args);
    }
}; */

struct find_add_gelu_erf
{
    static auto match_mul1()
    {
        return match::name("mul")(match::arg(0)(match::name("add").bind("add")),
                                  match::arg(1)(match::name("contiguous")));
    }

    /* static auto match_div()
    {
        return match::name("div")(
                    match::arg(0)(match::name("add").bind("add")),
                    match::arg(1)(match::skip_broadcasts(match::is_constant())));
    } */

    static auto match_erf() { return match::name("erf")(match::arg(0)(match_mul1())); }

    static auto match_add2()
    {
        return match::name("add")(match::arg(1)(match::name("contiguous")),
                                  match::arg(0)(match_erf()));
    }

    static auto match_add1()
    {
        return match::name("add")(match::args(match::name("dot"), match::name("@literal")));
    }

    static auto match_mul2() { return match::name("mul")(match::args(match_add1(), match_add2())); }

    auto matcher() const
    {
        // return match_add1();
        return match::name("mul")(match::args(match_mul2(), match::name("contiguous")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        //std::cout << "Match hit" << std::endl;
        auto ins     = r.result;
        auto add_ins = r.instructions["add"];
        auto args    = add_ins->inputs();

        m.replace_instruction(ins, make_op("bias_fast_gelu"), args);
    }
};

void rewrite_fastgelu::apply(module& m) const
{
    // m.debug_print();
    match::find_matches(m, find_add_gelu_erf{});
    dead_code_elimination{}.apply(m);
    //std::cout << "End matcher" << std::endl;
    /*  std::cout << "\n\nGraph at rewrite\n" << std::endl;
     m.debug_print();
     std::cout << "\n\nGraph at rewrite\n" << std::endl; */
    /* for (auto ins : iterator_for(m))
    {
        if (ins->name() == "")
            continue;
        if (ins->name() == "gpu::code_object")
        {
            auto op = ins->get_operator();
            auto val = op.to_value();
            if (val.contains("symbol_name"))
            {
                auto s = val.at("symbol_name");
                std::cout << s <<std::endl;
                value ss = "symbol_name: add_mul_erf_add_mul_mul_kernel";
                if (s == ss)
                    std::cout << "hit " << std::endl;
            }
        }
        //std::cout << "ins: " << ins->name() << std::endl;

    } */
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
