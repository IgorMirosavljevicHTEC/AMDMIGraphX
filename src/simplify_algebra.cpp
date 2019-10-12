#include <migraphx/simplify_algebra.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/rsqrt.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

auto lit_broadcast() { return match::any_of(match::is_constant(), match::name("broadcast")); }
auto not_lit_broadcast() { return match::none_of(match::is_constant(), match::name("broadcast")); }
auto op_lit_broadcast(std::string op, std::string x, std::string y)
{
    return match::name(std::move(op))(match::either_arg(0, 1)(
        lit_broadcast().bind(std::move(x)), not_lit_broadcast().bind(std::move(y))));
}

auto conv_const_weights()
{
    return match::name("convolution")(match::used_once(),
                                      match::args(match::any(), match::is_constant().bind("w")));
}

struct find_mul_conv
{
    auto matcher() const
    {
        return match::name("mul")(match::either_arg(0, 1)(conv_const_weights().bind("conv"),
                                                          match::name("broadcast").bind("a")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins      = r.result;
        auto conv_ins = r.instructions["conv"];
        auto a_ins    = r.instructions["a"];
        auto w_ins    = r.instructions["w"];

        auto broadcast_op = any_cast<op::broadcast>(a_ins->get_operator());
        if(broadcast_op.axis != 1)
            return;

        auto new_a = p.insert_instruction(
            ins, op::broadcast{0, w_ins->get_shape().lens()}, a_ins->inputs().front());
        auto new_mul  = p.insert_instruction(ins, op::mul{}, new_a, w_ins);
        auto new_conv = p.insert_instruction(
            ins, conv_ins->get_operator(), conv_ins->inputs().front(), new_mul);
        p.replace_instruction(ins, new_conv);
    }
};

// a * (x + b) => a * x + a * b
struct find_mul_add
{
    auto matcher() const
    {
        return match::name("mul")(match::either_arg(0, 1)(
            match::name("add")(
                match::either_arg(0, 1)(
                    match::any().bind("x"),
                    match::any_of(conv_const_weights(), match::is_constant()).bind("b")),
                match::none_of(match::args(match::is_constant(), match::is_constant())),
                match::used_once()),
            match::is_constant().bind("a")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto a_ins = r.instructions["a"];
        auto b_ins = r.instructions["b"];
        auto x_ins = r.instructions["x"];
        assert(x_ins != b_ins);

        auto ax_ins = p.insert_instruction(ins, op::mul{}, a_ins, x_ins);
        auto ab_ins = p.insert_instruction(ins, op::mul{}, a_ins, b_ins);
        p.replace_instruction(ins, op::add{}, ax_ins, ab_ins);
    }
};

struct find_add_lit_broadcast
{
    auto matcher() const
    {
        return match::name("add")(
            match::either_arg(0, 1)(op_lit_broadcast("add", "a", "x"), lit_broadcast().bind("b")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        auto a_ins = r.instructions["a"];
        auto b_ins = r.instructions["b"];

        auto sumab = p.insert_instruction(ins, op::add{}, a_ins, b_ins);
        p.replace_instruction(ins, op::add{}, x_ins, sumab);
    }
};

struct find_double_add_lit_broadcast
{
    auto matcher() const
    {
        return match::name("add")(
            match::args(op_lit_broadcast("add", "a", "x"), op_lit_broadcast("add", "b", "y")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        auto y_ins = r.instructions["y"];
        auto a_ins = r.instructions["a"];
        auto b_ins = r.instructions["b"];

        instruction_ref sumab;

        if(a_ins->name() == "broadcast" and b_ins->name() == "broadcast")
        {
            if(a_ins->inputs().at(0)->get_shape() != b_ins->inputs().at(0)->get_shape())
                return;
            auto op = a_ins->get_operator();
            auto presum =
                p.insert_instruction(ins, op::add{}, a_ins->inputs().at(0), b_ins->inputs().at(0));
            sumab = p.insert_instruction(ins, op, presum);
        }
        else
        {
            sumab = p.insert_instruction(ins, op::add{}, a_ins, b_ins);
        }

        auto sumxy = p.insert_instruction(ins, op::add{}, x_ins, y_ins);
        p.replace_instruction(ins, op::add{}, sumxy, sumab);
    }
};

struct find_inner_broadcast
{
    auto matcher() const
    {
        return match::name("mul", "add")(
            match::args(match::name("broadcast").bind("x"), match::name("broadcast").bind("y")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        auto y_ins = r.instructions["y"];

        auto xbroadcast = any_cast<op::broadcast>(x_ins->get_operator());
        auto ybroadcast = any_cast<op::broadcast>(y_ins->get_operator());

        if(xbroadcast.axis != ybroadcast.axis)
            return;

        auto op = p.insert_instruction(
            ins, ins->get_operator(), x_ins->inputs().front(), y_ins->inputs().front());
        p.replace_instruction(ins, xbroadcast, op);
    }
};

// a / sqrt(b) => a * rsqrt(b)
struct find_div_sqrt
{
    auto matcher() const
    {
        return match::name("div")(match::used_once(), 
            match::args(match::any().bind("a"),
                        match::name("multibroadcast")(
                        match::args(match::name("sqrt")(match::used_once()).bind("b"))).bind("bcst")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto a_ins = r.instructions["a"];
        auto b_ins = r.instructions["b"];
        auto bcst_ins = r.instructions["bcst"];

        p.replace_instruction(b_ins, op::rsqrt{}, b_ins->inputs());
        p.replace_instruction(ins, op::mul{}, a_ins, bcst_ins);
    }
};

// gemm[alpha = 1, b = 0] / c  ==> gemm[alpha = 1/c, b = 0]
struct find_div_gemm
{
    auto matcher() const
    {
        return match::name("div")(match::used_once(), 
            match::args(match::name("dot").bind("a"),
                        match::name("multibroadcast")(
                        match::args(match::is_constant().bind("alpha"))).bind("bcst")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto dot_ins = r.instructions["a"];
        auto one_alpha = r.instructions["alpha"];

        // ensure alpha is a scalar
        if (one_alpha->get_shape().elements() != 1)
            return;

        float scale = one_alpha->get_literal().at<float>();
        auto dot_op = any_cast<op::dot>(dot_ins->get_operator());
        auto r_dot = p.replace_instruction(dot_ins, op::dot{dot_op.alpha/scale, dot_op.beta/scale}, dot_ins->inputs());

        p.replace_instruction(ins, r_dot);
    }
};


void simplify_algebra::apply(program& p) const
{
    // Run simplifications multiple times
    for(int i = 0; i < 4; i++)
    {
        match::find_matches(p,
                            find_div_sqrt{},
                            find_div_gemm{},
                            find_inner_broadcast{},
                            find_double_add_lit_broadcast{},
                            find_add_lit_broadcast{},
                            find_mul_conv{},
                            find_mul_add{});
        dead_code_elimination{}.apply(p);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
