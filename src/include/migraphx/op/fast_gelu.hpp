#ifndef MIGRAPHX_GUARD_OPERATORS_FAST_GELU_HPP
#define MIGRAPHX_GUARD_OPERATORS_FAST_GELU_HPP

#include <array>
#include <migraphx/op/unary.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct fast_gelu : unary<fast_gelu>
{
    auto apply() const
    {
        return
            [](auto x) { return 0.5 * x * (1 + tanh(sqrt(M_2_PI) * (x + 0.044715 * x * x * x))); };
    }

    /* auto apply() const
    {
        return [](auto x) { return x * 0.5 * (1 + ::erf(x * M_SQRT1_2)); };
    } */
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
