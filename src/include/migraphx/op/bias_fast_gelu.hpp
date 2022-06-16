/* #ifndef MIGRAPHX_GUARD_OPERATORS_BIAS_FAST_GELU_HPP
#define MIGRAPHX_GUARD_OPERATORS_BIAS_FAST_GELU_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct bias_fast_gelu : binary<bias_fast_gelu>
{
    auto apply() const
    {
        return [](auto x, auto y) { x += y; return 0.5 * x * (1 + tanh(sqrt(M_2_PI) * (x + 0.044715 * x * x * x))); };
    }

    auto apply() const
    {
        return [](auto x, auto y) { x += y; return (x * 0.5 * (1 + ::erf(x * M_SQRT1_2))) + y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
 */

#ifndef MIGRAPHX_GUARD_OPERATORS_BIAS_FAST_GELU_HPP
#define MIGRAPHX_GUARD_OPERATORS_BIAS_FAST_GELU_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/par_for.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

# define M_SQRT_2_PI 0.79788456080286535588 /* sqrt(2/pi) */

struct bias_fast_gelu
{
    std::string name() const { return "bias_fast_gelu"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        return inputs.front();
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto bias_dim = args[1].get_shape().elements();
        visit_all(result, args[0], args[1])([&](auto output, auto input, auto bias) {
            par_for(output_shape.elements(), [&](auto i) {
                auto x = input[i] + bias[i % bias_dim];
                //output[i] = (0.5 * x * (1 + tanh(M_SQRT_2_PI * (x + 0.044715 * x * x * x))));
                output[i] = x * 0.5 * (1 + erf(x * M_SQRT1_2));
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
