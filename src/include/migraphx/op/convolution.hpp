#ifndef MIGRAPHX_GUARD_OPERATORS_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONVOLUTION_HPP

#include <array>
#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct convolution
{
    std::vector<std::size_t> padding  = {0, 0};
    std::vector<std::size_t> stride   = {1, 1};
    std::vector<std::size_t> dilation = {1, 1};

    int group                   = 1;
    padding_mode_t padding_mode = default_;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.group, "group"),
                    f(self.padding_mode, "padding_mode"));
    }

    std::string name() const { return "convolution"; }

    void check_attribute_size() const
    {
        if(not((padding.size() == stride.size() or (padding.size() / 2) == stride.size()) and
               stride.size() == dilation.size()))
        {
            MIGRAPHX_THROW("CONVOLUTION: inconsistent attribute sizes");
        }
    }

    value attributes() const { return {{"normalize_padding", "padding"}}; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type().same_ndims().min_ndims(3);
        check_attribute_size();
        // num of dims of input and attribute should match
        const auto input_size   = inputs[0].max_lens().size();
        const auto padding_size = padding.size();
        if(not(input_size == padding_size / 2 + 2 or input_size == padding_size + 2))
        {
            MIGRAPHX_THROW("CONVOLUTION: input and attribute size mismatch!");
        }

        const shape& input            = inputs.at(0);
        const shape& weights          = inputs.at(1);
        const size_t num_spatial_dims = input_size - 2;
        if(num_spatial_dims != this->kdims())
        {
            MIGRAPHX_THROW("CONVOLUTION: input k-dims does not match attribute size");
        }

        if(not input.dynamic() and not weights.dynamic() and
           input.lens().at(1) != (weights.lens().at(1) * group))
            MIGRAPHX_THROW("CONVOLUTION: mismatched channel numbers");

        auto calc_output_lens =
            [this, &num_spatial_dims, &padding_size](std::vector<std::size_t> i_lens,
                                                     std::vector<std::size_t> w_lens) {
                std::vector<size_t> ret = {};
                // calculate the output shape of the convolution: ((W - K + 2P) / S) + 1
                for(size_t i = 0; i < num_spatial_dims; i++)
                {
                    auto padding_factor = 2 * padding[i];
                    if(padding_size == 2 * num_spatial_dims)
                    {
                        // when padding is {x0_begin, x1_begin, ... x0_end , x1_end, ...}
                        padding_factor = padding[i] + padding[i + num_spatial_dims];
                    }
                    ret.push_back(std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        (i_lens[i + 2] - (1 + dilation[i] * (w_lens[i + 2] - 1)) + padding_factor) /
                                stride[i] +
                            1)));
                }
                return ret;
            };

        if(input.dynamic())
        {
            std::vector<shape::dynamic_dimension> output_dyn_dims = {input.dyn_dims().at(0),
                                                                     input.dyn_dims().at(1)};
            auto min_spatial_dims = calc_output_lens(input.min_lens(), weights.min_lens());
            auto max_spatial_dims = calc_output_lens(input.max_lens(), weights.max_lens());
            auto opt_spatial_dims = calc_output_lens(input.opt_lens(), weights.opt_lens());
            for(size_t i = 0; i < num_spatial_dims; ++i)
            {
                output_dyn_dims.push_back(shape::dynamic_dimension{
                    min_spatial_dims[i], max_spatial_dims[i], opt_spatial_dims[i]});
            }
            return shape{input.type(), output_dyn_dims};
        }
        else
        {
            std::vector<size_t> output_lens{input.lens()[0], weights.lens()[0]};
            auto spatial_lens = calc_output_lens(input.lens(), weights.lens());
            std::for_each(spatial_lens.begin(), spatial_lens.end(), [&output_lens](auto x) {
                output_lens.push_back(x);
            });
            return inputs[0].with_lens(output_lens);
        }
    }

    size_t kdims() const
    {
        check_attribute_size();
        return stride.size();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
