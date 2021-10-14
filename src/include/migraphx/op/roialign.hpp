#ifndef MIGRAPHX_GUARD_OPERATORS_ROIALIGN_HPP
#define MIGRAPHX_GUARD_OPERATORS_ROIALIGN_HPP

#include <limits>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/shape_for_each.hpp>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct roialign
{
    std::string coord_trans_mode = "half_pixel";
    std::string mode             = "avg";
    int64_t output_height        = 1;
    int64_t output_width         = 1;
    int64_t sampling_ratio       = 0;
    float spatial_scale          = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.coord_trans_mode, "coordinate_transformation_mode"),
                    f(self.mode, "mode"),
                    f(self.output_height, "output_height"),
                    f(self.output_width, "output_width"),
                    f(self.sampling_ratio, "sampling_ratio"),
                    f(self.spatial_scale, "spatial_scale"));
    }

    std::string name() const { return "roialign"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3).standard();
        auto x_lens   = inputs.at(0).lens();
        auto roi_lens = inputs.at(1).lens();
        auto bi_lens  = inputs.at(2).lens();
        auto type     = inputs.at(0).type();

        // check input correct
        if(bi_lens.size() != 1)
        {
            MIGRAPHX_THROW("ROIALIGN: batch indices should be 1 dimension!");
        }

        if(roi_lens.size() != 2 or roi_lens.at(1) != 4)
        {
            MIGRAPHX_THROW(
                "ROIALIGN: rois should be 2 dimensions, and the second dim should be 4!");
        }

        if(roi_lens.front() != bi_lens.front())
        {
            MIGRAPHX_THROW("ROIALIGN: rois and batch indices inputs should have the same number!");
        }

        std::vector<std::size_t> out_lens = x_lens;
        out_lens[0]                       = roi_lens[0];
        out_lens[2]                       = output_height;
        out_lens[3]                       = output_width;

        return {type, out_lens};
    }

    struct pos_weight
    {
        std::array<std::int64_t, 4> pos = {0, 0, 0, 0};
        std::array<float, 4> w          = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    auto calc_pos_weight(const int64_t height,
                         const int64_t width,
                         const shape& comp_s,
                         const std::array<float, 2>& roi_start,
                         const std::array<float, 2>& bin_size,
                         const std::array<int64_t, 2>& bin_grid_size) const
    {
        std::vector<pos_weight> results(bin_grid_size[0] * bin_grid_size[1] * output_height *
                                        output_width);
        shape_for_each(comp_s, [&](auto idx) {
            std::array<std::size_t, 2> p = {idx[0], idx[1]};
            std::array<std::size_t, 2> i = {idx[2], idx[3]};
            auto index                   = comp_s.index(idx);
            const float yy =
                roi_start[0] + p[0] * bin_size[0] + (i[0] + .5f) * bin_size[0] / bin_grid_size[0];
            const float xx =
                roi_start[1] + p[1] * bin_size[1] + (i[1] + .5f) * bin_size[1] / bin_grid_size[1];

            float x = (coord_trans_mode == "output_half_pixel") ? (xx - 0.5f) : xx;
            float y = (coord_trans_mode == "output_half_pixel") ? (yy - 0.5f) : yy;

            // deal with: inverse elements are out of feature map boundary
            if(y < -1.0 || y > height || x < -1.0 || x > width)
            {
                results[index] = pos_weight{};
                return;
            }

            y             = std::max(y, 0.0f);
            x             = std::max(x, 0.0f);
            int64_t y_low = y;
            int64_t x_low = x;
            int64_t y_high;
            int64_t x_high;

            y_high = y_low + 1;
            if(y_low >= height - 1)
            {
                y = y_high = y_low = height - 1;
            }

            x_high = x_low + 1;
            if(x_low >= width - 1)
            {
                x = x_high = x_low = width - 1;
            }
            results[index].pos = {y_low * width + x_low,
                                  y_low * width + x_high,
                                  y_high * width + x_low,
                                  y_high * width + x_high};

            float ly = y - y_low;
            float lx = x - x_low;
            float hy = 1.0f - ly;
            float hx = 1.0f - lx;
            // save weights and indeces
            results[index].w = {hy * hx, hy * lx, ly * hx, ly * lx};
        });

        return results;
    }

    struct max_pool
    {
        double init() { return std::numeric_limits<double>::lowest(); }

        double operator()(double x, double y)
        {
            return std::max(x, y);
        }

        double final(double x, std::size_t) { return (x); }
    };

    struct avg_pool
    {
        double init() { return 0.0; }

        double operator()(double x, double y) { return x + y; }

        double final(double x, std::size_t y) { return (y == 0) ? 0.0 : (x / y); }
    };

    template <class T, class Op>
    std::tuple<double, int64_t> calc_pooling(const T& data,
                                             const std::array<int64_t, 2>& bin_grid_size,
                                             const std::vector<pos_weight>& pos_weights,
                                             int64_t index,
                                             Op op) const
    {
        double output_val   = op.init();
        const int64_t count = bin_grid_size[0] * bin_grid_size[1];
        dfor(bin_grid_size[0], bin_grid_size[1])([&](auto, auto) {
            const auto& pc = pos_weights[index];
            std::array<double, 4> wv;
            std::transform(
                pc.w.begin(), pc.w.end(), pc.pos.begin(), wv.begin(), [&](auto w, auto pos) {
                    return *(data + pos) * w;
                });
            output_val = std::accumulate(wv.begin(), wv.end(), output_val, op);
            index += 1;
        });

        output_val = op.final(output_val, count);

        return {output_val, index};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        const auto& out_lens  = output_shape.lens();
        int64_t n_rois        = out_lens[0];
        int64_t channels      = out_lens[1];
        int64_t pooled_height = out_lens[2];
        int64_t pooled_width  = out_lens[3];
        const auto& x_lens    = args.at(0).get_shape().lens();
        auto height           = x_lens[2];
        auto width            = x_lens[3];
        auto roi_s            = args.at(1).get_shape();

        visit_all(result, args.at(0), args.at(1))([&](auto output, auto x, auto roi) {
            const auto* batch_indices = args.at(2).cast<int64_t>();
            par_for(n_rois, [&](auto n) {
                const auto bottom_data   = x.begin();
                const auto roi_batch_ind = batch_indices[n];
                // Do not using rounding; this implementation detail is critical
                float roi_start_w = roi[roi_s.index({n, 0})] * spatial_scale;
                float roi_start_h = roi[roi_s.index({n, 1})] * spatial_scale;
                float roi_end_w   = roi[roi_s.index({n, 2})] * spatial_scale;
                float roi_end_h   = roi[roi_s.index({n, 3})] * spatial_scale;

                // Force malformed ROIs to be 1x1
                float roi_width =
                    (roi_end_w - roi_start_w) > 1.0f ? (roi_end_w - roi_start_w) : 1.0f;
                float roi_height =
                    (roi_end_h - roi_start_h) > 1.0f ? (roi_end_h - roi_start_h) : 1.0f;
                float bin_size_h = roi_height / pooled_height;
                float bin_size_w = roi_width / pooled_width;

                // We use roi_bin_grid to sample the grid and mimic integral
                int64_t roi_bin_grid_h =
                    (sampling_ratio > 0) ? sampling_ratio : std::ceil(roi_height / pooled_height);
                int64_t roi_bin_grid_w =
                    (sampling_ratio > 0) ? sampling_ratio : std::ceil(roi_width / pooled_width);

                // we want to precalculate indices and weights shared by all channels,
                // this is the key point of optimization
                std::vector<int64_t> lens = {
                    pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w};
                std::vector<std::size_t> comp_lens(lens.begin(), lens.end());
                shape comp_s{shape::float_type, comp_lens};
                auto pre_calc = this->calc_pos_weight(height,
                                                      width,
                                                      comp_s,
                                                      {roi_start_h, roi_start_w},
                                                      {bin_size_h, bin_size_w},
                                                      {roi_bin_grid_h, roi_bin_grid_w});

                std::vector<int64_t> comp_lens1 = {channels, pooled_height, pooled_width};
                shape comp_s1{migraphx::shape::float_type, comp_lens1};
                std::vector<int64_t> vec_index(channels, 0);
                shape_for_each(comp_s1, [&](auto idx) {
                    auto c  = idx[0];
                    auto ph = idx[1];
                    auto pw = idx[2];

                    const auto offset_bottom_data =
                        bottom_data +
                        static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);
                    double output_val;
                    std::tie(output_val, vec_index[c]) =
                        (mode == "avg") ? this->calc_pooling(offset_bottom_data,
                                                             {roi_bin_grid_h, roi_bin_grid_w},
                                                             pre_calc,
                                                             vec_index[c],
                                                             avg_pool{})
                                        : this->calc_pooling(offset_bottom_data,
                                                             {roi_bin_grid_h, roi_bin_grid_w},
                                                             pre_calc,
                                                             vec_index[c],
                                                             max_pool{});
                    output[output_shape.index({n, c, ph, pw})] = output_val;
                });

            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
