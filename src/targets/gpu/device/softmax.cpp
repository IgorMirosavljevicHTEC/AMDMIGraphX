#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/softmax.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void softmax(hipStream_t stream, const argument& result, const argument& arg, int64_t axis)
{
    auto batch_lens          = result.get_shape().lens();
    const index_int batch_item_num = batch_lens[axis];
    batch_lens[axis]         = 1;
    migraphx::shape batch_shape{result.get_shape().type(), batch_lens};

    hip_visit_all(result, arg, batch_shape)([&](auto output, auto input, auto batch) {
        const index_int max_block_size = 64;
        const index_int block_size     = compute_block_size(batch_item_num, max_block_size);
        const auto* input_ptr = device_cast(input.data());
        auto* output_ptr        = device_cast(output.data());
        gs_launch(stream,
                  batch_shape.elements() * block_size,
                  block_size)([=](auto i, auto idx) __device__ {
            auto offset = batch_item_num * idx.group;
            using type    = device_type<std::remove_cv_t<typename decltype(input)::value_type>>;
            MIGRAPHX_DEVICE_SHARED type buffer[384];
            for (int tidx = idx.local; tidx < batch_item_num; tidx += block_size)
            {
                buffer[tidx] = input_ptr[offset + tidx];
            }

            // block_reduce to compute max
            int stride = batch_item_num / 2;
            while (stride > 0)
            {
                for (int tidx = idx.local; tidx < stride; ++tidx)
                {
                    buffer[tidx] = (buffer[tidx] < buffer[tidx + stride]) ? buffer[tidx + stride] : buffer[tidx];
                }
                __syncthreads();
                stride = stride / 2;
            }

            type max_val = buffer[0];
            for (int tidx = idx.local; tidx < batch_item_num; tidx += block_size)
            {
                buffer[tidx] = ::exp(to_hip_type(input_ptr[tidx] - max_val));
            }

            stride = batch_item_num / 2;
            while (stride > 0)
            {
                for (int tidx = idx.local; tidx < stride; ++tidx)
                {
                    buffer[tidx] = buffer[tidx] + buffer[tidx + stride];
                }
                __syncthreads();
                stride = stride / 2;
            }

            type sum_val = buffer[0];

            for (int tidx = idx.local; tidx < batch_item_num; tidx += block_size)
            {
                output_ptr[offset + tidx] = ::exp(to_hip_type(input_ptr[tidx] - max_val)) / sum_val;
            }

            // auto data_idx = batch.multi(i / block_size);
            // using type    = device_type<std::remove_cv_t<typename decltype(input)::value_type>>;
            // type init     = lowest();

            // auto batch_max = block_reduce<max_block_size>(
            //     idx, max{}, init, batch_item_num, [&](auto j) __device__ {
            //         data_idx[axis] = j;
            //         return input[data_idx];
            //     });

            // auto batch_sum =
            //     block_reduce<max_block_size>(idx, sum{}, 0, batch_item_num, [&](auto j) __device__ {
            //         data_idx[axis] = j;
            //         auto val       = input[data_idx] - batch_max;
            //         return ::exp(to_hip_type(val));
            //     });

            // idx.local_stride(batch_item_num, [&](auto j) __device__ {
            //     data_idx[axis]   = j;
            //     auto val         = input[data_idx] - batch_max;
            //     output[data_idx] = ::exp(to_hip_type(val)) / batch_sum;
            // });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
