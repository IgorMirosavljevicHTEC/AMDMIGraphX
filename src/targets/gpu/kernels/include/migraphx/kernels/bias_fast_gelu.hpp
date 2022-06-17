
#ifndef MIGRAPHX_GUARD_KERNELS_BIAS_FAST_GELU_HPP
#define MIGRAPHX_GUARD_KERNELS_BIAS_FAST_GELU_HPP

#include <migraphx/kernels/ops.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace migraphx {

#define M_SQRT_2_PI 0.79788456080286535588 /* sqrt(2/pi) */

// constants for approximating the normal cdf
constexpr float A   = 0.5;
constexpr float B   = 0.7978845608028654;   // sqrt(2.0/M_PI)
constexpr float C   = 0.035677408136300125; // 0.044715 * sqrt(2.0/M_PI)
constexpr float D   = 1.702; // used for sigmoid approximation
constexpr float one = 1.0;
constexpr float two = 2.0;

template <class T1, class T2>
struct biasfastgelu_settings
{
    T1 elements{};
    T2 bias_dim{};
};

template <class T1, class T2>
constexpr biasfastgelu_settings<T1, T2> make_biasfastgelu_settings(T1 x, T2 y)
{
    return {x, y};
}

template <class Settings>
__device__ void bias_fast_gelu_half2(void* input, void* bias, void* output, Settings s)
{
    __half2* hinput  = reinterpret_cast<__half2*>(input);
    __half2* hbias   = reinterpret_cast<__half2*>(bias);
    __half2* houtput = reinterpret_cast<__half2*>(output);

    auto index    = make_index();
    auto i        = index.global;
    auto elements = s.elements;
    auto bias_dim = s.bias_dim;

    if(i < elements)
    {
        auto sum = __hadd2(hinput[i], hbias[i % bias_dim]);

        /* // tanh approximation
        // Batch size: 1
        // Rate: 14480.7/sec
        // Batch size: 64
        // Rate: 93224.8/sec
        const half2 A2 = __float2half2_rn(A);
        const half2 B2 = __float2half2_rn(B);
        const half2 C2 = __float2half2_rn(C);

        auto u1 = __hmul2(C2, sum);
        u1 = __hmul2(u1, sum);
        u1 = __hadd2(u1, B2);
        u1 = __hmul2(sum, u1);
        auto f2 = __half22float2(u1);
        f2.x = ::tanh(f2.x);
        f2.y = ::tanh(f2.y);
        auto h2 = __floats2half2_rn(f2.x, f2.y);
        auto cdf = __hmul2(h2, A2);
        cdf = __hadd2(cdf, A2);
        houtput[i] = __hmul2(sum, cdf); */

        /* // ORT tanh approximation approximation; tanh(x) ~=  2/(1+exp(-2*x))-1
        // Batch size: 1
        // Rate: 15494.8/sec
        // Batch size: 64
        // Rate: 93833.3/sec
        const half2 A2   = __float2half2_rn(A);
        const half2 B2   = __float2half2_rn(B);
        const half2 C2   = __float2half2_rn(C);
        const half2 one2 = __float2half2_rn(one);
        const half2 two2 = __float2half2_rn(two);

        auto u1    = __hmul2(C2, sum);
        u1         = __hmul2(u1, sum);
        u1         = __hadd2(u1, B2);
        auto u2    = __hmul2(two2, sum);
        u2         = __hmul2(u2, u1);
        u2         = __hneg2(u2);
        auto emu   = h2exp(u2);
        auto cdf   = __hadd2(one2, emu);
        cdf        = __h2div(two2, cdf);
        cdf        = __hsub2(cdf, one2);
        cdf        = __hmul2(A2, cdf);
        cdf        = __hadd2(cdf, A2);
        houtput[i] = __hmul2(sum, cdf); */


        // Sigmoid approximation
        // Batch size: 1
        // Rate: 17930/sec
        // Batch size: 64
        // Rate: 93899.1/sec
        const half2 one2 = __float2half2_rn(one);
        const half2 D2 = __float2half2_rn(D);

        auto inner = __hmul2(D2, sum);
        inner = __hneg2(inner);
        auto sig = h2exp(inner);
        sig = __hadd2(one2, sig);
        sig = __h2div(one2, sig);
        houtput[i] = __hmul2(sig, sum);



        /* // erf forumaltion
        // Batch size: 1
        // Rate: 14084.4/sec
        // Batch size: 64
        // Rate: 92833.3/sec

        __half2 sqrt2 = __float2half2_rn(M_SQRT1_2);
        auto x        = __hmul2(sum, sqrt2);
        auto f2       = __half22float2(x);
        f2.x          = ::erff(f2.x);
        f2.y          = ::erff(f2.y);
        auto h2       = __floats2half2_rn(f2.x, f2.y);

        auto one2 = __float2half2_rn(1.0f);
        h2       = __hadd2(h2, one2);

        __half2 point5 = __float2half2_rn(0.5f);
        houtput[i]        = __hmul2(sum, __hmul2(point5, h2));  */

        // Compare to gpu::code_object::add_mul_erf_add_mul_mul_kernel
        // Batch size: 1
        // Rate: 14083/sec
        // Batch size: 64
        // Rate: 57932.8/sec
    }
}

template <class Input, class Bias, class Output>
__device__ void bias_fast_gelu(const Input& input, const Bias& bias, const Output& output)
{
    auto index       = make_index();
    auto i           = index.global;
    auto input_shape = input.get_shape();
    if(i < input_shape.elements())
    {
        auto x    = input[i] + bias[i];
        auto u    = 2.0 * x * (0.044715 * M_SQRT_2_PI * x * x + M_SQRT_2_PI);
        auto emu  = ::exp(-u);
        auto cdf  = 0.5 + 0.5 * (2.0 / (1.0 + emu) - 1.0);
        output[i] = x * cdf;
        // output[i] = x * 0.5 * (1 + ::erf(x * M_SQRT1_2));
        // output[i] = (0.5 * x * (1 + ::tanh(M_SQRT_2_PI * (x + 0.044715 * x * x * x))));
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_SOFTMAX_HPP
