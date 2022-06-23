#ifndef MIGRAPHX_GUARD_RTGLIB_FASTGELU_REWRITE_HPP
#define MIGRAPHX_GUARD_RTGLIB_FASTGELU_REWRITE_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct rewrite_fastgelu
{
    std::string name() const { return "rewrite_fastgelu"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
