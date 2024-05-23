/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <onnx_test.hpp>
#include <migraphx/op/common.hpp>

TEST_CASE(conv_dynamic_img_same_upper)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_parameter("0", {migraphx::shape::float_type, {{1, 1}, {3, 3}, {5, 10}, {5, 10}}});
    auto l1 = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 3, 3}});
    auto c0 = mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {0, 0}},
                           {"stride", {1, 1}},
                           {"dilation", {1, 1}},
                           {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
        l0,
        l1);
    mm->add_return({c0});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {5, 10};

    auto prog = read_onnx("conv_dynamic_img_same_upper_test.onnx", options);
    EXPECT(p == prog);
}
