import argparse
import numpy as np
import migraphx
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description='MIGraphX accuracy checker. Use to verify onnx files to ensure MIGraphX\'s output \
                                                  is within tolerance of onnx runtime\'s expected output.')
    req_args = parser.add_argument_group(title='required arguments')
    req_args.add_argument('--onnx',
                        type=str,
                        required=True,
                        help='path to onnx file')

    parser.add_argument('--batch',
                        type=int,
                        default=1,
                        help='batch size (if specified in onnx file)')
    parser.add_argument('--fill1',
                        action='store_true',
                        help='fill all arguments with a value of 1')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='show verbose information (for debugging)')
    parser.add_argument('--tolerance',
                        type=float,
                        default=1e-3,
                        help='accuracy tolerance (default = 1e-3)')
    args = parser.parse_args()
    
    return args

# taken from ../test_runner.py
def check_correctness(gold_outputs, outputs, rtol=1e-3, atol=1e-3, verbose=False):
    if len(gold_outputs) != len(outputs):
        print('Number of outputs {} is not equal to expected number {}'.format(
            len(outputs), len(gold_outputs)))
        return False

    out_num = len(gold_outputs)
    ret = True
    for i in range(out_num):
        if not np.allclose(gold_outputs[i], outputs[i], rtol, atol):
            ret = False
            if verbose:
                print('\nOutput {} is incorrect ...'.format(i))
                print('Expected value: \n{}'.format(gold_outputs[i]))
                print('......')
                print('Actual value: \n{}\n'.format(outputs[i]))
            else:
                print('Outputs do not match')
                break

    return ret

def get_np_datatype(in_type):
    datatypes = { 
                  'float64_type' : np.float64,
                  'float_type' : np.float32,
                  'half_type' : np.half,
                  'int64_type' : np.int64,
                  'int32_type' : np.int32,
                  'int8_type' : np.int8
                  }
    return datatypes[in_type]

def main():
    args = parse_args()

    model_name = args.onnx
    batch=args.batch

    model = migraphx.parse_onnx(model_name, default_dim_value=batch)


    model.compile(migraphx.get_target('gpu'), offload_copy=False)

    in_shape = []
    params = {}
    test_input = None
    test_inputs = []
    for key,value in model.get_parameter_shapes().items():
        if args.verbose:
            print('Parameter {} -> {}'.format(key,value))
        if not 'output' in key:
            in_shape = value.lens()
            in_type = value.type_string()
            if not args.fill1:
                test_input = np.random.rand(*(in_shape)).astype(get_np_datatype(in_type))
            else:
                test_input = np.ones(in_shape).astype(get_np_datatype(in_type))
            test_inputs.append(test_input)
            params[key] = migraphx.to_gpu(migraphx.argument(test_input))
        else:
            params[key] = migraphx.to_gpu(migraphx.generate_argument(value))


    pred_migx = np.array(migraphx.from_gpu(model.run(params)[-1]))


    sess = ort.InferenceSession(model_name)

    ort_params = {}
    for i,input in enumerate(sess.get_inputs()):
        ort_params[input.name] = test_inputs[i]

    pred_ort = sess.run(None, ort_params)[-1]


    is_correct = check_correctness(pred_ort, pred_migx, args.tolerance, args.tolerance, args.verbose)
    if is_correct:
        print('Passed: MIGraphX meets tolerance')
    else:
        print('FAILED: MIGraphX is not within tolerance. Rerun with --verbose for detailed information.')

if __name__ == '__main__':
    main()
