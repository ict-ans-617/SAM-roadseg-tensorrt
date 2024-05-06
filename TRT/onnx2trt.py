import tensorrt as trt

def convert(src, dst, dynamic_input, precision="fp32"):
    '''The function `convert` takes an ONNX file as input, converts it to a TensorRT engine, and saves the
    engine to a destination file.
    
    Parameters
    ----------
    src
        The `src` parameter is the path to the ONNX file that you want to convert to a TensorRT engine.
    dst
        The `dst` parameter is the destination file path where the converted TensorRT engine will be saved.
    dynamic_input
        The `dynamic_input` parameter is a dictionary that specifies the dynamic input shapes for the
    network. The keys of the dictionary are the names of the input tensors in the network, and the
    values are tuples representing the dynamic shape of each input tensor. example:
        dynamic_input = {
            "batched_images": [(1, 3, 512, 512), (1, 3, 1080, 1080), (1, 3, 1920, 1920)]
        }
    precision, optional
        The `precision` parameter specifies the precision of the engine to be built. It can be set to
    either "fp32" or "fp16". By default, it is set to "fp32".
    
    '''
    print(f"Begin converting: {src} --> {dst}")
    # logger = trt.Logger(trt.Logger.VERBOSE) #VERBOSE INFO WARNING ERROR 
    logger = trt.Logger(trt.Logger.WARNING) #VERBOSE INFO WARNING ERROR 
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = 20
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) #must set EXPLICIT_BATCH
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(src):
        raise RuntimeError(f'failed to load ONNX file: {src}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    profile = builder.create_optimization_profile()
    for inp in inputs:
        profile.set_shape(inp.name, *dynamic_input[inp.name])

    config.add_optimization_profile(profile)

    layers = [network.get_layer(i) for i in range(network.num_layers)]

    if precision.lower() == "fp16":
        # #norm层如果变为fp16，输出结果为NAN
        # for layer in layers:
        #     if layer.type == trt.LayerType.NORMALIZATION:
        #         layer.precision = trt.float32
        #         continue

        # fp32_layers = ["/image_encoder/neck/neck.1", "/image_encoder/neck/neck.3", "/Reshape_7", "/Reshape_8"]
        # for fp32_layer in fp32_layers:
        #     if fp32_layer in layer.name:
        #         layer.precision = trt.float32
        pass

    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    if precision.lower() == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f'building {precision} engine')
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    with builder.build_engine(network, config) as engine, open(dst, 'wb') as t:
        t.write(engine.serialize())