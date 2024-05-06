import os
import time
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
cuda.init()
import numpy as np

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# The HostDeviceMem class represents a memory object that can be accessed by both the host and device
# in a Python program.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTInferenceV2(object):
    # The `TRTInferenceV2` class manages TensorRT objects for model inference. It is responsible for
    # loading a TensorRT engine from a file, creating an execution context, allocating memory for
    # input and output tensors, and performing inference using the TensorRT engine. It provides a
    # method `inference` to perform inference on input data and return the output results. The class
    # also handles memory management and destruction of TensorRT objects.
    """Manages TensorRT objects for model inference."""
    def __init__(self, trt_engine_path, input_shapes={}, output_shapes={}, device=0):
        '''This function initializes a TensorRT engine for inference, loads the engine from a file, sets
        input and output shapes, allocates memory for input and output tensors, and binds the tensors to
        device memory.
        
        Parameters
        ----------
        trt_engine_path
            The path to the TensorRT engine file that contains the optimized model for inference.
        input_shapes
            The `input_shapes` parameter is a dictionary that specifies the shapes of the input tensors for
        the TensorRT engine. The keys of the dictionary are the names of the input tensors, and the
        values are the corresponding shapes of the tensors.
        output_shapes
            The `output_shapes` parameter is a dictionary that specifies the shapes of the output tensors
        of the TensorRT engine. The keys of the dictionary are the names of the output tensors, and the
        values are the corresponding shapes of the tensors.
        device, optional
            The `device` parameter specifies the GPU device index to be used for inference. It is an
        optional parameter and defaults to 0, which represents the first GPU device. If you have
        multiple GPUs in your system and want to use a specific GPU for inference, you can specify the
        index of that GPU
        
        '''

        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        self.ctx = cuda.Device(device).make_context()
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)

        # TRT engine placeholder
        self.engine = None

        if not os.path.exists(trt_engine_path):
            raise Exception('tensorRT decoder engine file not exist')

        with open(trt_engine_path, "rb") as f:
            self.engine = self.trt_runtime.deserialize_cuda_engine(f.read())            

        # Execution context and stream is needed for inference
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Specify input shape
        for name, shape in input_shapes.items():
            self.context.set_input_shape(name, shape)

        assert self.context.all_shape_inputs_specified
        # assert self.context.all_binding_shapes_specified

        # Set selected profile idx
        self.context.set_optimization_profile_async(0, self.stream.handle)    
        
        #为输出tensor分配内存空间
        self.output_buffers = []
        for name, shape in output_shapes.items():
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            #绑定tensor和显存
            self.context.set_tensor_address(name, device_mem)
            self.output_buffers.append(HostDeviceMem(host_mem, device_mem))
        
        #为输入tensor分配内存空间
        self.input_buffers = []
        for name, shape in input_shapes.items():
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            #绑定tensor和显存
            self.context.set_tensor_address(name, device_mem)
            self.input_buffers.append(HostDeviceMem(host_mem, device_mem))


    def inference(self, decoder_dict_input, ignore_inputs=[], d2h=True):
        '''The function "inference" takes in a dictionary as input and returns a result, while ignoring
        certain inputs if specified.
        
        Parameters
        ----------
        decoder_dict_input
            The `decoder_dict_input` parameter is a dictionary that contains the inputs for the decoder. It
        is used to provide the necessary information for the inference process.
        ignore_inputs
            The `ignore_inputs` parameter is a list that contains the inputs that should be ignored during memory copy.
        '''
        # Numpy dtype should be float32
        self.ctx.push()

        #input data: 将输入数据从CPU转移到GPU
        for i, (name, nd_array) in enumerate(decoder_dict_input.items()):
            if name in ignore_inputs:
                continue
            self.input_buffers[i].host = np.ascontiguousarray(nd_array).ravel()
            cuda.memcpy_htod_async(self.input_buffers[i].device, self.input_buffers[i].host, self.stream)
        
        #  # Assertion: to ensure all the inputs are set
        assert len(self.context.infer_shapes()) == 0  # similar to all_shape_inputs_specified / all_binding_shapes_specified asserting
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        if d2h:
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.output_buffers]

        self.stream.synchronize()

        # Return only the host outputs.
        if d2h:
            host_outputs = [out.host for out in self.output_buffers]
        else:
            host_outputs = None

        device_outputs = [out.device for out in self.output_buffers]

        self.ctx.pop()
        return host_outputs, device_outputs

    def destroy(self):
        '''The `destroy` function is used to perform some action or cleanup before an object is destroyed.
        This function should be called to avoid pycuda error
        '''
        self.ctx.pop()