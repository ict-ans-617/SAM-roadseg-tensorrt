from TRT.onnx2trt import  convert
import torch
mode = "fp32"
mode = "fp16"

model_size="l"
# model_size="b"

#目前batchsize只能支持固定的数值，不能设置动态的，否则报错
print("convert esam encoder")
dynamic_input = {
    "batched_images": [(1, 3, 1024, 1024), (1, 3, 1024, 1024), (1, 3, 1024, 1024)]
}
convert(f"./weights/sam_vit{model_size}_encoder.onnx", f"./weights/sam_vit{model_size}_encoder_{mode}.trt", dynamic_input, precision=mode)

input_channels = {
    "l": 1024,
    "b": 768,
}

print("convert esam decoder")
dynamic_input = {
    "image_embeddings": [(1, 256, 64, 64),(1, 256, 64, 64),(1, 256, 64, 64)],
    "inner_features": [(24, 1, 64, 64, input_channels[model_size]),(24, 1, 64, 64, input_channels[model_size]),(24, 1, 64, 64, input_channels[model_size])]
}
convert(f"./weights/sam_vit{model_size}_decoder.onnx", f"./weights/sam_vit{model_size}_decoder_{mode}.trt", dynamic_input, precision=mode)