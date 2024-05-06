
import onnxruntime
import torch, subprocess

# from efficient_sam.build_efficient_sam import build_efficient_sam_vits
# from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
from torch.nn import functional as F

import onnx_models
from segment_anything.build_sam import sam_model_registry

from seg_decoder import SegHead, SegHeadUpConv

OPSET=17

def preprocess(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    x = x.to(torch.float32).to(pixel_mean.device)
    
    if x.shape[2] != 1024 or x.shape[3] != 1024: 
        x = F.interpolate(
            x,
            (1024, 1024),
            mode="bilinear",
        )
    
    x = (x - pixel_mean) / pixel_std
    return x

def export_onnx(onnx_model, output, dynamic_axes, dummy_inputs, output_names):
    with open(output, "wb") as f:
        print(f"Exporting onnx model to {output}...")
        torch.onnx.export(
        # torch.onnx.dynamo_export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=OPSET, #17
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    inference_session = onnxruntime.InferenceSession(output)
    output = inference_session.run(
        output_names=output_names,
        input_feed={k: v.numpy() for k, v in dummy_inputs.items()},
    )
    print(output_names)
    print([output_i.shape for output_i in output])


def export_onnx_encoder(model, output):
    '''The function `export_onnx_esam_encoder` exports a PyTorch model to ONNX format, specifically for an
    EfficientSamEncoder model.
    
    Parameters
    ----------
    model
        The `model` parameter is the PyTorch model that you want to export to ONNX format. It should be an
    instance of the model class that you want to export.
    output
        The `output` parameter is the file path where the exported ONNX model will be saved. It should be a
    string representing the file path, including the file name and extension. For example,
    `output="model.onnx"`.
    
    '''
    onnx_model = onnx_models.SamEncoderOnnxModel(model=model)
    #Todo: 将preprocess加到onnx装换中
    img = preprocess(torch.randn(1, 3, 1080, 1920, dtype=torch.float))

    dynamic_axes = {
        "batched_images": {0: "batch", 2: "height", 3: "width"},
    }
    dummy_inputs = {
        "batched_images": img,
    }
    output_names = ["image_embeddings", "inner_features"]
    export_onnx(
        onnx_model=onnx_model,
        output=output,
        dynamic_axes=dynamic_axes,
        dummy_inputs=dummy_inputs,
        output_names=output_names,
    )

def export_onnx_decoder(checkpoint, vit_type ,output):
    #Todo：把后处理代码集成到SAMSegHeadOnnx中，直接导出onnx
    onnx_model = onnx_models.SAMSegHeadOnnx(vit_type)
    onnx_model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    onnx_model.eval()

    #Todo: inner_features根据vit模型自动选择
    dummy_inputs = {
        "image_embeddings": torch.randn(1, 256, 64, 64, dtype=torch.float),
        "inner_features": torch.randn(24, 1, 64, 64, onnx_model.neck_net.in_channels[0], dtype=torch.float)
    }

    output_names = ["masks"]

    with open(output, "wb") as f:
        print(f"Exporting onnx model to {output}...")
        torch.onnx.export(
        # torch.onnx.dynamo_export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            # dynamic_axes=dynamic_axes,
        )

sam_model_checkpoints = {
    'h': "sam_vit_h_4b8939.pth",
    'l': "sam_vit_l_0b3195.pth",
    'b': "sam_vit_b_01ec64.pth",
}

model_size = "l"
model_size = "b"

if __name__ == "__main__":
    '''导出decoder'''
    output_name = f"weights/sam_vit{model_size}_decoder.onnx"
    decoder_checkpoint = f"/disk/home/ans/SAM/ckpts/sam_{model_size}_best_epoch.pth"
    export_onnx_decoder(decoder_checkpoint, f"vit_{model_size}", output_name)
    # # 必须运行polygraphy，否则导出trt时会报错
    cmd = f"polygraphy surgeon sanitize {output_name} --fold-constants -o {output_name}"
    subprocess.run(cmd, shell=True, check=True)

    '''导出encoder'''
    output_name = f"weights/sam_vit{model_size}_encoder.onnx"
    sam_model_path = f'/disk/home/ans/SAM/SAM-checkpoint/{sam_model_checkpoints[model_size]}'
    sam_model = sam_model_registry[f'vit_{model_size}'](checkpoint=sam_model_path)

    export_onnx_encoder(
        model=sam_model,
        output=output_name,
    )
    # 必须运行polygraphy，否则导出trt时会报错
    cmd = f"polygraphy surgeon sanitize {output_name} --fold-constants -o {output_name}"
    subprocess.run(cmd, shell=True, check=True)


