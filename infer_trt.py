import tensorrt as trt
from PIL import Image
import numpy as np
import cv2, torch, time
from typing import Any, Dict, List, Tuple
from torch.nn import functional as F
from TRT.inference import TRTInferenceV2

from segment_anything.utils.transforms import ResizeLongestSide

import torch, os

# from efficient_sam.build_efficient_sam import build_efficient_sam_vits
# from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
from torch.nn import functional as F

import onnx_models
from segment_anything.build_sam import sam_model_registry

# os.environ['CUDA_MODULE_LOADING'] = 'LAZY' 

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

def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        (64, 64),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


sam_model_checkpoints = {
    'h': "sam_vit_h_4b8939.pth",
    'l': "sam_vit_l_0b3195.pth",
    'b': "sam_vit_b_01ec64.pth",
}

# model_size = "l"
model_size = "b"


def baseline():
    vit_type = f"vit_{model_size}"
    decoder_checkpoint = f"../SAM/ckpts/sam_{model_size}_best_epoch.pth"
    print(f"{decoder_checkpoint = }")
    device = torch.device("cuda:0")

    sam_model_path = f'/disk/home/ans/SAM/SAM-checkpoint/{sam_model_checkpoints[model_size]}'
    print(f"{sam_model_path = }")
    sam_model = sam_model_registry[vit_type](checkpoint=sam_model_path)

    encoder = onnx_models.SamEncoderOnnxModel(sam_model)

    decoder = onnx_models.SAMSegHeadOnnx(vit_type)
    decoder.load_state_dict(torch.load(decoder_checkpoint, map_location='cpu'))
    
    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    transform = ResizeLongestSide(1024)

    imgfile = "../SAM/camera/frame2991.png"
    image = cv2.imread(imgfile)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ori_size = rgb_image.shape[:2]
    
    input_image = transform.apply_image(rgb_image)
    input_size = input_image.shape[:2]
    
    input_image_torch = torch.as_tensor(input_image)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    save_path = f'./output/'
    os.makedirs(save_path, exist_ok=True)

    for _ in range(10):
        st_time = time.time()
        with torch.no_grad():

            img = preprocess(input_image_torch).to(device)

            image_embedding, inner_state = encoder(img)
            print(torch.mean(image_embedding), torch.mean(inner_state))

            pred_mask = decoder(image_embedding, inner_state)
            pred_mask = postprocess_masks(pred_mask, input_size, ori_size)

        post_start = time.time()
        pred = torch.softmax(pred_mask, dim=1).float()[0][1]
        pred = torch.where(pred > 0.98, 1, 0)
        pred = torch.as_tensor(pred, dtype=torch.uint8)

        pred = pred.cpu().detach().numpy()

        index = np.where(pred == 1)
        image[index[0], index[1], :] = [255, 0, 85]

        print(time.time() - st_time)
        # print(f"{time.time() - post_start = }")
    save_image_path = os.path.join(save_path, f"result_{model_size}.png")
    cv2.imwrite(save_image_path, image)
    print(f"Save to {save_image_path = }")

config = "fp32"
config = "fp16"

input_channels = {
    "l": 1024,
    "b": 768,
}

if __name__ == "__main__":

    # baseline()
    # exit()

    enc_enginefile = f"./weights/sam_vit{model_size}_encoder_{config}.trt"
    dec_enginefile = f"./weights/sam_vit{model_size}_decoder_{config}.trt"
    print(f"{enc_enginefile = }")
    print(f"{dec_enginefile = }")
    

    image = np.array(Image.open("../SAM/camera/frame2991.png"))
    ori_size = image.shape[:2]
    
    input_size = [1024, 1024]
    input_image = image.transpose(2, 0, 1)[None].astype(np.float32)

    img = preprocess(torch.from_numpy(input_image))

    #encoder设置
    encoder_inputs = {
        "batched_images": img
    }

    encoder_input_shapes = {
        "batched_images": img.shape
    }

    #Todo: inner_features根据ViT自动选择
    encoder_output_shapes = {
        "image_embeddings": (1,256, 64,64),
        "inner_features": (24, 1, 64, 64, input_channels[model_size])
    }

    #decoder设置
    decoder_inputs = { #推理时指定输入，此处暂时设为None
        "image_embeddings": None,
        "inner_features": None
    }

    decoder_input_shapes = encoder_output_shapes

    decoder_output_shapes = {
        "masks": (1, 2, 256, 256)
    }
    
    trt_enc_obj = TRTInferenceV2(enc_enginefile, encoder_input_shapes, encoder_output_shapes, device=0)

    ## decoder test
    trt_dec_obj = TRTInferenceV2(dec_enginefile, decoder_input_shapes, decoder_output_shapes, device=0)

    save_path = f'./output/'
    os.makedirs(save_path, exist_ok=True)
    try:
        import torch, time

        no_copy_image_embeddings = True

        for i in range(10):
            st_time  = time.time()

            # 不用将encoder的输出从GPU拷贝到CPU
            _, device_outs = trt_enc_obj.inference(encoder_inputs, d2h=False)

            #如果不拷贝数据，可以直接指定image_embeddings和inner_features在显存中的地址
            ignore_inputs = ['image_embeddings', 'inner_features']
            trt_dec_obj.context.set_tensor_address('image_embeddings', device_outs[0])
            trt_dec_obj.context.set_tensor_address('inner_features', device_outs[1])
            
            #decoder部分需要输出结果，所以需要将结果从GPU拷贝到CPU
            outs, _ = trt_dec_obj.inference(decoder_inputs, ignore_inputs, d2h=True)

            predicted_logits = torch.from_numpy(np.reshape(outs[0], decoder_output_shapes['masks']))

            pred_mask = postprocess_masks(predicted_logits, input_size, ori_size)

            pred = torch.softmax(pred_mask, dim=1).float()[0][1]
            rows, cols = torch.where(pred > 0.975)
            if len(rows) > 0:
                image[rows.numpy(), cols.numpy(), :] = [255, 0, 85]

            print(time.time() - st_time)

            save_image_path = os.path.join(save_path, f"result_{model_size}_{config}.png")
            cv2.imwrite(save_image_path, image)
            print(f"Save to {save_image_path = }")
            # break
            pass
            # outputs = []
            # height, width = input_image.shape[2:4]
            # mask = np.zeros((height, width), dtype=np.bool_)
            # for points, labels in zip(input_points_list, input_labels_list):
            #     decoder_inputs["batched_point_coords"] = points
            #     decoder_inputs["batched_point_labels"] = labels

            #     outs, _ = trt_dec_obj.inference(decoder_inputs, ignore_inputs)

            #     predicted_logits = torch.from_numpy(np.reshape(outs[0], input_image.shape)[None,:])
            #     predicted_iou = torch.from_numpy(np.reshape(outs[1], (1,1,3)))

            #     sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            #     predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            #     predicted_logits = torch.take_along_dim(
            #         predicted_logits, sorted_ids[..., None, None], dim=2
            #     )
            #     # The masks are already sorted by their predicted IOUs.
            #     # The first dimension is the batch size (we have a single image. so it is 1).
            #     # The second dimension is the number of masks we want to generate (in this case, it is only 1)
            #     # The third dimension is the number of candidate masks output by the model.
            #     # For this demo we use the first mask.
            #     mask += torch.ge(predicted_logits[0, 0, 0, :, :], -0.0).cpu().numpy()

        #     print(f"time: {time.time() - st_time}")

        # masked_image_np = image.copy().astype(np.uint8) * mask[:,:,None]
        # Image.fromarray(masked_image_np).save(f"figs/examples/dogs_trt_mask.png")

    finally:
        trt_dec_obj.destroy()
        trt_enc_obj.destroy()

    pass
    

