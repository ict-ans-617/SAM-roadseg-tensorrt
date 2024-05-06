import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Tuple

from segment_anything.modeling import Sam

from seg_decoder import ConvModule, MLP, SAMAggregatorNeck, SegHead
import einops
   
class SamEncoderOnnxModel(nn.Module):
    
    def __init__(
        self,
        model: Sam
    ) -> None:
        super().__init__()
        self.vit_model = model.image_encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit_model.patch_embed(x)
        if self.vit_model.pos_embed is not None:
            x = x + self.vit_model.pos_embed 

        out = []
        for blk in self.vit_model.blocks:
            x = blk(x)
            out.append(x)

        x = self.vit_model.neck(x.permute(0, 3, 1, 2))
        out = torch.stack(out, axis=0)
        return x, out

class SAMAggregatorNeckOnnx(SAMAggregatorNeck):
    def __init__(self, in_channels=[384] * 12, inner_channels=128, selected_channels=range(1, 12, 1), out_channels=256, up_sample_scale=4):
        super().__init__(in_channels, inner_channels, selected_channels, out_channels, up_sample_scale)

        pass

    #重新继承是为了 将forward函数改成 输入为两个tensor的格式
    def forward(self, image_embedding, inner_states ):     
        ######################
      
        inner_states = [einops.rearrange(inner_states[idx], 'b h w c -> b c h w') for idx in self.selected_channels]
        inner_states = [layer(x) for layer, x in zip(self.down_sample_layers, inner_states)]

        x = None
        for inner_state, layer in zip(inner_states, self.fusion_layers):
            if x is not None:
                inner_state = x + inner_state
            x = inner_state + layer(inner_state)
        x = self.up_layers[0](x) + x
        img_feats_0 = self.up_layers[1](x)

        img_feats_1 = self.up_sample_layers[0](img_feats_0) + self.up_sample_layers[1](img_feats_0)

        img_feats_2 = self.up_sample_layers[2](img_feats_1) + self.up_sample_layers[3](img_feats_1)

        return img_feats_2, img_feats_1, img_feats_0, image_embedding

class SAMSegHeadOnnx(SegHead):
    def __init__(self, vit_type="vit_h"):
        super().__init__(vit_type)
        self.neck_net = SAMAggregatorNeckOnnx()
        if vit_type == "vit_l":
            self.neck_net = SAMAggregatorNeckOnnx(in_channels=[1024]*24, selected_channels = range(4, 24, 2))
        #Todo: 补充其他vit版本的self.neck_net
        elif vit_type == "vit_b":
            self.neck_net = SAMAggregatorNeckOnnx(in_channels=[768]*12, selected_channels = range(4, 12, 2))
            pass
        elif vit_type == "vit_s":
            pass
        elif vit_type == "vit_ti":
            pass
        pass
    
    #重新继承是为了 将forward函数改成 输入为两个tensor的格式
    def forward(self, image_embedding, inner_states):
        x = self.neck_net(image_embedding, inner_states)
        c1, c2, c3, c4 = x
        # print('c4.shape:',c4.shape)
        # print('c1.shape:',c1.shape)
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.linear_pred(x)

        return x