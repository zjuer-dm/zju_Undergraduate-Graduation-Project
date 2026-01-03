import torch
import torch.nn as nn
import torchvision
import numpy as np
import clip  # CLIP encoder

from ddppo_resnet.resnet_policy import PNResnetDepthEncoder

class RGBEncoder(nn.Module):
    """CLIP ViT-B/32 encoder for RGB images (replaces ResNet50)"""
    def __init__(self, resnet_pretrain=True, trainable=False):
        super(RGBEncoder, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if resnet_pretrain:
            print('\nLoading CLIP ViT-B/32 for RGB ...')
        
        self.model, _ = clip.load("ViT-B/32", device=device)
        for param in self.model.parameters():
            param.requires_grad_(trainable)
        self.model.eval()
        
        from torchvision import transforms
        self.rgb_transform = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073], 
                [0.26862954, 0.26130258, 0.27577711]
            ),
        )

    def forward(self, rgb_imgs):
        # rgb_imgs shape from dataloader: (batch, num_imgs, 3, 224, 224)
        # Dataloader only converts to float [0,1], no normalization applied
        rgb_shape = rgb_imgs.size()
        rgb_imgs = rgb_imgs.reshape(
            rgb_shape[0]*rgb_shape[1], 
            rgb_shape[2], rgb_shape[3], rgb_shape[4]
        )
        # Now: (batch*num_imgs, 3, 224, 224)
        
        # Apply CLIP normalization
        rgb_imgs = self.rgb_transform(rgb_imgs)
        
        # CLIP encoding
        rgb_feats = self.model.encode_image(rgb_imgs.contiguous())
        
        # Output: (batch*num_imgs, 512)
        return rgb_feats.float()



class DepthEncoder(nn.Module):
    def __init__(self, resnet_pretrain=True, trainable=False):
        super(DepthEncoder, self).__init__()

        self.depth_net = PNResnetDepthEncoder()
        if resnet_pretrain:
            print('Loading PointNav pre-trained Resnet50 for Depth ...')
            # TODO: Change to your ddppo model path
            ddppo_pn_depth_encoder_weights = torch.load('/home/wdm/ICRA2026_etpnav/data/ddppo-models/gibson-2plus-resnet50.pth')
            weights_dict = {}
            for k, v in ddppo_pn_depth_encoder_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue
                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v
            del ddppo_pn_depth_encoder_weights
            self.depth_net.load_state_dict(weights_dict, strict=True)
        for param in self.depth_net.parameters():
            param.requires_grad_(trainable)

    def forward(self, depth_imgs):
        depth_shape = depth_imgs.size()
        depth_imgs = depth_imgs.reshape(depth_shape[0]*depth_shape[1],
                                    depth_shape[2], depth_shape[3], depth_shape[4])
        depth_feats = self.depth_net(depth_imgs)

        # print('depth_imgs', depth_imgs.shape)
        # print('depth_feats', depth_feats.shape)
        #
        # import pdb; pdb.set_trace()

        return depth_feats
