import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from daod.modeling.style_transfer.net import net_decoder, net_vgg


class StyleTransfer(nn.Module):
    def __init__(self, vgg, decoder, style=None, preserve_color=False, alpha=0.4, do_interpolation = False):
        super(StyleTransfer, self).__init__()
        # torch.multiprocessing.set_start_method('spawn')
        self.style = style
        self.vgg = vgg
        self.decoder = decoder
        self.preserve_color = preserve_color
        self.alpha = alpha
        self.device = "cuda"
        self.decoder = net_decoder
        self.vgg = net_vgg
        self.decoder.eval()
        self.vgg.eval()
        self.decoder.load_state_dict(torch.load(decoder))
        self.vgg.load_state_dict(torch.load(vgg))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        self.vgg.to(self.device)
        self.decoder.to(self.device)
        self.do_interpolation = do_interpolation
        
        #if self.style != None:
        # print(self.style)
        self.style = np.transpose(self.style, (2, 0, 1))
        self.style = torch.FloatTensor(self.style.copy())
        # print(self.style)
        #self.style = self.style.unsqueeze(0)
        #print(self.device)
        #print(self.style.device)
        self.style = self.style.to(self.device)
        print(self.style.device)
        #print(self.vgg.device)
        #self.style_f = self.vgg(self.style)

    def _mat_sqrt(self, x):
        U, D, V = torch.svd(x)
        return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def _calc_feat_flatten_mean_std(self, feat):
        # takes 3D feat (C, H, W), return mean and std of array within channels
        assert (feat.size()[0] == 3)
        assert (isinstance(feat, torch.cuda.FloatTensor))
        feat_flatten = feat.view(3, -1)
        mean = feat_flatten.mean(dim=-1, keepdim=True)
        std = feat_flatten.std(dim=-1, keepdim=True)
        return feat_flatten, mean, std

    def coral(self, source, target):
        # assume both source and target are 3D array (C, H, W)
        # Note: flatten -> f
        # print(source.type())
        source_f, source_f_mean, source_f_std = self._calc_feat_flatten_mean_std(source)
        source_f_norm = (source_f - source_f_mean.expand_as(
            source_f)) / source_f_std.expand_as(source_f)
        source_f_cov_eye = \
            torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3, device = "cuda")

        target_f, target_f_mean, target_f_std = self._calc_feat_flatten_mean_std(target)
        target_f_norm = (target_f - target_f_mean.expand_as(
            target_f)) / target_f_std.expand_as(target_f)
        target_f_cov_eye = \
            torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3, device = "cuda")

        source_f_norm_transfer = torch.mm(
            self._mat_sqrt(target_f_cov_eye),
            torch.mm(torch.inverse(self._mat_sqrt(source_f_cov_eye)),
                        source_f_norm)
        )

        source_f_transfer = source_f_norm_transfer * \
                            target_f_std.expand_as(source_f_norm) + \
                            target_f_mean.expand_as(source_f_norm)

        return source_f_transfer.view(source.size())


    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def style_transfer(self, style_f, content, style=None,
                   interpolation_weights=None):
        alpha = self.alpha
        content_f = self.vgg(content)
        if interpolation_weights:
            _, C, H, W = content_f.size()
            feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
            base_feat = self.adaptive_instance_normalization(content_f, style_f)
            for i, w in enumerate(interpolation_weights):
                feat = feat + w * base_feat[i:i + 1]
            content_f = content_f[0:1]
        else:
            feat = self.adaptive_instance_normalization(content_f, style_f)
            feat = feat * alpha + content_f * (1 - alpha)
        return self.decoder(feat)

    def forward(self, content):
        #content = np.transpose(content, (2, 0, 1))
        #content = torch.Tensor(content)
        style = self.coral(self.style, content)
        content = content.unsqueeze(0)
        style = style.unsqueeze(0)
        style_f = self.vgg(style)
        # content = content.to(self.device)
        with torch.no_grad():
            output = self.style_transfer(style_f, content)
        #output = output.cpu().detach().numpy()
        #output = np.transpose(output[0], (1, 2, 0))
        return output[0]