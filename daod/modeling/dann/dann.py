"""
Domain disciminators.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Image discriminator (used in adaptive teacher)
class FCDiscriminator_img(nn.Module):
    def __init__(self, in_channels, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


# Gradient scalar layer
class GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        # store context for backprop
        ctx.alpha = alpha
        # forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # backward pass is just to -alpha the gradient
        output = grad_output.clone() * ctx.alpha

        # Must return same number as inputs to forward()
        return output, None


def gradient_scalar(x, alpha=1.0):
    return GradientScalarLayer.apply(x, alpha)


class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels, levels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            levels (List[str]): feature levels
        """
        super(DAImgHead, self).__init__()

        self.da_img_conv1_layers = []
        self.da_img_conv2_layers = []
        for level in levels:
            conv1_block = "DC_img_conv1_level_{}".format(level)
            conv2_block = "DC_img_conv2_level_{}".format(level)
            conv1_block_module = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
            conv2_block_module = nn.Conv2d(512, 1, kernel_size=1, stride=1)
            for module in [conv1_block_module, conv2_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                torch.nn.init.normal_(module.weight, std=0.001)
                torch.nn.init.constant_(module.bias, 0)
            self.add_module(conv1_block, conv1_block_module)
            self.add_module(conv2_block, conv2_block_module)
            self.da_img_conv1_layers.append(conv1_block)
            self.da_img_conv2_layers.append(conv2_block)

    def forward(self, x):
        img_features = {}

        for level, conv1_block, conv2_block in zip(
            x, self.da_img_conv1_layers, self.da_img_conv2_layers
        ):
            inner_lateral = getattr(self, conv1_block)(x[level])
            last_inner = F.relu(inner_lateral)
            img_features[level] = getattr(self, conv2_block)(last_inner)

        return img_features


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels, levels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            levels (List[str]): feature levels
        """
        super(DAInsHead, self).__init__()

        self.da_ins_fc1_layers = []
        self.da_ins_fc2_layers = []
        self.da_ins_fc3_layers = []

        for level in levels:
            fc1_block = "da_ins_fc1_level_{}".format(level)
            fc2_block = "da_ins_fc2_level_{}".format(level)
            fc3_block = "da_ins_fc3_level_{}".format(level)
            fc1_block_module = nn.Linear(in_channels, 1024)
            fc2_block_module = nn.Linear(1024, 1024)
            fc3_block_module = nn.Linear(1024, 1)
            for module in [fc1_block_module, fc2_block_module, fc3_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)
            self.add_module(fc1_block, fc1_block_module)
            self.add_module(fc2_block, fc2_block_module)
            self.add_module(fc3_block, fc3_block_module)
            self.da_ins_fc1_layers.append(fc1_block)
            self.da_ins_fc2_layers.append(fc2_block)
            self.da_ins_fc3_layers.append(fc3_block)


    def forward(self, x, levels=None):

        result = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

        for level, (fc1_da, fc2_da, fc3_da) in \
                enumerate(zip(self.da_ins_fc1_layers,
                    self.da_ins_fc2_layers, self.da_ins_fc3_layers)):

            idx_in_level = torch.nonzero(levels == level).squeeze(1)

            if len(idx_in_level) > 0:
                xs = x[idx_in_level, :]

                xs = F.relu(getattr(self, fc1_da)(xs))
                xs = F.dropout(xs, p=0.5, training=self.training)

                xs = F.relu(getattr(self, fc2_da)(xs))
                xs = F.dropout(xs, p=0.5, training=self.training)

                result[idx_in_level] = getattr(self, fc3_da)(xs)

        return result
