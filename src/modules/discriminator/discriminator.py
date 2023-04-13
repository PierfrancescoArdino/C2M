import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.layers.down_block import DownBlock2d


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_d(input_nc, ndf, n_layers_d, num_d=1, padding_mode="zeros"):
    net_d = MultiScaleDiscriminator(input_nc, ndf, n_layers_d, num_d, padding_mode)
    net_d.apply(weights_init)
    return net_d


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, input_nc, ndf, n_layers_d, num_d, padding_mode):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = num_d
        discs = {}
        for scale in range(self.scales):
            discs[str(scale).replace('.', '-')] = Discriminator(num_channels=input_nc, block_expansion=ndf,
                                                                num_blocks=n_layers_d, padding_mode=padding_mode)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            feature_maps, prediction_map = disc(x)
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=True, padding_mode="zeros"):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            kernel_size=4, stride=2, padding=1,
                            padding_mode=padding_mode, use_norm=True))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        feature_maps = []
        out = x
        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map

        ##############################################################################


# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input_tensor, target_is_real):
        gpu_id = input_tensor.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input_tensor.numel()))
            if create_label:
                real_tensor = self.Tensor(input_tensor.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input_tensor.numel()))
            if create_label:
                fake_tensor = self.Tensor(input_tensor.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input_tensor, target_is_real):
        if isinstance(input_tensor[0], list):
            loss = 0
            for input_i in input_tensor:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input_tensor[-1], target_is_real)
            return self.loss(input_tensor[-1], target_tensor)
