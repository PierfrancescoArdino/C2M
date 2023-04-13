from torch.autograd import Variable as Vb
import torch.nn as nn
from utils.utils import *
from modules.layers.vgg import Vgg19
from torchvision import transforms as trn
from utils.ops import resample


preprocess = trn.Compose([
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mean = Vb(torch.FloatTensor([0.485, 0.456, 0.406])).view([1, 3, 1, 1])
std = Vb(torch.FloatTensor([0.229, 0.224, 0.225])).view([1, 3, 1, 1])


def normalize(x):
    x = (x+1) / 2
    gpu_id = x.get_device()
    return (x - mean.cuda(gpu_id))/std.cuda(gpu_id)


class PerceptualLoss(nn.Module):
    def __init__(self, train_params):
        super(PerceptualLoss, self).__init__()
        self.vgg19 = Vgg19()
        self.train_params = train_params
        self.criterion = nn.L1Loss()


    @staticmethod
    def compute_gram(x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_t = f.transpose(1, 2)
        g = f.bmm(f_t) / (h * w * ch)

        return g

    def forward(self, gt, fake):
        out_dict = {}
        content_loss = 0.0
        style_loss = 0.0
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        for i in range(self.train_params["num_predicted_frames"]):
            gt_frame = gt[:, :, i, ...]
            fake_frame = fake[:, :, i, ...]
            x_vgg, y_vgg = self.vgg19(gt_frame), self.vgg19(fake_frame)

            if self.train_params["loss_weights"].get("style", 0) > 0:
                style_loss += \
                    self.criterion(self.compute_gram(x_vgg['relu2_2'].detach()), self.compute_gram(y_vgg['relu2_2']))
                style_loss += \
                    self.criterion(self.compute_gram(x_vgg['relu3_4'].detach()), self.compute_gram(y_vgg['relu3_4']))
                style_loss += \
                    self.criterion(self.compute_gram(x_vgg['relu4_4'].detach()), self.compute_gram(y_vgg['relu4_4']))
                style_loss += \
                    self.criterion(self.compute_gram(x_vgg['relu5_2'].detach()), self.compute_gram(y_vgg['relu5_2']))

            if self.train_params["loss_weights"].get("perceptual", 0) > 0:
                content_loss += weights[0] * self.criterion(x_vgg['relu1_1'].detach(), y_vgg['relu1_1'])
                content_loss += weights[1] * self.criterion(x_vgg['relu2_1'].detach(), y_vgg['relu2_1'])
                content_loss += weights[2] * self.criterion(x_vgg['relu3_1'].detach(), y_vgg['relu3_1'])
                content_loss += weights[3] * self.criterion(x_vgg['relu4_1'].detach(), y_vgg['relu4_1'])
                content_loss += weights[4] * self.criterion(x_vgg['relu5_1'].detach(), y_vgg['relu5_1'])
        if content_loss > 0:
            out_dict["perceptual"] = content_loss / self.train_params["num_predicted_frames"]
        if style_loss > 0:
            out_dict["style"] = style_loss / self.train_params["num_predicted_frames"]
        return out_dict


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    @staticmethod
    def gradient_x(img):
        gx = (img[:, :, :-1, :] - img[:, :, 1:, :])
        return gx

    @staticmethod
    def gradient_y(img):
        gy = (img[:, :, :, :-1] - img[:, :, :, 1:])
        return gy

    def compute_smooth_loss(self, flow_x, img):
        flow_gradients_x = self.gradient_x(flow_x)
        flow_gradients_y = self.gradient_y(flow_x)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, True))

        smoothness_x = flow_gradients_x * weights_x
        smoothness_y = flow_gradients_y * weights_y

        return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

    def compute_flow_smooth_loss(self, flow, img):
        smoothness = 0
        _, c, h, w = flow.size()
        # flow = torch.cat([flow[:, 0:1, :, :] * ((w - 1.0) / 2.0), flow[:, 1:2, :, :] * ((h - 1.0) / 2.0)], dim=1)
        for i in range(2):
            smoothness += self.compute_smooth_loss(flow[:, i:i+1, :, :], img)
        return smoothness/2

    def forward(self, flow, image):
        return self.compute_flow_smooth_loss(torch.cat(torch.unbind(flow, dim=2), dim=0),
                                             torch.cat(torch.unbind(image, dim=2), dim=0))


class FlowConsistLoss(nn.Module):

    def __init__(self, train_params):
        super(FlowConsistLoss, self).__init__()
        self.train_params = train_params

    @staticmethod
    def _flowconsist(flow, flowback, mask_fw=None, mask_bw=None):
        if mask_fw is not None:
            nextloss = (mask_fw * torch.abs(resample(flowback, flow) + flow)).mean()
            prevloss = (mask_bw * torch.abs(resample(flow, flowback) + flowback)).mean()
        else:
            nextloss = torch.abs(resample(flowback, flow) + flow).mean()
            prevloss = torch.abs(resample(flow, flowback) + flowback).mean()
        return prevloss + nextloss

    def forward(self, flow, flowback, mask_fw=None, mask_bw=None):
        if mask_bw is not None:
            flowcon = self._flowconsist(torch.cat(torch.unbind(flow, dim=2), dim=0),
                                        torch.cat(torch.unbind(flowback, dim=2), dim=0),
                                        mask_fw=torch.cat(torch.unbind(mask_fw, dim=2), dim=0),
                                        mask_bw=torch.cat(torch.unbind(mask_bw, dim=2), dim=0))
        else:
            flowcon = self._flowconsist(torch.cat(torch.unbind(flow, dim=2), dim=0),
                                        torch.cat(torch.unbind(flowback, dim=2), dim=0))
        return flowcon * self.train_params["num_predicted_frames"]


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD / mu.numel()


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    @staticmethod
    def ssim(x, y):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

        ssim = ssim_n / ssim_d

        return torch.clamp((1 - ssim) / 2, 0, 1).mean()

    def forward(self, x, y):
        sim = self.ssim(torch.cat(torch.unbind(x, dim=2), dim=0), torch.cat(torch.unbind(y, dim=2), dim=0))
        return sim


class L1MaskedLoss(nn.Module):
    def __init__(self):
        super(L1MaskedLoss, self).__init__()

    def forward(self, source, target, mask=None):
        if mask is not None:
            mask = mask.expand_as(source)
            return F.l1_loss(source * mask, target * mask)
        else:
            return F.l1_loss(source, target)


class TrainingLosses(nn.Module):
    def __init__(self, train_params, model_params):
        super(TrainingLosses, self).__init__()
        self.train_params = train_params
        self.model_params = model_params
        if self.train_params["loss_weights"]["perceptual"] > 0:
            self.perceptual_loss = PerceptualLoss(train_params)
        self.flow_consist = FlowConsistLoss(self.train_params)
        self.smooth_loss = SmoothLoss()
        self.kl_loss = KLLoss()
        self.ssim_loss = SSIMLoss()
        self.l1_masked_loss = L1MaskedLoss()

    def forward(self, data, frames, bw_optical_flows, fw_optical_flows, bw_occlusion_masks, fw_occlusion_masks,
                generated, tracking_gnn):
        loss_dict = {}
        source_frame = frames[:, :, self.train_params["num_input_frames"] - 1, ...]
        target_frames = frames[:, :, self.train_params["num_input_frames"]:, ...]
        '''flowloss'''

        loss_dict["flow_reconstruction"] = self.l1_masked_loss(generated["dense_motion_bw"], bw_optical_flows,
                                                               bw_occlusion_masks)
        if fw_optical_flows is not None:
            loss_dict["flow_reconstruction"] += self.l1_masked_loss(generated["dense_motion_fw"],
                                                                    fw_optical_flows, fw_occlusion_masks)
            loss_dict["flowcon"] = self.flow_consist(generated["dense_motion_fw"], generated["dense_motion_bw"],
                                                     generated["occlusion_fw"], generated["occlusion_bw"])
        warped_frames = torch.cat([torch.unsqueeze(resample(source_frame,
                                                            generated["dense_motion_bw"][:, :, i, ...]),
                                                   2) for i in range(self.train_params["num_predicted_frames"])], 2)
        loss_dict["warped"] = self.l1_masked_loss(warped_frames, target_frames)
        if self.train_params["loss_weights"]["flow_smooth"] > 0:
            loss_dict["flow_smooth"] = self.smooth_loss(generated["dense_motion_bw"], target_frames)
            if fw_optical_flows is not None:
                loss_dict["flow_smooth"] += self.smooth_loss(generated["dense_motion_fw"],
                                                   torch.unsqueeze(source_frame, 2).repeat(1, 1, target_frames.shape[2],
                                                                                           1, 1))
        '''kldloss'''
        loss_dict["kl"] = self.kl_loss(generated["mu"], generated["logvar"])
        '''Image Similarity loss'''
        loss_dict["ssim"] = self.ssim_loss(generated["generated"], target_frames)
        loss_dict["reconstruction"] = self.l1_masked_loss(generated["generated"], target_frames)
        '''vgg loss'''
        loss_dict.update(self.perceptual_loss(target_frames,
                                              generated["generated"]))
        '''mask loss'''
        loss_dict["occlusion_bw"] = self.l1_masked_loss(bw_occlusion_masks, generated["occlusion_bw"])
        if fw_optical_flows is not None:
            loss_dict["occlusion_fw"] = self.l1_masked_loss(fw_occlusion_masks, generated["occlusion_fw"])
        scale = 0
        rotation = 0
        translation = 0
        for t in range(self.train_params["num_predicted_frames"]):
            translation += self.l1_masked_loss(generated[f"theta_{t}"][:, 2], tracking_gnn.targets_theta[:, t][:, 2])
            translation += self.l1_masked_loss(generated[f"theta_{t}"][:, 5], tracking_gnn.targets_theta[:, t][:, 5])
            scale += self.l1_masked_loss(generated[f"theta_{t}"][:, 0], tracking_gnn.targets_theta[:, t][:, 0])
            scale += self.l1_masked_loss(generated[f"theta_{t}"][:, 4], tracking_gnn.targets_theta[:, t][:, 4])
            rotation += self.l1_masked_loss(generated[f"theta_{t}"][:, 1], tracking_gnn.targets_theta[:, t][:, 1])
            rotation += self.l1_masked_loss(generated[f"theta_{t}"][:, 3], tracking_gnn.targets_theta[:, t][:, 3])
        loss_dict["translation"] = loss_dict.get("translation", 0) + isnan(translation)
        loss_dict["scale"] = loss_dict.get("scale", 0) + isnan(scale)
        loss_dict["rotation"] = loss_dict.get("rotation", 0) + isnan(rotation)

        return loss_dict
