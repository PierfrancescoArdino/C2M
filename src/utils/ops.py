from __future__ import division
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as transform
import imageio

input_transform = transform.Compose([
    transform.ToTensor()])

RY = 15
YG = 6
GC = 4
CB = 11
BM = 13
MR = 6
UNKNOW_FLOW_THRESHOLD = 1e7


def make_color_wheel():
    r"""Generate color wheel according Middlebury color code
    :return: Color wheel
    """

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def merge(images, size):
    cdim = images.shape[-1]
    h, w = images.shape[1], images.shape[2]
    if cdim == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = np.squeeze(image)
        return img
    else:
        img = np.zeros((h * size[0], w * size[1], cdim))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        # print img.shape
        return img


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nan_idx = np.isnan(u) | np.isnan(v)
    u[nan_idx] = 0
    v[nan_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nan_idx)))

    return img


def saveflow(flows, imgsize, savepath):
    u = flows[:, :, 0]*3
    v = flows[:, :, 1]*3
    image = compute_color(u, v)
    flow = cv2.resize(image, imgsize)
    cv2.imwrite(savepath, flow)


def compute_flow_color_map(flows):
    u = flows[:, :, 0] * 3
    v = flows[:, :, 1] * 3
    flow = compute_color(u, v)
    return flow


def compute_flow_img(flows, imgsize, size):
    num_images = size[0] * size[1]
    flows = merge(flows[0:num_images], size)
    image = flow2img(flows)
    return image


def flow2img(flow_data):
    """
    convert optical flow into color image
    :param flow_data:
    :return: color image
    """
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def save_flow_sequence(flows, length, imgsize, size, savepath):
    flow_seq = [np.uint8(compute_flow_img(flows[:, i, ...], imgsize, size)) for i in range(length)]
    imageio.mimsave(savepath, flow_seq, fps=int(length))


def grid_sample(input1, input2, mode='bilinear'):
    return torch.nn.functional.grid_sample(input1, input2, mode=mode, padding_mode='border')


def resample(image, flow, mode='bilinear'):
    b, c, h, w = image.size()
    grid = get_grid(b, h, w, gpu_id=flow.get_device())
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    final_grid = (grid + flow).permute(0, 2, 3, 1)
    output = grid_sample(image, final_grid, mode)
    return output


def get_grid(batchsize, rows, cols, gpu_id=0):
    base_grid = torch.zeros([batchsize, 2, rows, cols])
    linear_points = torch.linspace(-1, 1, cols) if cols > 1 else torch.Tensor([-1])
    base_grid[:, 0, :, :] = torch.ger(torch.ones(rows), linear_points).expand_as(base_grid[:, 0, :, :])
    linear_points = torch.linspace(-1, 1, rows) if rows > 1 else torch.Tensor([-1])
    base_grid[:, 1, :, :] = torch.ger(linear_points, torch.ones(cols)).expand_as(base_grid[:, 1, :, :])
    return base_grid.cuda(gpu_id)


def get_corresponding_map(data):
    """
    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    b, _, h, w = data.size()

    x = data[:, 0, :, :].view(b, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(b, -1)

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, w - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, h - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, w - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, h - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(b, h * w).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * w,
                         x_ceil + y_floor * w,
                         x_floor + y_ceil * w,
                         x_floor + y_floor * w], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(b, h, w)

    return corresponding_map.unsqueeze(1)


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def get_occlusion_map(flow):
    """
    :param flow: flow
    :return:
    """
    b, _, h, w = flow.size()

    base_grid = mesh_grid(b, h, w).type_as(flow)  # B2HW
    with torch.no_grad():
        # apply backward flow to get occlusion map
        corr_map = get_corresponding_map(base_grid + flow)  # BHW
    # soft mask, 0 means occlusion, 1 means empty
    return corr_map.clamp(min=0., max=1.)


def get_edges(instance):
    edge = torch.ByteTensor(instance.size()).zero_().to(instance.device)
    edge[:, :, :, :, 1:] = edge[:, :, :, :, 1:] | (instance[:, :, :, :, 1:] != instance[:, :, :, :, :-1])
    edge[:, :, :, :, :-1] = edge[:, :, :, :, :-1] | (instance[:, :, :, :, 1:] != instance[:, :, :, :, :-1])
    edge[:, :, :, 1:, :] = edge[:, :, :, 1:, :] | (instance[:, :, :, 1:, :] != instance[:, :, :, :-1, :])
    edge[:, :, :, :-1, :] = edge[:, :, :, :-1, :] | (instance[:, :, :, 1:, :] != instance[:, :, :, :-1, :])
    return edge.float()
