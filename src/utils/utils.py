from __future__ import division
import os
import numpy as np
import imageio
from utils import ops
import torch.nn.functional as F
import torch
import torch.distributed as dist
import random
import torch.backends.cudnn as cudnn
import requests
import cv2


def save_gif(images, length, size, gifpath):
    num_images = size[0] * size[1]
    images = np.array(images[0:num_images])
    savegif = [np.uint8(merge(images[:, times, ...], size)) for times in range(0, length)]
    imageio.mimsave(gifpath, savegif, fps=int(length))


def save_image(image_numpy, image_path):
    imageio.mimsave(image_path, image_numpy, fps=int(len(image_numpy)))


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


def tensor2im(image_tensor, imtype=np.uint8, normalize=False, size=None):
    if size is None:
        size = [8, 4]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        # -1 1 -> 0 - 2 -> 0 - 1
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 4, 1)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (0, 2, 3, 4, 1)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    num_images = image_numpy.shape[0]
    images = np.array(image_numpy[0:num_images])
    savegif = [np.uint8(merge(images[:, times, ...], size)) for times in range(0, image_numpy.shape[1])]
    return np.array(savegif).astype(imtype)


def tensor2occ(image_tensor, imtype=np.uint8, size=None):
    if size is None:
        size = [8, 4]
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = image_numpy * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = np.transpose(image_numpy, (0, 2, 3, 4, 1))
    num_images = image_numpy.shape[0]
    images = np.array(image_numpy[0:num_images])
    savegif = np.stack([np.uint8(merge(images[:, times, ...], size)) for times in range(0, image_numpy.shape[1])], 0)
    return np.array(savegif).astype(imtype)


def tensor2flow(output, size):
    output = output.cpu().float()
    output = output.permute(0, 2, 3, 4, 1).numpy()
    flow_to_save =\
        np.stack([np.uint8(ops.compute_flow_img(output[:, i, ...], None, size)) for i in range(output.shape[1])], 0)
    return flow_to_save


def save_samples(video, generated_video, dense_bw_of, dense_fw_of, sparse_bw_of, sparse_fw_of, sparse_bin, bw_occ,
                 fw_occ, target_bw_of, target_fw_of, target_bw_occ, target_fw_occ,
                 iteration, sampledir, opt, is_eval=False, use_mask=True, grid=None, sample_number=0):

    if grid is None:
        grid = [8, 4]
    source_frames = video[:, :, :opt["test_params"]["num_input_frames"], ...].contiguous()
    num_predicted_frames = generated_video.size()[2]
    num_frames = video.size()[2]
    if is_eval:
        save_file_name = 'sample'
    else:
        save_file_name = 'recon'

    if sparse_bin is not None:
        save_gif(sparse_bin.permute(0, 2, 3, 4, 1).data.cpu().numpy() * 255., num_predicted_frames, grid, sampledir +
                 '/{:06d}_{:02d}_%s_sparse_bin.gif'.format(iteration, sample_number) % save_file_name)
    if use_mask:
        save_gif(bw_occ.permute(0, 2, 3, 4, 1).data.cpu().numpy() * 255., num_predicted_frames, grid, sampledir +
                 '/{:06d}_{:02d}_%s_bw_occ.gif'.format(iteration, sample_number) % save_file_name)
        save_gif(target_bw_occ.permute(0, 2, 3, 4, 1).data.cpu().numpy() * 255., num_predicted_frames, grid,
                 sampledir +
                 '/{:06d}_%s_target_bw_occ.gif'.format(iteration) % save_file_name)
        save_gif(fw_occ.permute(0, 2, 3, 4, 1).data.cpu().numpy() * 255., num_predicted_frames, grid, sampledir +
                 '/{:06d}_{:02d}_%s_fw_occ.gif'.format(iteration, sample_number) % save_file_name)
        save_gif(target_fw_occ.permute(0, 2, 3, 4, 1).data.cpu().numpy() * 255., num_predicted_frames, grid,
                 sampledir +
                 '/{:06d}_%s_target_fw_occ.gif'.format(iteration) % save_file_name)

    # Save reconstructed or sampled video
    fakegif = torch.cat([source_frames, generated_video], 2)
    fakegif = fakegif.permute(0, 2, 3, 4, 1).data.cpu().numpy()
    dense_bw_of = dense_bw_of.permute(0, 2, 3, 4, 1).cpu().data.numpy()

    target_bw_of = target_bw_of.permute(0, 2, 3, 4, 1).cpu().data.numpy()
    dense_fw_of = dense_fw_of.permute(0, 2, 3, 4, 1).cpu().data.numpy()

    target_fw_of = target_fw_of.permute(0, 2, 3, 4, 1).cpu().data.numpy()

    if eval:
        save_file_name = 'sample'
    else:
        save_file_name = 'recon'
        # Save ground truth sample
    video = video.cpu().data.permute(0, 2, 3, 4, 1).numpy()
    save_gif(video * 255, num_frames, [8, 4], sampledir + '/{:06d}_%s_gt.gif'.format(iteration) % save_file_name)

    save_gif(fakegif * 255, num_frames, grid,
             sampledir + '/{:06d}_{:02d}_%s.gif'.format(iteration, sample_number) % save_file_name)
    # ops.saveflow(_flow, opt.input_size, grid, sampledir + '/{:06d}_%s_flow.jpg'.format(iteration)%save_file_name)
    ops.save_flow_sequence(dense_fw_of, num_predicted_frames, opt["test_params"]["input_size"], grid,
                           sampledir + '/{:06d}_{:02d}_%s_dense_fw_of.gif'.format(iteration,
                                                                                  sample_number) % save_file_name)
    ops.save_flow_sequence(target_fw_of, num_predicted_frames, opt["test_params"]["input_size"], grid,
                           sampledir + '/{:06d}_%s_target_fw_of.gif'.format(iteration) % save_file_name)
    ops.save_flow_sequence(dense_bw_of, num_predicted_frames, opt["test_params"]["input_size"], grid,
                           sampledir + '/{:06d}_{:02d}_%s_dense_bw_of.gif'.format(iteration,
                                                                                  sample_number) % save_file_name)
    ops.save_flow_sequence(target_bw_of, num_predicted_frames, opt["test_params"]["input_size"], grid,
                           sampledir + '/{:06d}_%s_target_bw_of.gif'.format(iteration) % save_file_name)
    if sparse_bw_of is not None:
        sparse_bw_of = sparse_bw_of.permute(0, 2, 3, 4, 1).cpu().data.numpy()
        ops.save_flow_sequence(sparse_bw_of, num_predicted_frames, opt["train_params"]["input_size"], grid,
                               sampledir + '/{:06d}_{:02d}_%s_sparse_bw_of.gif'.format(iteration,
                                                                                       sample_number) % save_file_name)
        sparse_fw_of = sparse_fw_of.permute(0, 2, 3, 4, 1).cpu().data.numpy()
        ops.save_flow_sequence(sparse_fw_of, num_predicted_frames, opt["test_params"]["input_size"], grid,
                               sampledir + '/{:06d}_{:02d}_%s_sparse_fw_of.gif'.format(iteration,
                                                                                       sample_number) % save_file_name)


def save_parameters(parameterdir, jobname, opt, load, iter_to_load):
    """
    Write parameters setting file
    """
    with open(os.path.join(parameterdir, 'params.txt'), 'w') as file:
        file.write(jobname)
        file.write('Training Parameters: \n')
        file.write(str(opt) + '\n')
        if load:
            file.write('Load pretrained model: ' + str(load) + '\n')
            file.write('Iteration to load:' + str(iter_to_load) + '\n')


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def dist_all_reduce_tensor(tensor, reduce="mean"):
    r""" Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        if reduce == 'mean':
            tensor /= world_size
        elif reduce == 'sum':
            pass
        else:
            raise NotImplementedError
    return tensor


def dist_all_gather_tensor(tensor):
    r""" gather to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor_list = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)

    return torch.cat(tensor_list, dim=0)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_images(root_dir, data, y_pred, paths, sample_number=0):
    data = data
    frame1 = data[:, :, 0, :, :]
    frame1_ = torch.unsqueeze(frame1, 2)
    frame_sequence = torch.cat([frame1_, y_pred], 2)
    frame_sequence = frame_sequence.permute((0, 2, 3, 4, 1)).cpu().data.numpy() * 255

    for i in range(y_pred.size()[0]):

        #  save images as gif
        frames_fo_save = [np.uint8(frame_sequence[i][frame_id]) for frame_id in range(frame_sequence.shape[1])]
        # 3fps
        aux_dir = os.path.join(root_dir, "/".join(paths[0][i].split("/")[:-1]))
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)

        imageio.mimsave(os.path.join(root_dir, paths[0][i][0:-4] + '_{:02d}.gif'.format(sample_number)),
                        frames_fo_save, fps=int(frame_sequence.shape[1]))

        # new added

        for j, frame in enumerate(frames_fo_save):
            cv2.imwrite(os.path.join(root_dir, paths[0][i][0:-4] + '{:02d}.png'.format(j)),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def draw_bbox(image, bbox):
    import cv2
    x1, x2, y1, y2 = bbox

    start_point = (int(x1), int(y1))
    end_point = (int(x2), int(y2))
    color = (255, 0, 0)
    img = image.copy()
    image = cv2.rectangle(img, start_point, end_point, color, 1, cv2.LINE_8)
    return image


def save_images_w_bbox(root_dir, data, y_pred, paths, opt, tracking_gnn, sample_number=0):
    import cv2
    data = data
    frame1 = data[:, :, 0, :, :]
    frame1_ = torch.unsqueeze(frame1, 2)
    frame_sequence = torch.cat([frame1_, y_pred], 2)
    frame_sequence = frame_sequence.permute((0, 2, 3, 4, 1)).cpu().data.numpy() * 255
    for inst_id, batch_id, bbox_start in zip(tracking_gnn.source_frames_nodes_instance_ids[:, -1].long(),
                                             tracking_gnn.batch.long(),
                                             tracking_gnn.source_frames_nodes_roi[:, -1]):
        if inst_id == 0:
            continue
        bbox_width = int(bbox_start[1]) - int(bbox_start[0])
        bbox_height = int(bbox_start[3]) - int(bbox_start[2])
        if bbox_height * bbox_width < 0.001 * opt["test_params"]["input_size"][1] * opt["test_params"]["input_size"][0]:
            continue
        #  save images as gif
        frames_fo_save = [np.uint8(draw_bbox(frame_sequence[batch_id][frame_id],
                                             bbox_start)) for frame_id in range(frame_sequence.shape[1])]
        # 3fps
        aux_dir = os.path.join(root_dir, "/".join(paths[0][batch_id].split("/")[:-1]))
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)
        imageio.mimsave(os.path.join(root_dir,
                                     paths[0][batch_id][0:-4] + f'_{inst_id}_' + '_{:02d}.gif'.format(sample_number)),
                        frames_fo_save, fps=int(frame_sequence.shape[1]))
        for j, frame in enumerate(frames_fo_save):
            cv2.imwrite(os.path.join(root_dir, paths[0][batch_id][0:-4] + f'_{inst_id}_' + '{:02d}.png'.format(j)),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def save_flows(root_dir, flow, paths, sample_number=0):
    # print(flow.size())
    _flow = flow.permute(0, 2, 3, 4, 1)
    _flow = _flow.cpu().data.numpy()
    # mask  = mask.unsqueeze(4)
    # # print (mask.size())
    # mask = mask.data.cpu().numpy() * 255.

    for i in range(flow.size()[0]):
        aux_dir = os.path.join(root_dir, "/".join(paths[0][i].split("/")[:-1]))
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)
        # save flow*mask as gif
        # *mask[i][frame_id])
        flow_fo_save = [np.uint8(ops.compute_flow_color_map(_flow[i][frame_id])) for frame_id in range(flow.size()[2])]
        # 3fps
        imageio.mimsave(os.path.join(root_dir,
                                     paths[0][i][0:-4] + '_{:02d}.gif'.format(sample_number)),
                        flow_fo_save, fps=int(flow.size()[2]))

        for j in range(flow.size()[2]):
            ops.saveflow(_flow[i][j],
                         (256, 128), os.path.join(root_dir, paths[j+1][i][0:-4] + '_{:02d}.png'.format(sample_number)))


def save_occ_map(root_dir, mask, paths, sample_number=0):
    mask = mask.permute(0, 2, 3, 4, 1).data.cpu().numpy() * 255.
    for i in range(mask.shape[0]):
        aux_dir = os.path.join(root_dir, "/".join(paths[0][i].split("/")[:-1]))
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)
        imageio.mimsave(os.path.join(root_dir, paths[0][i][0:-4] + '_{:02d}.gif'.format(sample_number)),
                        [np.uint8(mask[i][frame_id]) for frame_id in range(mask.shape[1])], fps=int(mask.shape[1]))
        for j in range(mask.shape[1]):
            cv2.imwrite(os.path.join(root_dir, paths[j+1][i][0:-4] + '_{:02d}.png'.format(sample_number)), mask[i][j])


def read_flow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # Reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def resize_flow(flow, new_shape):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                           mode='bilinear', align_corners=True)
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow


def resize_video(video, scale_factor, mode="nearest", is_flow=False):
    if video is not None:
        if is_flow:
            _, _, _, h, w = video.shape
            h *= scale_factor
            w *= scale_factor
            resized_video = resize_flow(torch.cat(torch.unbind(video, 2), 0), [h, w])
        else:
            if type(scale_factor) == list:
                resized_video = F.interpolate(torch.cat(torch.unbind(video, 2), 0), size=scale_factor, mode=mode)
            else:
                resized_video = F.interpolate(torch.cat(torch.unbind(video, 2), 0), scale_factor=scale_factor,
                                              mode=mode)
        return torch.cat(torch.chunk(resized_video.unsqueeze(2), video.shape[2], 0), 2)
    else:
        return None


def isnan(x, input_tensor=None):
    if torch.any(torch.isnan(x)):
        raise ValueError(f"Value is nan {x}, input tensor is {input_tensor}")
    else:
        return x


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def set_random_seed(seed, by_rank=False):
    r"""Set random seeds for everything.
    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    if by_rank:
        seed += get_rank()
    print(f"Using random seed {seed} rank: {get_rank()}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_cudnn(deterministic, benchmark):
    r"""Initialize the cudnn module. The two things to consider is whether to
    use cudnn benchmark and whether to use cudnn deterministic. If cudnn
    benchmark is set, then the cudnn deterministic is automatically false.
    Args:
        deterministic (bool): Whether to use cudnn deterministic.
        benchmark (bool): Whether to use cudnn benchmark.
    """
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark
    print('cudnn benchmark: {}'.format(benchmark))
    print('cudnn deterministic: {}'.format(deterministic))


def download_file(url, destination):
    r"""Download a file from Google Drive or pbss by using the url.

    Args:
        url: GDrive URL or PBSS pre-signed URL for the checkpoint.
        destination: Path to save the file.

    Returns:

    """
    session = requests.Session()
    response = session.get(url, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    r"""Get confirm token

    Args:
        response: Check if the file exists.

    Returns:

    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    r"""Save response content

    Args:
        response:
        destination: Path to save the file.

    Returns:

    """
    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def get_checkpoint(checkpoint_path, url=''):
    r"""Get the checkpoint path. If it does not exist yet, download it from
    the url.

    Args:
        checkpoint_path (str): Checkpoint path.
        url (str): URL to download checkpoint.
    Returns:
        (str): Full checkpoint path.
    """
    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = os.getcwd()
    save_dir = os.path.join(os.environ['TORCH_HOME'], 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    full_checkpoint_path = os.path.join(save_dir, checkpoint_path)
    if not os.path.exists(full_checkpoint_path):
        os.makedirs(os.path.dirname(full_checkpoint_path), exist_ok=True)
        if is_master():
            print('Downloading {}'.format(url))
            if 'pbss.s8k.io' not in url:
                url = f"https://docs.google.com/uc?export=download&id={url}"
            download_file(url, full_checkpoint_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return full_checkpoint_path
