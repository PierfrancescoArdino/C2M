import torch
from torch.utils.data import DataLoader
import os
from modules.model import GeneratorFullModel
from modules.third_party.flow_net.flow_net import FlowNet
from datasets.dataset import get_training_set, get_test_set
from torch.nn.parallel import DistributedDataParallel
from trainer.trainer import Trainer
from torch_geometric.data import Batch, Data
from options.options import BaseOptions
from utils.utils import set_random_seed, init_cudnn
from yaml import load
from torch.profiler import schedule

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import torch.distributed as dist
import numpy as np


class BatchCollate:
    def __init__(self, data):
        data = {k: [d.get(k) for d in data] for k in sorted(set().union(*data))}
        self.data = {}
        for k, v in data.items():
            if torch.is_tensor(v[0]):
                self.data[k] = torch.stack(v, dim=0)
            elif isinstance(v[0], Data):
                self.data[k] = Batch.from_data_list(v)
            else:
                self.data[k] = v

    def pin_memory(self):
        self.data =\
            {k: v.pin_memory() if k not in ["complete_list"] else v for k, v in self.data.items()}
        return self


def collate_wrapper(batch):
    return BatchCollate(batch)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main(local_rank):
    with open(opt.config) as f:
        config = load(f, Loader=Loader)
    init_cudnn(deterministic=False, benchmark=True)

    ''' Cityscapes'''
    train_dataset = get_training_set(config)
    test_dataset = get_test_set(config)
    if len(opt.device_ids) > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, seed=opt.seed)
        val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    trainloader = DataLoader(train_dataset, batch_size=config["train_params"]["batch_size"],
                             shuffle=(train_sampler is None), num_workers=config["train_params"]["workers"],
                             pin_memory=True, drop_last=True, collate_fn=collate_wrapper, sampler=train_sampler)
    testloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=config["train_params"]["workers"],
                            pin_memory=True, drop_last=True, collate_fn=collate_wrapper,
                            sampler=val_sampler)
    set_random_seed(opt.seed, by_rank=False)
    c2m = GeneratorFullModel(train_params=config["train_params"], model_params=config["model_params"],
                             dataset=config["dataset_params"]["dataset"])
    if not config["train_params"]["use_pre_processed_of"]:
        flownet = FlowNet(pretrained=True)
        flownet.to(local_rank)
    else:
        flownet = None
    c2m.to(local_rank)
    set_random_seed(opt.seed, by_rank=True)
    if len(opt.device_ids) > 1:
        c2m = DistributedDataParallel(c2m, device_ids=[local_rank], output_device=local_rank)
        c2m = c2m.module
        if not config["train_params"]["use_pre_processed_of"]:
            flownet = DistributedDataParallel(flownet, device_ids=[local_rank], output_device=local_rank).module
        distributed = True
    else:
        distributed = False
    optimizer_vae = c2m.optimizer
    optimizer_gnn = c2m.optimizer_gnn
    scheduler_vae = c2m.scheduler_g
    scheduler_gnn = c2m.scheduler_gnn
    scheduler_d_image = c2m.scheduler_d_image if config["train_params"]["use_image_discriminator"] else None
    optimizer_d_image = c2m.d_optimizer_image if config["train_params"]["use_image_discriminator"] else None
    scheduler_d_video = c2m.scheduler_d_video if config["train_params"]["use_video_discriminator"] else None
    optimizer_d_video = c2m.d_optimizer_video if config["train_params"]["use_video_discriminator"] else None
    if distributed:
        dist.barrier()
    trainer = Trainer(config, opt, c2m, flownet, optimizer_vae, optimizer_gnn, optimizer_d_image, optimizer_d_video,
                      scheduler_vae, scheduler_gnn, scheduler_d_image, scheduler_d_video, trainloader, testloader,
                      local_rank)
    start_epoch, epoch_iter = trainer.load_checkpoint()
    trainer.initialize_deltas(start_epoch, epoch_iter)
    if opt.profile:
        profiler_path = f'{trainer.sampledir}/logs_profiler/profiler_workers'
        with torch.profiler.profile(schedule=schedule(wait=1, warmup=1, active=5),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_path),
                                    record_shapes=True, with_stack=True, profile_memory=True) as prof:
            for epoch in range(start_epoch, config["train_params"]["num_epochs"]):
                trainer.start_of_epoch(epoch)
                if local_rank == 0:
                    print('Epoch {}/{}'.format(epoch, config["train_params"]["num_epochs"] - 1))
                    print('-' * 10)
                if distributed:
                    train_sampler.set_epoch(epoch)
                for i, train_batch in enumerate(iter(trainloader)):
                    if i >= 7 and opt.profile:
                        break
                    data = trainer.start_of_iteration(train_batch)
                    trainer.update_model(data)
                    trainer.end_of_iteration(data)
                    prof.step()
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000))
                exit()
    else:
        for epoch in range(start_epoch, config["train_params"]["num_epochs"]):
            trainer.start_of_epoch(epoch)
            if local_rank == 0:
                print('Epoch {}/{}'.format(epoch, config["train_params"]["num_epochs"] - 1))
                print('-' * 10)
            if distributed:
                train_sampler.set_epoch(epoch)
            for i, train_batch in enumerate(iter(trainloader)):
                data = trainer.start_of_iteration(train_batch)
                trainer.update_model(data)
                trainer.end_of_iteration(data)
            trainer.end_of_epoch()
        if local_rank == 0:
            trainer.save_checkpoint()


if __name__ == '__main__':
    opt = BaseOptions().parse()
    set_random_seed(opt.seed, by_rank=True)
    if len(opt.device_ids) > 1:
        device_id = int(os.getenv('LOCAL_RANK', 0))
        torch.cuda.set_device(device_id)
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        print(
            f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )
        main(device_id)
        dist.destroy_process_group()
    else:
        main(int(os.getenv('LOCAL_RANK', 0)))
