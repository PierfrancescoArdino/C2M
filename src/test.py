import torch
from torch.utils.data import DataLoader
import os
from modules.model import GeneratorFullModel
from modules.third_party.flow_net.flow_net import FlowNet
from datasets.dataset import get_test_set
from evaluator.evaluator import Evaluator
from torch_geometric.data import Batch, Data
from utils.utils import set_random_seed, init_cudnn
from options.options import BaseOptions
from yaml import load
from tqdm import tqdm
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


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
                self.data[k] = list(zip(*v))

    def pin_memory(self):
        self.data =\
            {k: v.pin_memory() if k not in ["complete_list"] else v for k, v in self.data.items()}
        return self


def collate_wrapper(batch):
    return BatchCollate(batch)


def main(local_rank):
    with open(opt.config) as f:
        config = load(f, Loader=Loader)
    init_cudnn(deterministic=False, benchmark=True)
    set_random_seed(config["test_params"]["seed"], by_rank=True)
    ''' Cityscapes'''
    test_dataset = get_test_set(config)
    val_sampler = None
    testloader = DataLoader(test_dataset, batch_size=config["test_params"]["batch_size"], shuffle=False,
                            num_workers=config["test_params"]["workers"], pin_memory=True, drop_last=True,
                            collate_fn=collate_wrapper, sampler=val_sampler)
    set_random_seed(config["test_params"]["seed"], by_rank=False)
    c2m = GeneratorFullModel(train_params=config["test_params"], model_params=config["model_params"], is_inference=True,
                             dataset=config["dataset_params"]["dataset"])
    if not config["test_params"]["use_pre_processed_of"]:
        flownet = FlowNet(pretrained=True)
        flownet.to(local_rank)
    else:
        flownet = None
    c2m.to(local_rank)
    set_random_seed(config["test_params"]["seed"], by_rank=True)
    evaluator = Evaluator(config, opt, c2m, flownet, testloader, local_rank)
    evaluator.load_checkpoint()
    evaluator.set_eval()
    iteration = 0
    for i, batch in enumerate(tqdm(iter(testloader))):
        with torch.no_grad():
            batch = evaluator.start_iteration(batch.data, iteration)
            # zero the parameter gradients
            for sample_number in range(config["test_params"]["num_samples"]):
                generated_data = evaluator.evaluate(i, batch)
                evaluator.save_samples(batch, generated_data, sample_number)
                evaluator.compute_detection(batch, generated_data, sample_number)
                if sample_number == 0:
                    evaluator.fetch_metrics_data(batch, generated_data, sample_number)
            iteration += 1
    evaluator.generate_metrics()
    if config["test_params"]["save_index_user_guidance"]:
        evaluator.save_user_guidance()


if __name__ == '__main__':
    opt = BaseOptions().parse()
    main(int(os.getenv('LOCAL_RANK', 0)))
