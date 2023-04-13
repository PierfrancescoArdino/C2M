import argparse


class BaseOptions:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--config", required=True, help="path to config")
        self.parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                                 help="Names of the devices comma separated.")
        self.parser.add_argument("--seed", default=0, type=int, help="seed of the training")
        self.parser.add_argument("--profile", default=False, action='store_true', help='use_profiler')
        self.parser.set_defaults(verbose=False)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
