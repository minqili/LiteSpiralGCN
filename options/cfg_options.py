import argparse


class CFGOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        # parser.add_argument('--rank', type=int, default=0)
        parser.add_argument('--exp_name', type=str, default='test')
        parser.add_argument('--config_file', type=str, default='LSG/configs/LiteSpiralGCN.yml')
        parser.add_argument('--opts', type=str, nargs='+', default=[]) # default=['TRAIN.LR', 0.1]
        parser.add_argument('--PHASE', type=str, default='demo')
        parser.add_argument('--Local_testing', action='store_true')

        self.initialized = True
        return parser

    def str2bool(self, v):
        return v.lower() in ("yes", "true", "t", "1")

    def parse(self):

        parser = argparse.ArgumentParser(description='mesh generator')
        self.initialize(parser)
        args = parser.parse_args()

        return args
