import os
os.environ['TRANSFORMERS_CACHE'] = '/data/tungtx2/tmp/transformers_hub'
import json
from pathlib import Path
import torch
import evaluate
import numpy as np
import yaml
import shutil
from lightning.pytorch import Trainer
from lightning.pytorch.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor, RichModelSummary
from lightning.pytorch.cli import LightningCLI
from lmv3 import LMv3ClassifierModule
from lilt import LiltClassifierModule
from dataset import VATDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--save_config_overwrite", default=False)
        parser.add_argument("--save_config_callback", default=None)


    def before_instantiate_classes(self) -> None:
        self.save_config_kwargs['overwrite'] = self.config[self.config.subcommand].save_config_overwrite
        if self.config[self.config.subcommand].save_config_callback == 'None':
            self.save_config_callback = None



def cli_main():
    cli = MyLightningCLI(
        LMv3ClassifierModule, 
        VATDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_overwrite=False
    )


if __name__ == '__main__':
    cli_main()
