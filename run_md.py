from vamd.parsing import parse_train_args
args = parse_train_args()
from vamd.logger import get_logger
logger = get_logger(__name__)

import torch, wandb, os
import pytorch_lightning as pl
from vamd.runner import VAMDRunner

torch.set_float32_matmul_precision('medium')

runner = VAMDRunner(args)
runner.run()