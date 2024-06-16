from vamd.parsing import parse_train_args
args = parse_train_args()
from vamd.logger import get_logger
logger = get_logger(__name__)

import torch, wandb, os
import pytorch_lightning as pl
from vamd.runner import VAMDRunner

torch.set_float32_matmul_precision('medium')

if args.wandb:
    wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        settings=wandb.Settings(start_method="fork"),
        project="vamd",
        name=args.run_name,
        config=args,
    )

runner = VAMDRunner(args)
runner.run()