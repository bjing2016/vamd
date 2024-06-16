from vamd.parsing import parse_train_args
args = parse_train_args()
from vamd.logger import get_logger
logger = get_logger(__name__)

import torch, wandb, os
from vamd.wrapper import VAMDWrapper
from vamd.dataset import VAMDDataset, VAMDValDataset
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')

if args.wandb:
    wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        settings=wandb.Settings(start_method="fork"),
        project="vamd",
        name=args.run_name,
        config=args,
    )
ds = VAMDValDataset(args)
trainset, valset = torch.utils.data.random_split(ds, [len(ds) - args.val_examples, args.val_examples])

if not args.no_md:
    trainset = VAMDDataset(args)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)
val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)

model = VAMDWrapper(args)
    
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=0.0 if args.no_validate else (args.val_batches or 1.0),
    num_sanity_val_steps=0,
    enable_progress_bar=not args.wandb,
    gradient_clip_val=args.grad_clip,
    default_root_dir=os.environ["MODEL_DIR"], 
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"], 
            save_top_k=0,
            save_last=True,
            every_n_train_steps=args.ckpt_freq,
        ),
        ModelSummary(max_depth=2),
    ],
    accumulate_grad_batches=args.accumulate_grad,
    val_check_interval=args.val_freq,
    logger=False
)

if args.validate:
    trainer.validate(model, val_loader, ckpt_path=args.ckpt)
else:
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)