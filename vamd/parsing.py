from argparse import ArgumentParser
import subprocess, time, os


def parse_train_args():
    parser = ArgumentParser()

    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    
    ## Epoch settings
    group = parser.add_argument_group("Epoch settings")
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--batch_size", type=int, default=8)
    group.add_argument("--val_freq", type=int, default=None)
    group.add_argument("--val_epoch_freq", type=int, default=1)
    group.add_argument("--no_validate", action='store_true')
    
    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=1)
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--run_name", type=str, default="default")

    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=1)
    group.add_argument("--grad_clip", type=float, default=1.)
    group.add_argument("--check_grad", action='store_true')
    group.add_argument('--grad_checkpointing', action='store_true')
    group.add_argument('--adamW', action='store_true')
    group.add_argument('--ema', action='store_true')
    group.add_argument('--ema_decay', type=float, default=0.999)
    group.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    os.environ["WANDB_LOGGING"] = str(int(args.wandb))
    if args.wandb:
        if subprocess.check_output(["git", "status", "-s"]):
            print("There were uncommited changes"); exit()
    args.commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

    return args


