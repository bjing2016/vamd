from argparse import ArgumentParser
import subprocess, time, os


def parse_train_args():
    parser = ArgumentParser()

    ## Trainer settings
    group = parser.add_argument_group("Trainer settings")
    group.add_argument("--ckpt", type=str, default=None)
    group.add_argument("--validate", action='store_true', default=False)
    group.add_argument("--num_workers", type=int, default=4)
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--batch_size", type=int, default=10)
    group.add_argument("--val_freq", type=int, default=None)
    group.add_argument("--no_validate", action='store_true')
    group.add_argument("--inference_freq", type=int, default=1)
    group.add_argument("--inference_batches", type=int, default=0)
    
    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=None)
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

    ## Model settings
    group = parser.add_argument_group("Model settings")
    group.add_argument("--embed_dim", type=int, default=256)
    group.add_argument("--mha_heads", type=int, default=16)
    group.add_argument("--num_layers", type=int, default=8)
    group.add_argument("--abs_pos_emb", type=int, default=None)
    
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="GVP", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--sampling_method", type=str, default="dopri5", choices=["dopri5", "euler"])
    # group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    
    group = parser.add_argument_group("MD arguments")
    group.add_argument("--num_samples", type=int, default=1000)
    group.add_argument("--num_steps", type=int, default=10000)
    group.add_argument("--sample_dir", type=str, default='/tmp/default')
    
    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    os.environ["WANDB_LOGGING"] = str(int(args.wandb))
    if args.wandb:
        if subprocess.check_output(["git", "status", "-s"]):
            print("There were uncommited changes"); exit()
    args.commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

    return args



