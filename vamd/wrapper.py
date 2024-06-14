from .logger import get_logger
logger = get_logger(__name__)
import pytorch_lightning as pl
from collections import defaultdict
import torch, time, mdtraj, os, wandb
import numpy as np
from .model import VAMDModel
from sit.transport import create_transport, Sampler
from functools import partial

def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except:
            pass
    return out

class Wrapper(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = 0

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.mean().item()
        log = self._log
        if self.stage == 'train' or self.args.validate:
            log["iter_" + key].append(data)
        log[self.stage + "_" + key].append(data)

    def load_ema_weights(self):
        logger.info('Loading EMA weights')
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        logger.info('Restoring cached weights')
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        if self.args.ema:
            self.ema.update(self.model)

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        if self.args.ema:
            if (self.ema.device != self.device):
                self.ema.to(self.device)
        return self.general_step(batch)

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        if self.args.ema:
            if (self.ema.device != self.device):
                self.ema.to(self.device)
            if (self.cached_weights is None):
                self.load_ema_weights()

        self.general_step(batch)
        self.validation_step_extra(batch, batch_idx)
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()

    def general_step(self, batch):
        out = self.general_step(batch)
        self.log('dur', time.time() - self.last_log_time)
        self.last_log_time = time.time()
        return out

    def validation_step_extra(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self.print_log(prefix='train', save=False)

    def on_validation_epoch_end(self):
        if self.args.ema:
            self.restore_cached_weights()
        self.print_log(prefix='val', save=False)

    def on_before_optimizer_step(self, optimizer):
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            self.print_log()

        if self.args.check_grad:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    logger.warning(f"Param {name} has no grad")

    def on_load_checkpoint(self, checkpoint):
        if self.args.ema:
            logger.info('Loading EMA state dict')
            ema = checkpoint["ema"]
            self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        if self.args.ema:
            if self.cached_weights is not None:
                self.restore_cached_weights()
            checkpoint["ema"] = self.ema.state_dict()

    def print_log(self, prefix='iter', save=False, extra_logs=None):
        log = self._log
        log = {key: log[key] for key in log if f"{prefix}_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update({
            'epoch': self.trainer.current_epoch,
            'trainer_step': self.trainer.global_step + int(prefix == 'iter'),
            'iter_step': self.iter_step,
            f'{prefix}_count': len(log[next(iter(log))]),

        })
        if extra_logs:
            mean_log.update(extra_logs)
        try:
            for param_group in self.optimizers().optimizer.param_groups:
                mean_log['lr'] = param_group['lr']
        except:
            pass

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.args.wandb:
                wandb.log(mean_log)
            if save:
                path = os.path.join(
                    os.environ["MODEL_DIR"],
                    f"{prefix}_{self.trainer.current_epoch}.csv"
                )
                pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if f"{prefix}_" in key:
                del self._log[key]

    def configure_optimizers(self):
        cls = torch.optim.AdamW if self.args.adamW else torch.optim.Adam
        optimizer = cls(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
        )
        return optimizer

class VAMDWrapper(Wrapper):

    def __init__(self, args):
        super().__init__(args)
        self.model = VAMDModel(args)

        self.transport = create_transport(
            args.path_type,
            args.prediction,
            None,  # args.loss_weight,
            # args.train_eps,
            # args.sample_eps,
        )  # default: velocity; 
        self.transport_sampler = Sampler(self.transport)

        if args.ema:
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=args.ema_decay
            )
            self.cached_weights = None



    def general_step(self, batch):
        
        
        x = batch['pos']
        x = x - x.mean(-2, keepdims=True)
        out_dict = self.transport.training_losses(
            model=self.model,
            x1=x,
            mask=torch.ones_like(x),
            model_kwargs={'mask': x.new_ones(*x.shape[:-1])},
        )
        loss = out_dict['loss']
        self.log('loss', loss)
        self.log('time', out_dict['t'])
        return loss.mean()

    def inference(self, batch):
        B, N, _ = batch['pos'].shape
        zs = torch.randn(B, N, 3, device=self.device)
        sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method)
        samples = sample_fn(zs, partial(self.model, mask=zs.new_ones(*zs.shape[:-1])))[-1]
        return samples

    def validation_step_extra(self, batch, batch_idx):

        do_inference = batch_idx < self.args.inference_batches and (
                (self.current_epoch + 1) % self.args.inference_freq == 0 or \
                self.args.validate) and self.trainer.is_global_zero
        if do_inference:
            samples = self.inference(batch)
            top = mdtraj.load('/data/cb/scratch/share/mdgen/4AA_sims/LIFE/LIFE.pdb').top
            traj = mdtraj.Trajectory(samples.cpu().numpy(), top)
            path = os.path.join(os.environ["MODEL_DIR"], f'epoch{self.current_epoch}_batch{batch_idx}.pdb')
            traj.save(path)
                     
