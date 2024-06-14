import torch, mdtraj, os, random
import numpy as np
from .logger import get_logger
logger = get_logger(__name__)

class VAMDDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args

    def __len__(self):
        return self.args.num_samples

    def __getitem__(self, idx):
        path = f"{self.args.sample_dir}/{idx}.pdb"
        if not os.path.exists(path):
            path = random.choice(os.listdir(self.args.sample_dir))
            path = f"{self.args.sample_dir}/{path}"
        try:
            traj = mdtraj.load(path)
        except:
            logger.warning(f"Error loading {path}")
            return self[(idx+1)%len(self)]
        return {'pos': traj.xyz[0]}


        
        
### It takes ~N times longer to generate a sample than to train on it
### That means if we allocate equal amounts of compute to inf and train, we train on a sample ~N times
### In reality we need to spend more time on inf for MD, so we train on a sample > N times
### The longer the MD, the more times we train on a sample
### We cannot refresh after a fixed number of epochs because this is a unstable equilibrium
### We instead refresh after a fixed number of samples
### Or maybe better to repeat epochs until MD runner says ready...
### After it's probably best to switch over immediately when MD runner says ready