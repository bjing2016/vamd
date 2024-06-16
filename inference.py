import argparse, tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
args = parser.parse_args()

import torch, mdtraj
import numpy as np
from vamd.wrapper import VAMDWrapper
from vamd.runner import TopologySupplier, batch_to_tensor

model = VAMDWrapper.load_from_checkpoint(args.ckpt).cuda()
model.eval()

top = mdtraj.load('/data/cb/scratch/share/mdgen/4AA_sims/LIFE/LIFE.pdb').top


out = []
for _ in tqdm.trange(10):
    pos = model.inference({'pos': torch.randn(1000, 37, 3, device='cuda')}).cpu().numpy()
    out.append(pos)

traj = mdtraj.Trajectory(np.concatenate(out), top)
traj[0].save('output.pdb')
traj.save('output.xtc')

    