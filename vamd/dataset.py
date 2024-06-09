import torch, mdtraj

class VAMDDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.traj = mdtraj.load(
            '/data/cb/scratch/share/mdgen/4AA_sims/LIFE/LIFE.xtc', 
            top='/data/cb/scratch/share/mdgen/4AA_sims/LIFE/LIFE.pdb'
        )

    def __len__(self):
        return len(self.traj)

    def __getitem__(self):
        return self.traj.xyz[0]