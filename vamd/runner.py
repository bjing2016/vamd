import torch, os, tqdm, mdtraj
from .wrapper import VAMDWrapper
import numpy as np
from openmm.app import PDBFile, ForceField, Modeller, PME, HBonds, Simulation, StateDataReporter
from openmm import unit, LangevinMiddleIntegrator, Platform, MonteCarloBarostat
from .logger import get_logger
logger = get_logger(__name__)

get_positions = lambda sim: sim.context.getState(getPositions=True).getPositions(asNumpy=True)

class TopologySupplier(torch.utils.data.IterableDataset):
    def __init__(self, args):
        pass

    def __iter__(self):
        top = PDBFile('/data/cb/scratch/share/mdgen/4AA_sims/LIFE/LIFE.pdb').topology
        
        while True:
            yield {
                'top': top,
                'pos': np.random.randn(top.getNumAtoms(), 3).astype(np.float32), 
            }

def collate_fn(data):
    out = {}
    for key in data[0]:
        if type(data[0][key]) == np.ndarray:
            out[key] = np.stack([dat[key] for dat in data])
        elif type(data[0][key]) == torch.Tensor:
            out[key] = torch.stack([dat[key] for dat in data])
        else:
            out[key] = [dat[key] for dat in data]
    return out
    
def batch_to_tensor(batch, device='cuda'):
    out = {**batch}
    for key in batch:
        if type(batch[key]) is np.ndarray:
            out[key] = torch.from_numpy(batch[key]).to(device)
        if type(batch[key]) is torch.Tensor:
            out[key] = batch[key].to(device)
    return out
    
class VAMDRunner:
    def __init__(self, args):
        self.ckpt_dir = os.environ["MODEL_DIR"]
        self.args = args
        
        self.model = self.load_model(None)
        self.ckpt_time = None
        
        self.iter = 0
        self.supplier = iter(torch.utils.data.DataLoader(TopologySupplier(args), collate_fn=collate_fn, batch_size=args.batch_size))
        

    def load_model(self, ckpt):
        if ckpt is None:
            return VAMDWrapper(self.args)
        else:
            return VAMDWrapper.load_from_checkpoint(ckpt, map_location='cpu')

    @torch.no_grad()
    def run(self, limit=np.inf):
        if self.iter > limit:
            return
        while True:
            self.check_load_model()
            self.run_single()

    def check_load_model(self):
        if not os.path.exists(f"{self.ckpt_dir}/last.ckpt"):
            return False
        mtime = os.path.getmtime(f"{self.ckpt_dir}/last.ckpt")
        if (self.ckpt_time is None) or (mtime > self.ckpt_time):
            self.load_model(f"{self.ckpt_dir}/last.ckpt")
        self.ckpt_time = mtime
        logger.info(f"Reloading checkpoint {self.ckpt_dir}/last.ckpt")
        return True
        

    def run_single(self):
        batch = next(self.supplier)
        batch = batch_to_tensor(batch, 'cuda')

        logger.info(f"Running model inference")
        self.model.to('cuda')
        batch['pos'] = self.model.inference(batch)
        batch = batch_to_tensor(batch, 'cpu')
        self.model.to('cpu')
        torch.cuda.empty_cache()

        logger.info(f"Running MD simulations")
        batch = self.md_batch(batch, callback=self.save_sample)

    def save_sample(self, top, pos):
        
        idx = self.iter % self.args.num_samples

        top = mdtraj.Topology.from_openmm(top)
        traj = mdtraj.Trajectory(pos, top)
        traj = traj.atom_slice(top.select("protein and (symbol != H)"))
        logger.info(f"Saving to {self.args.sample_dir}/{idx}.pdb")
        os.makedirs(self.args.sample_dir, exist_ok=True)

        try:
            traj.save(f'{self.args.sample_dir}/{idx}.pdb')
        except:
            logger.warning(f'Error saving {self.args.sample_dir}/{idx}.pdb')
        
        self.iter += 1

    def md_batch(self, batch, callback = lambda x, y: None):

        forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

        
        for i in range(self.args.batch_size):
            top = batch['top'][i]
            pos = batch['pos'][i].numpy()

            modeller = Modeller(top, pos * unit.nanometer)
            modeller.addHydrogens()
            
            system = forcefield.createSystem(modeller.topology, constraints=HBonds)
            integrator = LangevinMiddleIntegrator(350 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
        
            sim = Simulation(modeller.topology, system, integrator,
                            platform=Platform.getPlatformByName('CUDA'))
            
            sim.context.setPositions(modeller.positions)

            sim.minimizeEnergy()
            sim.step(100000)

            callback(modeller.topology, get_positions(sim))
        