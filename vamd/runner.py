import torch, os, tqdm, mdtraj, time, wandb, io
from .wrapper import VAMDWrapper, Wrapper
import numpy as np
from openmm.app import PDBFile, ForceField, Modeller, PME, HBonds, Simulation, StateDataReporter
from openmm import unit, LangevinMiddleIntegrator, Platform, MonteCarloBarostat
import rdkit.Chem
from .logger import get_logger
from collections import defaultdict
from scipy.spatial.transform import Rotation
logger = get_logger(__name__)

get_positions = lambda sim: sim.context.getState(getPositions=True).getPositions(asNumpy=True)

class TopologySupplier(torch.utils.data.IterableDataset):
    def __init__(self, args):
        pass

    def __iter__(self):
        pdb = PDBFile('/data/cb/scratch/share/mdgen/4AA_sims/LIFE/LIFE.pdb')
        mol = rdkit.Chem.rdmolfiles.MolFromPDBFile('/data/cb/scratch/share/mdgen/4AA_sims/LIFE/LIFE.pdb')
        
        ref_smi = rdkit.Chem.rdmolfiles.MolToSmiles(mol)
        
        while True:
            yield {
                'top': pdb.topology,
                'pos': np.random.randn(pdb.topology.getNumAtoms(), 3).astype(np.float32), 
                'ref_pos': np.array(pdb.positions / unit.nanometer),
                'ref_smi': ref_smi,
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


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except:
            pass
    return out
    
class VAMDRunner:
    def __init__(self, args):
        
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        
        self.ckpt_dir = os.environ["MODEL_DIR"]
        self.args = args
        
        self.model = self.load_model(None)
        self.ckpt_time = None
        
        self.iter = 0
        self.supplier = iter(torch.utils.data.DataLoader(TopologySupplier(args), collate_fn=collate_fn, batch_size=args.batch_size))

        os.makedirs(self.args.sample_dir, exist_ok=True)
        
    def log(self, key, data):
        self._log[key].append(data)

    def print_log(self):
        log = self._log
        mean_log = get_log_mean(log)
        logger.info(str(mean_log))

        if self.args.wandb:
            wandb.log(mean_log)
        for key in list(log.keys()):
            del self._log[key]        
    
    def load_model(self, ckpt):
        if ckpt is None:
            return VAMDWrapper(self.args).eval()
        else:
            return VAMDWrapper.load_from_checkpoint(ckpt, map_location='cpu').eval()

    @torch.no_grad()
    def run(self):
        loop_idx = 0
        while True:
            loop_idx += 1
            if self.args.md_only:
                self.md_only_iter()
            else:
                self.check_load_model()
                self.generate_batch()
            if loop_idx % self.args.print_freq == 0:
                self.print_log()

    def md_only_iter(self):
        idx = self.iter % self.args.num_samples
        logger.info(f'Loading {self.args.sample_dir}/{idx}.pdb')
        pdb = PDBFile(f'{self.args.sample_dir}/{idx}.pdb')
        top, pos = self.md_single(pdb.topology, pdb.positions / unit.nanometer)
        self.save_sample(top, pos)

        self.log('md_dur', time.time() - self.last_log_time)
        self.last_log_time = time.time()
        
        
    def check_load_model(self):
        if not os.path.exists(f"{self.ckpt_dir}/last.ckpt"):
            return False
        mtime = os.path.getmtime(f"{self.ckpt_dir}/last.ckpt")
        if (self.ckpt_time is None) or (mtime > self.ckpt_time):
            logger.info(f"Reloading checkpoint {self.ckpt_dir}/last.ckpt")
            try:
                self.model = self.load_model(f"{self.ckpt_dir}/last.ckpt")
                self.ckpt_time = mtime
            except:
                logger.warning("Error reloading checkpoint")
        return True
        

    def generate_batch(self):
        batch = next(self.supplier)
        batch = batch_to_tensor(batch, 'cuda')

        start = time.time()
        logger.info(f"Running model inference")
        self.model.to('cuda')
        batch['pos'] = self.model.inference(batch)
        batch = batch_to_tensor(batch, 'cpu')
        self.model.to('cpu')
        torch.cuda.empty_cache()
        self.log('inference_dur', time.time() - start)

        start = time.time()
        logger.info(f"Running MD simulations")
        batch = self.md_batch(batch, callback=self.save_sample)
        self.log('md_dur', time.time() - start)

        
        self.log('batch_dur', time.time() - self.last_log_time)
        self.last_log_time = time.time()
        

    def save_sample(self, top, pos):
        
        idx = self.iter % self.args.num_samples

        top = mdtraj.Topology.from_openmm(top)
        traj = mdtraj.Trajectory(pos, top)
        traj = traj.atom_slice(top.select("protein and (symbol != H)"))
        logger.info(f"Saving to {self.args.sample_dir}/{idx}.pdb")
        

        try:
            traj.save(f'{self.args.sample_dir}/{idx}.pdb')
        except:
            logger.warning(f'Error saving {self.args.sample_dir}/{idx}.pdb')
        
        self.iter += 1

    def check_stereochemistry(self, top, pos, ref_smi):
        smi = self.mol_to_smiles(top, pos)
        return smi == ref_smi

    def mol_to_smiles(self, top, pos):
        top = mdtraj.Topology.from_openmm(top)
        traj = mdtraj.Trajectory(pos, top)
        traj = traj.atom_slice(top.select("protein and (symbol != H)"))

        traj.save(f'{self.args.sample_dir}/tmp.pdb')
               
        
        mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(f'{self.args.sample_dir}/tmp.pdb', proximityBonding=True)
        if not mol:
            return None
        smi = rdkit.Chem.rdmolfiles.MolToSmiles(mol)
        return smi

    def md_single(self, top, pos, check_stereo=False, ref_smi=None):
        forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        modeller = Modeller(top, pos)
        modeller.addHydrogens()
        
        system = forcefield.createSystem(modeller.topology, constraints=HBonds)
        integrator = LangevinMiddleIntegrator(350 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    
        sim = Simulation(modeller.topology, system, integrator,
                        platform=Platform.getPlatformByName('CUDA'))
        
        sim.context.setPositions(modeller.positions)

        if not self.args.no_relax:
            sim.minimizeEnergy()

        if check_stereo and (not self.check_stereochemistry(modeller.topology, get_positions(sim), ref_smi)):
            logger.info("Rejecting relaxed structure: wrong stereochemistry")
            return top, None
            
        sim.context.setVelocitiesToTemperature(350 * unit.kelvin)
        sim.step(self.args.num_steps)

        return modeller.topology, get_positions(sim)
        

    def md_batch(self, batch, callback = lambda x, y: None):
        
        for i in range(self.args.batch_size):
            top = batch['top'][i]
            pos = batch['pos'][i].numpy()

            top, pos = self.md_single(top, pos * unit.nanometer, check_stereo=True, ref_smi=batch['ref_smi'][i])

            self.log('stereo_check', int(pos is not None))

            if pos is None:
                pos = batch['ref_pos'][i] # stereochemistry check failed
                pos = pos.numpy() @ Rotation.random().as_matrix().T.astype(np.float32)
            
            callback(top, pos)
        