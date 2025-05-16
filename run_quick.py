# train_quicktest.py

import os
import os.path as osp

# Prevent numpy over multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D, PatchGaussianDiffusion1D, PatchTrainer1D
from models import EBM, DiffusionWrapper, PatchEBM, PatchDiffusionWrapper
from models import SudokuEBM, SudokuTransformerEBM, SudokuDenoise, SudokuLatentEBM, AutoencodeModel
from models import GraphEBM, GraphReverse, GNNConvEBM, GNNDiffusionWrapper, GNNConvDiffusionWrapper, GNNConv1DEBMV2, GNNConv1DV2DiffusionWrapper, GNNConv1DReverse
from dataset import Addition, LowRankDataset, Inverse
from reasoning_dataset import FamilyTreeDataset, GraphConnectivityDataset, FamilyDatasetWrapper, GraphDatasetWrapper
from planning_dataset import PlanningDataset, PlanningDatasetOnline
from sat_dataset import SATNetDataset, SudokuDataset, SudokuRRNDataset, SudokuRRNLatentDataset
import torch
import argparse

try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    print('Warning: MKL not initialized.')

def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError('Invalid value: {}'.format(x))

parser = argparse.ArgumentParser(description='Train Diffusion Reasoning Model')
parser.add_argument('--dataset', default='inverse', type=str)
parser.add_argument('--model', default='mlp-patch', type=str)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--diffusion_steps', default=2, type=int)
parser.add_argument('--rank', default=2, type=int)
parser.add_argument('--supervise-energy-landscape', type=str2bool, default=True)
parser.add_argument('--ood', action='store_true', default=False)
parser.add_argument('--patchsize', default=2, type=int)
parser.add_argument('--patchwise_inference', default=True, type=bool)

if __name__ == "__main__":
    FLAGS = parser.parse_args()

    dataset = Inverse("train", FLAGS.rank, FLAGS.ood)
    validation_dataset = dataset
    metric = 'mse'

    model = PatchEBM(
        inp_dim=dataset.inp_dim,
        out_dim=dataset.out_dim,
        patchsize=FLAGS.patchsize
    )
    model = PatchDiffusionWrapper(model)

    diffusion = PatchGaussianDiffusion1D(
        model,
        seq_length=32,
        objective='pred_noise',
        timesteps=FLAGS.diffusion_steps,
        sampling_timesteps=FLAGS.diffusion_steps,
        supervise_energy_landscape=FLAGS.supervise_energy_landscape,
        use_innerloop_opt=False,
        show_inference_tqdm=False,
        continuous=True
    )

    result_dir = osp.join('results', f'ds_{FLAGS.dataset}', f'model_{FLAGS.model}_quicktest')
    os.makedirs(result_dir, exist_ok=True)

    trainer = PatchTrainer1D(
        diffusion,
        dataset,
        train_batch_size=FLAGS.batch_size,
        validation_batch_size=2,
        train_lr=1e-4,
        train_num_steps=10,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        data_workers=0,
        amp=False,
        metric=metric,
        results_folder=result_dir,
        cond_mask=False,
        validation_dataset=validation_dataset,
        extra_validation_datasets={},
        extra_validation_every_mul=1,
        save_and_sample_every=10,
        evaluate_first=False,
        latent=False,
        autoencode_model=None,
        patchwise_inference=FLAGS.patchwise_inference
    )

    trainer.train()
