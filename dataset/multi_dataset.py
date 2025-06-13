import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist

from utils.distribute import main_print

class MultiTypeDataset(Dataset):
    def __init__(self, datasets, batch_size, seed=42, world_size=1, rank=0):
        """
        Combines multiple datasets and ensures each batch contains only one type.
        Supports distributed training by ensuring all processes sample from the same dataset.
        
        Args:
            datasets (list): List of datasets [dataset1, dataset2, dataset3, dataset4]
            batch_size (int): Batch size used during training
            seed (int): Random seed for shuffling
            world_size (int): Total number of processes (default: 1)
            rank (int): Process rank (default: 0)
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.global_batch_size = batch_size * world_size
        
        # Calculate number of complete batches per dataset (global batches)
        self.batches_per_dataset = [length // self.global_batch_size for length in self.dataset_lengths]
        
        # Create indices map for reorganizing data
        self._create_indices_map()
        
        main_print(f"** DATA ** Combine {len(datasets)} datasets for Training (rank {rank}/{world_size})")
    
    def _create_indices_map(self):
        """Create a mapping from linear indices to (dataset_idx, sample_idx) that's consistent across processes"""
        # Set seed the same for all processes
        np.random.seed(self.seed)
        
        # Generate global indices map first (shared across all processes)
        global_indices = []
        dataset_indices_list = []
        
        # For each dataset, create and shuffle indices
        for dataset_idx, dataset_len in enumerate(self.dataset_lengths):
            # Create indices for this dataset and shuffle them
            dataset_indices = list(range(dataset_len))
            np.random.shuffle(dataset_indices)
            dataset_indices_list.append(dataset_indices)
            
            # Calculate number of complete global batches
            num_batches = self.batches_per_dataset[dataset_idx]
            
            # Add batches to global indices list (dataset_idx, batch_idx)
            for i in range(num_batches):
                global_indices.append((dataset_idx, i))
        
        # Shuffle the batches (not individual samples)
        np.random.shuffle(global_indices)
        
        # Now create the process-specific indices map (all processors have different samples but always the same dataset within a batch)
        self.indices_map = []
        for global_batch_idx, (dataset_idx, batch_idx) in enumerate(global_indices):
            # Calculate the global samples for this batch
            global_start = batch_idx * self.global_batch_size
            
            # Calculate this process's samples
            local_start = global_start + self.rank * self.batch_size
            local_end = local_start + self.batch_size
            
            # Get the actual dataset indices
            dataset_indices = dataset_indices_list[dataset_idx]
            
            # Add the (dataset_idx, sample_idx) pairs for this process's portion
            for idx in dataset_indices[local_start:local_end]:
                self.indices_map.append((dataset_idx, idx))
    
    def __len__(self):
        return len(self.indices_map)
    
    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.indices_map[idx]
        return self.datasets[dataset_idx][sample_idx]
 
    def set_epoch(self, epoch):
        """Reshuffle for a new epoch with updated seed"""
        self.seed = int(self.seed + epoch)
        self._create_indices_map()

    @classmethod
    def from_distributed_env(cls, datasets, batch_size, seed=42):
        """Factory method that creates dataset with current distributed settings"""
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
            
        return cls(datasets, batch_size, seed, world_size, rank)