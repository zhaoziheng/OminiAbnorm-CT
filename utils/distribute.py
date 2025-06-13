import torch.distributed as dist

def is_dist_initialized():
    """
    Check if distributed training is initialized.
    
    Returns:
        bool: True if distributed training is initialized, False otherwise.
    """
    return dist.is_initialized()

def main_print(*args, **kwargs):
    """
    Print only from the main process.
    
    Args:
        *args: Positional arguments to pass to print function
        **kwargs: Keyword arguments to pass to print function
    """
    if is_main_process():
        print(*args, **kwargs)
        
def is_main_process():
    """
    Check if the current process is the main process.
    
    Returns:
        bool: True if the current process is the main process (rank 0) or if DDP is not being used.
    """
    if is_dist_initialized():
        # DDP is being used
        return dist.get_rank() == 0
    else:
        # DDP is not being used
        return True

def get_world_size():
    """
    Get the number of processes in the distributed training.
    
    Returns:
        int: Number of processes if DDP is being used, 1 otherwise.
    """
    if is_dist_initialized():
        return dist.get_world_size()
    else:
        return 1

def get_rank():
    """
    Get the rank of the current process.
    
    Returns:
        int: Rank of the current process if DDP is being used, 0 otherwise.
    """
    if is_dist_initialized():
        return dist.get_rank()
    else:
        return 0