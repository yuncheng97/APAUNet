import os
import torch
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

def padding(sample, target, input_shape):
    """
    Pads the input sample and target arrays to match the desired input_shape.

    Args:
        sample (numpy.ndarray): The input sample array.
        target (numpy.ndarray): The target array.
        input_shape (tuple): The desired shape of the padded arrays (H, W, D).

    Returns:
        tuple: The padded sample and target arrays.
    """
    HWD = np.array(input_shape)
    hwd = np.array(target.shape)
    pad_diff = np.clip((HWD - hwd) / 2, 0, None)
    pad_lower = np.floor(pad_diff).astype(int)
    pad_upper = np.ceil(pad_diff).astype(int)

    pad_width_sample = ((pad_lower[0], pad_upper[0]), (pad_lower[1], pad_upper[1]), (pad_lower[2], pad_upper[2]))
    pad_width_target = pad_width_sample if sample.ndim == 3 else ((0, 0), *pad_width_sample)

    sample = np.pad(sample, pad_width_sample, 'constant', constant_values=-3)
    target = np.pad(target, pad_width_target, 'constant')
    
    return sample, target

def random_crop(sample, target, input_shape):
    """
    Randomly crops the input sample and target arrays to match the desired input_shape.

    Args:
        sample (numpy.ndarray): The input sample array.
        target (numpy.ndarray): The target array.
        input_shape (tuple): The desired shape of the cropped arrays (H, W, D).

    Returns:
        tuple: The cropped sample and target arrays.
    """
    H, W, D = input_shape
    h, w, d = target.shape
    
    x = np.random.randint(0, h - H + 1)
    y = np.random.randint(0, w - W + 1)
    z = np.random.randint(0, d - D + 1)

    cropped_sample = sample[x:x+H, y:y+W, z:z+D]
    cropped_target = target[x:x+H, y:y+W, z:z+D]

    return cropped_sample, cropped_target

def bounding_crop(sample, target, input_shape):
    """
    Crops the input sample and target arrays to match the desired input_shape,
    while ensuring that the bounding box of the target is preserved.

    Args:
        sample (numpy.ndarray): The input sample array.
        target (numpy.ndarray): The target array.
        input_shape (tuple): The desired shape of the cropped arrays (H, W, D).

    Returns:
        tuple: The cropped sample and target arrays.
    """
    H, W, D = input_shape
    source_shape = np.array(target.shape)
    xyz = np.array(np.where(target > 0))

    lower_bound = []
    upper_bound = []
    for A, a, b in zip(xyz, source_shape, input_shape):
        lb = max(np.min(A), 0)
        ub = min(np.max(A) - b + 1, a - b + 1)

        if ub <= lb:
            lb = max(np.max(A) - b, 0)
            ub = min(np.min(A), a - b + 1)
        
        lower_bound.append(lb)
        upper_bound.append(ub)
    x, y, z = np.random.randint(lower_bound, upper_bound)

    cropped_sample = sample[x:x+H, y:y+W, z:z+D]
    cropped_target = target[x:x+H, y:y+W, z:z+D]

    return cropped_sample, cropped_target

def random_mirror(sample, target, prob=0.5):
    """
    Randomly flips the input sample and target arrays along each axis with a given probability.

    Args:
        sample (numpy.ndarray): The input sample array.
        target (numpy.ndarray): The target array.
        prob (float): The probability of flipping along each axis (default: 0.5).

    Returns:
        tuple: The flipped sample and target arrays.
    """
    p = np.random.uniform(size=3)
    flip_axes = tuple(np.where(p < prob)[0])
    sample = np.flip(sample, flip_axes)
    target = np.flip(target, flip_axes)
    return sample, target

def totensor(data):
    return torch.from_numpy(np.ascontiguousarray(data))

def mp_get_datas(data_dir, data_ids, dataset, num_worker=6):
    # Load data from files and preprocess target values
    def get_item(idx):
        sample, target = np.load(os.path.join(data_dir, idx))['data']
        target = np.maximum(target, 0)
        return (sample, target)
    
    # Use ThreadPool for parallel processing
    pool = ThreadPool(num_worker)
    datas = pool.map(get_item, data_ids)
    pool.close()
    pool.join()
    return datas

def mp_get_batch(data, idxs, input_shape, aug='random', num_worker=6):
    """
    Generates a batch of samples and targets with the specified augmentation method.

    Args:
        data (list): A list of tuples containing samples and targets.
        idxs (list): A list of indices to select samples and targets from the data.
        input_shape (tuple): The desired shape of the output arrays (H, W, D).
        aug (str): The augmentation method to use, either 'random' or 'bounding' (default: 'random').
        num_worker (int): The number of workers to use for parallel processing (default: 6).

    Returns:
        tuple: A tuple containing the batch of samples and targets.
    """
    crop_func = random_crop if aug == 'random' else bounding_crop

    def batch_and_aug(idx):
        sample, target  = data[idx]
        sample, target  = padding(sample, target, input_shape)
        sample, target  = crop_func(sample, target, input_shape)
        sample, target  = random_mirror(sample, target)
        sample          = torch.unsqueeze(totensor(sample), 0).unsqueeze(0)
        target          = torch.unsqueeze(totensor(target), 0).unsqueeze(0)
        return sample, target
    
    with ThreadPool(num_worker) as pool:
        batch = pool.map(batch_and_aug, idxs)

    samples, targets = zip(*batch)
    samples = torch.cat(samples)
    targets = torch.cat(targets)
    return samples, targets
