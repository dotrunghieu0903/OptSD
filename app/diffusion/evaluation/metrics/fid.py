import os
import numpy as np
import torch
from cleanfid import fid

def get_fid_features(dataset_or_folder_path, cache_path, mode='clean', device: str | torch.device ='cuda'):
    """
    Extract features for FID calculation from a dataset or folder.

    Args:
        dataset_or_folder (str): Path to the dataset or folder containing images.
        cache_path (str): Path to cache the extracted features.

    Returns:
        tuple: Mean and covariance of the features.
    """
    np_feats = []
    feat_model = fid.build_feature_extractor(mode, device)
    if isinstance(dataset_or_folder_path, str):
        np_feats = fid.get_folder_features()   
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    mu, sigma = fid.get_features(dataset_or_folder_path, cache_path=cache_path)
    return mu, sigma

def fid_calculate(real_features, fake_features):
    """
    Calculate the Frechet Inception Distance (FID) between real and fake features.

    Args:
        real_features (np.ndarray): Features extracted from real images.
        fake_features (np.ndarray): Features extracted from generated images.

    Returns:
        float: The FID score.
    """
    mu1, sigma1 = get_fid_features(dataset_or_folder_path=ref_dirpath, cache_path=ref_cache_path)
    mu2, sigma2 = get_fid_features(dataset_or_folder_path=gen_dirpath, cache_path=gen_cache_path)
    fid_score = fid.frechet_distance(mu1, sigma1, mu2, sigma2)
    fid_score = float(fid_score)
    if sym_ref_dirpath is not None:
        os.remove(sym_ref_dirpath)
    os.remove(sym_gen_dirpath)
    return fid_score
