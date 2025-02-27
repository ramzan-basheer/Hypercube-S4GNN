import numpy as np
import random
import os
import pyedflib
import pickle
import torch
import torch_geometric
from scipy.signal import resample, welch
from scipy.stats import spearmanr
import math
from sklearn.manifold import MDS
from scipy import special
from constants import TUH_LABEL_DICT

def getOrderedChannels(file_name, verbose, labels_object, channel_names):
    labels = list(labels_object)
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def getEDFsignals(edf):
    """
    Get EEG signal in edf file
    Args:
        edf: edf object
    Returns:
        signals: shape (num_channels, num_data_points)
    """
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except:
            pass
    return signals


def resampleData(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freq
    Args:
        signals: EEG signal slice, (num_channels, num_data_points)
        to_freq: Re-sampled frequency in Hz
        window_size: time window in seconds
    Returns:
        resampled: (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)
    return resampled

def getSeizureTimes(file_name):
    """
    Args:
        file_name: edf file name
    Returns:
        seizure_times: list of times of seizure onset in seconds
    """
    tse_file = file_name.split(".edf")[0] + ".tse_bi"

    seizure_times = []
    with open(tse_file) as f:
        for line in f.readlines():
            if "seiz" in line:  # if seizure
                # seizure start and end time
                seizure_times.append(
                    [
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    ]
                )
    return seizure_times

def surface_laplacian(eeg_clip, distance_matrix):
    leg_order = 10
    m = 4
    smoothing = 1e-5
    
    # Assuming that eeg_clip is a numpy array with shape (batch, num_electrodes, seq_length)
    batch_size, numelectrodes, seq_length = eeg_clip.shape
    # Get electrodes positions
    locs = np.zeros((batch_size, numelectrodes, 3))
    for b in range(batch_size):
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
        locs[b] = mds.fit_transform(distance_matrix[b])

    x, y, z = locs[:, :, 0], locs[:, :, 1], locs[:, :, 2]

    # Arrange data
    orig_data_size = np.squeeze(eeg_clip.shape)

    # Normalize Cartesian coordinates to sphere unit
    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    _, _, spherical_radii = cart2sph(x, y, z)
    maxrad = np.max(spherical_radii)
    x = x / maxrad
    y = y / maxrad
    z = z / maxrad
    
    # Compute cosine distance between all pairs of electrodes
    cosdist = 1 - ((x[:, None, :] - x[:, :, None])**2 + 
                   (y[:, None, :] - y[:, :, None])**2 + 
                   (z[:, None, :] - z[:, :, None])**2) / 2

    cosdist = cosdist + np.transpose(cosdist, (0, 2, 1)) + np.identity(numelectrodes)

    # Get Legendre polynomials
    legpoly = np.zeros((batch_size, leg_order, numelectrodes, numelectrodes))
    # for ni in range(leg_order):
    #     for i in range(numelectrodes):
    #         for j in range(i + 1, numelectrodes):
    #             # Use a loop to calculate lpn for each element in the slices
    #             legpoly[:, ni, i, j] = np.array([special.lpn(ni + 1, val)[0][ni + 1] for val in cosdist[:, i, j]])
    for b in range(batch_size):
        for ni in range(leg_order):
            for i in range(numelectrodes):
                for j in range(i + 1, numelectrodes):
                    legpoly[b, ni, i, j] = special.lpn(ni + 1, cosdist[b, i, j])[0][ni + 1]


    legpoly = legpoly + np.transpose(legpoly, (0, 1, 3, 2))

    for i in range(leg_order):
        legpoly[:, i, :, :] = legpoly[:, i, :, :] + np.identity(numelectrodes)

    # Compute G and H matrices
    twoN1 = 2 * np.arange(1, leg_order + 1) + 1
    gdenom = np.power(np.multiply(np.arange(1, leg_order + 1), np.arange(2, leg_order + 2)), m, dtype=float)
    hdenom = np.power(np.multiply(np.arange(1, leg_order + 1), np.arange(2, leg_order + 2)), m - 1, dtype=float)

    G = np.zeros((batch_size, numelectrodes, numelectrodes))
    H = np.zeros((batch_size, numelectrodes, numelectrodes))

    for i in range(numelectrodes):
        for j in range(i, numelectrodes):
            g = 0
            h = 0

            for ni in range(leg_order):
                g = g + (twoN1[ni] * legpoly[:, ni, i, j]) / gdenom[ni]
                h = h - (twoN1[ni] * legpoly[:, ni, i, j]) / hdenom[ni]

            G[:, i, j] = g / (4 * np.pi)
            H[:, i, j] = -h / (4 * np.pi)

    G = G + np.transpose(G, (0, 2, 1))
    H = H + np.transpose(H, (0, 2, 1))

    G = G - np.identity(numelectrodes) * G[0, 1, 1] / 2
    H = H - np.identity(numelectrodes) * H[0, 1, 1] / 2

    # if np.any(orig_data_size == 1):
    #     eeg_clip = eeg_clip[:]
    # else:
        # eeg_clip = np.reshape(eeg_clip, (orig_data_size[0], np.prod(orig_data_size[1:3])))

    # Compute C matrix
    Gs = G + np.identity(numelectrodes) * smoothing
    GsinvS = np.sum(np.linalg.inv(Gs), axis=1)
    dataGs = np.einsum('ijk,ikl->ijl', eeg_clip.transpose(0, 2, 1), np.linalg.inv(Gs))
    C = dataGs - np.einsum('ijk,ikl->ijl', (np.sum(dataGs, axis=2) / np.sum(GsinvS, axis=1)[:, np.newaxis])[:, :, np.newaxis], GsinvS[:, np.newaxis, :])

    # Apply transform
    eeg_clip_surf_lap = np.reshape(np.transpose(np.matmul(C, np.transpose(H, (0, 2, 1)))), orig_data_size)
    eeg_clip_surf_lap = eeg_clip_surf_lap.astype(np.float32)

    return torch.tensor(eeg_clip_surf_lap)

def compute_dtf(x, order=5, epsilon=1e-10):
    # Get dimensions
    batch_size, n_channels, n_samples = x.shape
    
    # Center the data
    x = x - x.mean(dim=-1, keepdim=True)
    
    # Fit VAR model
    A, sigma = fit_var(x, order)
    
    # Compute transfer function H(f) using Fourier transform of AR coefficients
    # Define frequency range
    n_freqs = n_samples // 2 + 1
    
    # Initialize transfer matrix
    H = torch.zeros((batch_size, n_freqs, n_channels, n_channels), dtype=torch.complex64, device=x.device)
    
    # Compute transfer function for each frequency
    for f in range(n_freqs):
        # Compute normalized frequency
        omega = 2 * np.pi * f / n_samples
        
        # Initialize identity matrix
        A_f = torch.eye(n_channels, dtype=torch.complex64, device=x.device)
        
        # Subtract Fourier transformed AR coefficients
        for p in range(1, order + 1):
            A_f = A_f - A[:, p-1] * torch.exp(-1j * omega * p)
        
        # Compute transfer function H(f) = A^(-1)(f)
        for b in range(batch_size):
            H[b, f] = torch.inverse(A_f[b])
    
    # Compute DTF: |H_ij(f)|^2 / sum_k |H_ik(f)|^2
    H_abs_squared = torch.abs(H) ** 2
    denominator = H_abs_squared.sum(dim=-1, keepdim=True) + epsilon
    dtf = H_abs_squared / denominator
    
    return dtf

def fit_var(x, order):
    batch_size, n_channels, n_samples = x.shape
    
    # Initialize coefficient matrices
    A = torch.zeros((batch_size, order, n_channels, n_channels), dtype=torch.complex64, device=x.device)
    
    # Compute autocovariance matrices for lags 0 to order
    R = []
    for lag in range(order + 1):
        if lag == 0:
            # Covariance at lag 0
            R.append(torch.matmul(x, x.transpose(-1, -2)) / n_samples)
        else:
            # Covariance at lag > 0
            x_lagged = x[:, :, :-lag]
            x_current = x[:, :, lag:]
            R.append(torch.matmul(x_current, x_lagged.transpose(-1, -2)) / (n_samples - lag))
    
    # Construct Yule-Walker equations
    for b in range(batch_size):
        # Construct block Toeplitz matrix for Yule-Walker equations
        R_matrix = torch.zeros((n_channels * order, n_channels * order), dtype=torch.complex64, device=x.device)
        for i in range(order):
            for j in range(order):
                block_i_j = R[abs(i - j)][b]
                if i < j:
                    block_i_j = block_i_j.transpose(-1, -2).conj()
                R_matrix[i*n_channels:(i+1)*n_channels, j*n_channels:(j+1)*n_channels] = block_i_j
        
        # Right-hand side of Yule-Walker
        r_vec = torch.zeros((n_channels * order, n_channels), dtype=torch.complex64, device=x.device)
        for i in range(order):
            r_vec[i*n_channels:(i+1)*n_channels] = R[i+1][b]
        
        # Solve Yule-Walker equations for AR coefficients
        A_flat = torch.linalg.solve(R_matrix, r_vec)
        
        # Reshape solution to coefficient matrices
        for p in range(order):
            A[b, p] = A_flat[p*n_channels:(p+1)*n_channels]
    
    # Compute residual covariance
    sigma = R[0].clone()
    for p in range(order):
        sigma = sigma - torch.matmul(A[:, p], R[p+1])
    
    return A, sigma

def get_knn_graph(x, k, dist_measures, undirected=True):
    all_edge_index = []
    all_edge_weight = []
    all_adj_mat = []

    for dist_measure in dist_measures:
        epsilon = 1e-8
        if dist_measure == "euclidean":
            dist = torch.cdist(x, x, p=2.0)
            dist = (dist - dist.min()) / (dist.max() - dist.min())
            # batch_indices, node_indices = knn_ind.nonzero(as_tuple=True)
            # coordinates = x[batch_indices, node_indices, :]

            knn_val, knn_ind = torch.topk(
                dist, k, dim=-1, largest=False
            )  # smallest distances
        elif dist_measure == "cosine":
            norm = torch.norm(x, dim=-1, p="fro")[:, :, None]
            x_norm = x / norm
            dist = torch.matmul(x_norm, x_norm.transpose(1, 2))
            knn_val, knn_ind = torch.topk(
                dist, k, dim=-1, largest=True
            )  # largest similarities
        elif dist_measure == "spearmanr":
            ranks = x.argsort(dim=-1).argsort(dim=-1).float()
            centered_ranks = ranks - ranks.mean(dim=-1, keepdim=True)
            cov_matrix = torch.matmul(centered_ranks, centered_ranks.transpose(-1, -2)) / (ranks.shape[-1] - 1)
            
            std_dev = torch.sqrt(torch.diagonal(cov_matrix, dim1=-2, dim2=-1))+ epsilon
            dist = cov_matrix / (std_dev.unsqueeze(-1) * std_dev.unsqueeze(-2))
            knn_val, knn_ind = torch.topk(
                dist, k, dim=-1, largest=True
            )  # largest similarities
        elif dist_measure == "dtf":
            cross_spectral_density = torch.fft.fft(x, dim=-1)
            cross_spectral_density = torch.matmul(cross_spectral_density, cross_spectral_density.transpose(-1, -2).conj())
            auto_spectral_density = torch.abs(torch.fft.fft(x, dim=-1)) ** 2
            dist = torch.abs(cross_spectral_density) / (auto_spectral_density.sum(dim=-1, keepdim=True) + epsilon)
            knn_val, knn_ind = torch.topk(
                dist, k, dim=-1, largest=True
            )  # largest similarities
        elif dist_measure == "coherence":
            adj_mat_dir = '/home/jayakumar/SC22D036/ramzan/graphs4mer/data/eeg_electrode_graph/adj_mx_3d.pkl'
            with open(adj_mat_dir, "rb") as pf:
                adj_mat_dist = pickle.load(pf)
                adj_mat_dist = adj_mat_dist[-1]
            nodes = adj_mat_dist.shape[0]
            batch_size = x.size(0)
            adj_mat_dist = np.tile(adj_mat_dist.reshape(1, nodes, nodes), (batch_size, 1, 1))
            # x_surf = surface_laplacian(x.detach().cpu().numpy(), adj_mat_dist)
            x_surf = x
            cross_spectral_density = torch.fft.fft(x_surf, dim=-1)
            cross_spectral_density = torch.matmul(cross_spectral_density, cross_spectral_density.transpose(-1, -2).conj())

            auto_spectral_density = torch.abs(torch.fft.fft(x_surf, dim=-1)) ** 2

            auto_spectral_density = auto_spectral_density.sum(dim=-1, keepdim=True)

            auto_spectral_density = auto_spectral_density.expand_as(cross_spectral_density)

            dist = (torch.abs(cross_spectral_density) ** 2) / ((auto_spectral_density * auto_spectral_density.transpose(-1, -2)) + epsilon)
            knn_val, knn_ind = torch.topk(
                    dist, k, dim=-1, largest=True
                )  # largest similarities
        elif dist_measure == "pli":
            x_fft = torch.fft.fft(x, dim=-1)
            analytic_signal = x_fft
            phase = torch.angle(analytic_signal)
            phase_diff = phase.unsqueeze(2) - phase.unsqueeze(1)
            phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
            dist = torch.mean(torch.sign(torch.sin(phase_diff_wrapped)), dim=-1)
            knn_val, knn_ind = torch.topk(
                    dist, k, dim=-1, largest=True
                )  # largest similarities
        elif dist_measure == "wpli":
            x_fft = torch.fft.fft(x, dim=-1)
            analytic_signal = x_fft
            phase = torch.angle(analytic_signal)
            magnitude = torch.abs(analytic_signal)
            weighted_phase_diff = phase.unsqueeze(2) - phase.unsqueeze(1)
            weighted_phase_diff_wrapped = torch.atan2(torch.sin(weighted_phase_diff), torch.cos(weighted_phase_diff))
            dist = torch.mean(torch.sign(torch.sin(weighted_phase_diff_wrapped)) * magnitude.unsqueeze(2), dim=-1) / \
                torch.mean(magnitude.unsqueeze(2), dim=-1)
            knn_val, knn_ind = torch.topk(
                    dist, k, dim=-1, largest=True
                )  # largest similarities
        else:
            raise NotImplementedError

        adj_mat = (torch.ones_like(dist) * 0).scatter_(-1, knn_ind, knn_val).to(x.device)
        adj_mat = torch.clamp(adj_mat, min=0.0)  # remove negatives

        if undirected:
            adj_mat = (adj_mat + adj_mat.transpose(1, 2)) / 2

        # add self-loop
        I = (
            torch.eye(adj_mat.shape[-1], adj_mat.shape[-1])
            .unsqueeze(0)
            .repeat(adj_mat.shape[0], 1, 1)
            .to(bool)
        ).to(x.device)
        adj_mat = adj_mat * (~I) + I

        all_adj_mat.append(adj_mat)

        # to sparse graph
        edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat)

        # Expand the edge_weight to match the number of edge types
        edge_weight = edge_weight.unsqueeze(-1).expand(-1, len(dist_measures)).reshape(-1, len(dist_measures))

        all_edge_index.append(edge_index)
        all_edge_weight.append(edge_weight)

    # Concatenate the results to get the final multi-edge graph
    final_edge_index = torch.cat(all_edge_index, dim=-1)
    final_edge_weight = torch.cat(all_edge_weight, dim=0)  # Concatenate along dimension 0

    final_adj_mat = torch.stack(all_adj_mat, dim=-1)

    return final_edge_index, final_edge_weight, final_adj_mat
