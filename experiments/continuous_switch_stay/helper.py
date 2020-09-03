import os 
import sys 
sys.path.append(os.getcwd())

import numpy as np
import scipy.special
import torch 
import torch.nn as nn
import quadpy

import src.utils.math_utils as utils

import env

n_states = 2 
n_actions = 2


def init_params(param_range):
    # convert numbers to torch params
    return nn.Parameter(torch.tensor(param_range))

def get_random_inits(muspace, logstdspace, n_samples):
    # randomly initialize mean and std
    mu = np.random.uniform(low = muspace[0], high = muspace[1], size = n_samples)
    std = np.random.uniform(low = logstdspace[0], high = logstdspace[1], size = n_samples)
    return mu, std

def approx_log(x):
    # approx log for numerical error
    return torch.log(x + 1e-5)
    

def gauss_cdf(x, mu, sigma):
    assert (sigma <= 0).sum() == 0, sigma
    return 1/2 * (1 + scipy.special.erf((x - mu) / sigma / np.sqrt(2)))

def tanh_gauss_cdf(x, mu, sigma):
    scheme = quadpy.line_segment.clenshaw_curtis(2048)
    ixs = np.argwhere(scheme.points[1:-1] <= x) # exclude endpoints
    points = torch.tensor(scheme.points[1:-1][ixs]).squeeze()
    weights = torch.tensor(scheme.weights[1:-1][ixs]).squeeze()
    # print(points)
    pdfs = tanh_gauss(points.unsqueeze(-1), torch.tensor(mu).unsqueeze(0), torch.tensor(sigma).unsqueeze(0)) # calculate across states
    # assert pdfs.numel() == points.numel(), (pdfs.numel(), points.numel())
    # print(pdfs.shape)
    assert pdfs.shape == (points.numel(), mu.size)
    cdf = torch.sum(weights.unsqueeze(-1) * pdfs, dim = 0).numpy()
    # assert cdf.max() <= 1, cdf.max()
    # assert cdf.min() >= 0, cdf.min()
    return np.clip(cdf, 0, 1) #clamp for numerical stuff

def batch_tanh_gauss_cdf(x, mu, sigma):
    # Calculate batch of tanh-gauss pdfs
    scheme = quadpy.line_segment.clenshaw_curtis(2048)
    ixs = np.argwhere(scheme.points[1:-1] <= x) # for getting cdf
    points = torch.tensor(scheme.points[1:-1][ixs]).squeeze()
    weights = torch.tensor(scheme.weights[1:-1][ixs]).squeeze()
    # print(points)
    pdfs = tanh_gauss(points.reshape((-1, 1, 1)), torch.tensor(mu).unsqueeze(0), torch.tensor(sigma).unsqueeze(0)) # calculate across states
    # assert pdfs.numel() == points.numel(), (pdfs.numel(), points.numel())
    # print(pdfs.shape)
    assert pdfs.shape == (points.numel(), *mu.shape), pdfs.shape
    cdfs = torch.sum(weights.reshape((-1, 1, 1)) * pdfs, dim = 0).numpy()
    # assert cdfs.max() <= 1, cdfs.max()
    # assert cdfs.min() >= 0, cdfs.min()
    return np.clip(cdfs, 0, 1) # clip for numerical issues


def gauss(a, mu, sigma):
    # calculate Gaussian pdf
    # assert torch.min(sigma) > 0, sigma[sigma <= 0]
    norm = (sigma + 1e-3) * np.sqrt(2 * np.pi)
    pdf = torch.exp(-1/2 * torch.pow((a - mu) / (sigma + 1e-3), 2)) / norm
    # assert torch.min(pdf) >= 0, pdf
    # assert pdf.shape == (a.numel(), mu.numel()), print(a.shape, mu.shape)
    # assert pdf.numel() == a.numel() * mu.numel(), a.numel()
    # assert pdf[pdf > 0].numel() == pdf.numel(), print(pdf, mu, sigma)
    return torch.clamp(pdf, min = 0) # just in case there is numerical error

def tanh_gauss(a, mu, sigma): 
    norm = 1 - torch.tanh(utils.atanh(a)).pow(2)
    before = gauss(utils.atanh(a), mu, sigma) 
    assert torch.isnan(norm).sum() == 0
    assert torch.isnan(before).sum() == 0
    pdf = before / norm
    assert torch.min(pdf) >= 0, pdf
    # assert pdf.shape == (a.numel(), mu.numel())
    # assert pdf.numel() == a.numel() * mu.numel()
    # assert pdf[pdf > 0].numel() == pdf.numel()
    return pdf


def tanh_gauss_to_discrete(pi):
    # convert gaussian policy to equivalent discrete policy
    p_stay = tanh_gauss_cdf(0, pi[:, 0], pi[:, 1]).reshape((-1, 1))
    pi = np.concatenate((p_stay, 1 - p_stay), axis = 1)
    assert pi.shape == (n_states, n_actions)
    return pi

def gauss_to_discrete(pi):
    # convert gaussian policy to equivalent discrete policy
    p_stay = gauss_cdf(0, pi[:, 0], pi[:, 1]).reshape((-1, 1))
    pi = np.concatenate((p_stay, 1 - p_stay), axis = 1)
    assert pi.shape == (n_states, n_actions)
    return pi

def transform_std_dev(param):
    return torch.log(1 + torch.exp(param))

def get_r_pi(pi, tanh = True):
    # given mu, sigma in states, return r_pi
    if tanh is True:
        p_stay = tanh_gauss_cdf(0, pi[:, 0], pi[:, 1])
    else:
        p_stay = gauss_cdf(0, pi[:, 0], pi[:, 1])
    r_pi = env.r[:, 0] * p_stay + env.r[:, 1] * (1 - p_stay) 
    return r_pi

def get_P_pi(pi, tanh = True):
    # get probability transition corresponding to policy
    if tanh is True:
        P_pi = np.sum(env.P * np.expand_dims(tanh_gauss_to_discrete(pi), axis = -1), axis = 1)
    else:   
        P_pi = np.sum(env.P * np.expand_dims(gauss_to_discrete(pi), axis = -1), axis = 1)
    return P_pi

def get_batch_P_r_pi(pis, tanh = True):
    # arg is (n_states, n_inits, 2)
    if tanh is True:
        p_stays = batch_tanh_gauss_cdf(0, pis[:, :, 0], pis[:, :, 1])
        assert np.max(p_stays) <= 1, p_stays.max()
        assert np.min(p_stays) >= 0, p_stays.min()
        assert p_stays.shape == pis.shape[:-1]
        discrete_pis = np.concatenate((p_stays[:, np.newaxis, :], 1 - p_stays[:, np.newaxis, :]), axis = 1).transpose((2, 0, 1))
        assert discrete_pis.shape == (pis.shape[1], n_states, n_actions)
        P_pis = np.sum(env.P[np.newaxis, :, :, :] * discrete_pis[:, :, :, np.newaxis], axis = 2)
        r_pis = env.r[:, 0][:, np.newaxis] * p_stays + env.r[:, 1][:, np.newaxis] * (1 - p_stays)
    assert P_pis.shape == (pis.shape[1], n_states, n_states), P_pis.shape
    assert r_pis.shape == p_stays.shape, r_pis.shape
    return P_pis, r_pis.transpose((1, 0))

def get_V(pi, gamma):
    return np.linalg.inv(np.eye(n_states) - gamma * get_P_pi(pi)) @ get_r_pi(pi)

def get_batch_V(pis, gamma):
    # P_pis = np.stack([get_P_pi(pis[:, i, :]) for i in range(pis.shape[1])])
    # r_pis = np.stack([get_r_pi(pis[:, i, :]) for i in range(pis.shape[1])])
    P_pis, r_pis = get_batch_P_r_pi(pis, True)
    assert P_pis.shape == (pis.shape[1], n_states, n_states), P_pis.shape
    assert r_pis.shape == (pis.shape[1], n_states), r_pis.shape
    inv_arg = np.eye(n_states)[np.newaxis, :, :] - gamma * P_pis
    inv = np.linalg.inv(inv_arg)
    return np.sum(inv * r_pis[:, np.newaxis, :], axis = 2)

def get_batch_Q(pis, gamma):
    Vp = get_batch_V(pis, gamma)[:, np.newaxis, np.newaxis, :]
    assert Vp.shape == (pis.shape[1], 1, 1, n_states)
    EVp = np.sum(env.cont_P[np.newaxis, :, :, :] * Vp, axis = -1)
    return env.cont_r[np.newaxis, :, :] + gamma * EVp

def get_batch_soft_Q(pis, gamma, tau):
    Vp = get_batch_soft_V(pis, gamma, tau)[:, np.newaxis, np.newaxis, :]
    assert Vp.shape == (pis.shape[1], 1, 1, n_states)
    EVp = np.sum(env.cont_P[np.newaxis, :, :, :] * Vp, axis = -1)
    return env.cont_r[np.newaxis, :, :] + gamma * EVp

def get_batch_soft_V(pis, gamma, tau):
    # P_pis = np.stack([get_P_pi(pis[:, i, :]) for i in range(pis.shape[1])])
    P_pis, r_pis = get_batch_P_r_pi(pis, True)
    H = 1/2 * np.log(2 * np.pi * np.exp(1) * np.power(pis[:, :, 1], 2))
    assert H.shape == (n_states, pis.shape[1])
    r_pis += tau * H.T
    assert P_pis.shape == (pis.shape[1], n_states, n_states)
    assert r_pis.shape == (pis.shape[1], n_states)
    inv_arg = np.eye(n_states)[np.newaxis, :, :] - gamma * P_pis
    inv = np.linalg.inv(inv_arg)
    return np.sum(inv * r_pis[:, np.newaxis, :], axis = 2)
    

def get_Q(pi, gamma):
    # return (n_states, n_points, n_inits)
    Vp = get_V(pi, gamma).reshape((1, 1, n_states))
    EVp = np.sum(env.cont_P * Vp, axis = -1)
    # assert EVp.shape == (n_states, n_points)
    return env.cont_r + gamma * EVp

def get_soft_Q(pi, gamma, tau):
    Vp = get_soft_V(pi, gamma, tau).reshape((1, 1, n_states))
    EVp = np.sum(env.cont_P * Vp, axis = -1)
    # assert EVp.shape == (n_states, n_points)
    return env.cont_r + gamma * EVp

def get_soft_V(pi, gamma, tau):
    H = 1/2 * np.log(2 * np.pi * np.exp(1) * np.power(pi[:, 1], 2))
    r_pi = get_r_pi(pi) + tau * H
    return np.linalg.inv(np.eye(n_states) - gamma * get_P_pi(pi)) @ r_pi