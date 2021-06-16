import numpy as np
import torch
import torch.nn.functional as F

from utils.config import cfg


def cox_loss_ties(pred, cens, tril, tied_matrix):
    """
    Compute the Efron version of the Cox loss. This version take into
    account the ties.
    t unique time
    H_t denote the set of indices i such that y^i = t and c^i =1.
    c^i = 1 event occured.
    m_t = |H_t| number of elements in H_t.
    l(theta) = sum_t (sum_{i in H_t} h_{theta}(x^i)
                     - sum_{l=0}^{m_t-1} log (
                        sum_{i: y^i >= t} exp(h_{theta}(x^i))
                        - l/m_t sum_{i in H_t} exp(h_{theta}(x^i)))

    Parameters
    ----------
    pred : torch tensor
        Model prediction.
    cens : torch tensor
        Event tensor.
    tril : torch tensor
        Lower triangular tensor.
    tied_matrix : torch tensor
        Diagonal by block tensor.

    Returns
    -------
    loss : float
        Efron version of the Cox loss.
    """

    # Note that the observed variable is not required as we are sorting the
    # inputs when generating the batch according to survival time.

    # exp(h_{theta}(x^i))
    exp_pred = torch.exp(pred)
    # Term corresponding to the sum over events in the risk pool
    # sum_{i: y^i >= t} exp(h_{theta}(x^i))
    future_theta = torch.mm(tril.transpose(1, 0), exp_pred)
    # sum_{i: y^i >= t} exp(h_{theta}(x^i))
    # - l/m_t sum_{i in H_t} exp(h_{theta}(x^i))
    tied_term = future_theta - torch.mm(tied_matrix, exp_pred)
    # log (sum_{i: y^i >= t} exp(h_{theta}(x^i))
    #      - l/m_t sum_{i in H_t} exp(h_{theta}(x^i))
    tied_term = torch.log(tied_term)
    # event row vector to column
    tied_term = tied_term.view((-1, 1))
    cens = cens.view((-1, 1))
    # sum_t (sum_{i in H_t} h_{theta}(x^i)
    #       - sum_{l=0}^{m_t-1} log (
    #          sum_{i: y^i >= t} exp(h_{theta}(x^i))
    #          - l/m_t sum_{i in H_t} exp(h_{theta}(x^i)))
    loss = (pred - tied_term) * cens
    # Negative loglikelihood
    loss = -torch.mean(loss)
    return loss


def cox_loss_basic(pred, cens, tril, tied_matrix):
    """
    Compute the basic version of the Cox loss.
    Subjects i and j
    c^i = 1 event occurred
    l(theta) = sum_{i:c^i=1}
                (h_{theta}(x^i)
                 - log sum_{j:y^j >= y^i} exp(h_{theta}(x^j)))
    The second summation is over the set of subjects j where the event has not
    occured before time Y_i (including subject i itself).

    Parameters
    ----------
    pred : torch tensor
        Model prediction.
    cens : torch tensor
        Event tensor.
    tril : torch tensor
        Lower triangular tensor.
    tied_matrix : torch tensor
        Diagonal by block tensor.

    Returns
    -------
    loss : float
        Basic Cox loss.
    """

    # Note that the observed variable is not required as we are sorting the
    # inputs when generating the batch according to survival time.

    # Compute the second summation
    # exp(h_{theta}(x^j))
    exp_pred = torch.exp(pred)
    tril = np.tril(np.ones(tril.size(), dtype="float32"))
    tril = torch.from_numpy(tril)
    if cfg.CUDA:
        tril = tril.cuda()
    # Sum of the exp pred of the risk pool for each i
    # sum_{j:y^j >= y^i} exp(h_{theta}(x^j)))
    future_theta = torch.mm(tril.transpose(1, 0), exp_pred)
    # log sum_{j:y^j >= y^i} exp(h_{theta}(x^j)))
    tied_term = torch.log(future_theta)

    tied_term = tied_term.view((-1, 1))
    # event row vector to column
    cens = cens.view((-1, 1))
    # sum_{i:c^i=1} (h_{theta}(x^i)
    #                - log sum_{j:y^j >= y^i} exp(h_{theta}(x^j)))
    loss = (pred - tied_term) * cens
    # Negative loglikelihood
    loss = -torch.mean(loss)
    return loss


def rank_loss(pred, rank, cens, f="RankingSVM"):
    """
    Compute the ranking loss.

    Parameters
    ----------
    pred : torch tensor
        Model prediction.
    rank : torch tensor
        Time tensor (ground truth).
    cens : torch tensor
        Event tensor.
    f : str
        Ranking function. Default: "RankingSVM".

    Returns
    -------
    loss : float
        Ranking loss.
    """
    # Find acceptable pairs
    mat_rank = rank.unsqueeze(0).repeat(rank.size(0), 1)
    mat_rank = mat_rank.transpose(1, 0) - mat_rank
    diagonal = torch.eye(mat_rank.size()[0])
    if cfg.CUDA:
        diagonal = diagonal.cuda()
    mat_rank = mat_rank - diagonal
    mat_rank = mat_rank >= 0
    mat_rank = mat_rank.float()
    cens = cens.unsqueeze(0).repeat(cens.size(0), 1)
    mat_rank = mat_rank * cens

    # Differences between the pairs in the prediction
    t_diff = pred.transpose(1, 0).repeat(pred.size(0), 1)
    t_diff = t_diff.transpose(1, 0) - t_diff

    # Apply f to the time differences
    if f == "RankingSVM":
        t_diff = (t_diff - 1)
        t_diff = t_diff.clamp(0)
    elif f == "RankBoost":
        # To avoid inf with torch.exp
        t_diff = torch.clamp(t_diff, -88, 88)
        t_diff = 1. - torch.exp(-t_diff)
    elif f == "sigm":
        t_diff = torch.sigmoid(t_diff)
    elif f == "log_sigm":
        t_diff = F.logsigmoid(t_diff)
    else:
        raise NotImplementedError()

    loss = torch.sum(t_diff * mat_rank) / torch.sum(mat_rank).data.item()
    return -loss


def emd_loss(cdf_pred, cdf_time, weight_mat, exponent=1.5):
    """
    compute the EMD loss.

    Parameters
    ----------
    cdf_pred : torch tensor
        Model predicted CDF.
    cdf_time : torch tensor
        Time CDF (ground truth).
    weight_mat : torch tensor
        Weight matrix.
    exponent : float
        EMD exponent.

    Returns
    -------
    loss : float
        EMD loss.
    """
    ntime = cdf_pred.size(1)
    emd = torch.abs(cdf_pred - cdf_time)
    emd = emd ** exponent
    emd = torch.mm(emd, (weight_mat**exponent).view(-1, 1))
    loss = torch.sum(emd, 1) / ntime
    loss = torch.mean(loss)
    return loss
