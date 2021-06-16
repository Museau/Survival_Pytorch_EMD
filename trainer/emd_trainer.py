import math

import numpy as np
import torch
import torch.nn.functional as F

from models.networks import MLP
from trainer.trainer_base import TrainerBase
from utils.config import cfg
from utils.loss import emd_loss


class EMDTrainer(TrainerBase):
    def __init__(self, split):
        """
        EMD trainer.

        Parameters
        ----------
        split : int
            Split number.
        """
        super(EMDTrainer, self).__init__(split)

        if self.loss_type == "emd_loss":
            self.emd_loss = emd_loss
        else:
            raise NotImplementedError()

        self.div_time = int(cfg.TRAIN.DIV_TIME)
        self.dtime_shape = math.ceil(self.time_shape / self.div_time)

        self.model = MLP(input_shape=self.X_train_shape, output_shape=self.dtime_shape + 1)
        if cfg.CUDA:
            self.model.cuda()

    def before_train(self, train_data):
        """
        Get distance matrix for EMD.

        Parameters
        ----------
        train_data : ndarray
            Training set.
        """
        self.get_distance_matrix(train_data)

    def get_distance_matrix(self, train_data):
        """
        Compute the distance matrix for EMD.

        Parameters
        ----------
        train_data : ndarray
            Training set.
        """

        non_c_event_times = train_data[:, 0][train_data[:, 1] == 1]
        hist = np.histogram(non_c_event_times.tolist(), bins=list(range(0, math.ceil(self.time_shape / self.div_time))))
        hist = hist[0]

        prior = float(cfg.EMD.PRIOR)
        elts = (prior + hist).tolist()
        self.distance_mat = np.array([elt * (len(elts)) / (sum(elts)) for elt in elts])
        self.distance_mat = torch.from_numpy(self.distance_mat)
        self.distance_mat = self.distance_mat.float()
        padding = torch.zeros(2)
        self.distance_mat = torch.cat([self.distance_mat, padding])

        if cfg.CUDA:
            self.distance_mat = self.distance_mat.cuda()

    def get_pred_loss(self, batch):
        """
        Compute the model loss and prediction.

        Parameters
        ----------
        batch : ndarray
            Batch data.

        Returns
        -------
        concat_pred : ndarray
            Matrix containing the model predictions.
        loss : float
            Model loss.
        """
        time, event, X, _, _ = self.process_batch(batch)

        survival_estimate = self.survival_estimate[time.astype("int32")].astype("float32")

        # Creating tensors
        time = torch.from_numpy(time)
        event = torch.from_numpy(event)
        if cfg.DATA.DEATH_AT_CENSOR_TIME:
            event = torch.ones(event.size())
        X = torch.from_numpy(X)

        if survival_estimate.shape[0] != X.shape[0]:
            survival_estimate = survival_estimate[:X.shape[0], :]

        if self.div_time != 1:
            step_time = [i for i in range(1, survival_estimate.shape[1], self.div_time)]

            surv = []
            for stp in range(len(step_time)):
                if stp < len(step_time) - 1:
                    surv.append(np.median(survival_estimate[:, step_time[stp]:step_time[stp + 1]], axis=1))
                else:
                    surv.append(np.median(survival_estimate[:, step_time[stp]:], axis=1))

            survival_estimate = np.array(surv).T

        survival_estimate = torch.from_numpy(survival_estimate)

        if cfg.CUDA:
            time = time.cuda()
            event = event.cuda()
            X = X.cuda()
            survival_estimate = survival_estimate.cuda()

        time_output = self.model(X)
        time_output = F.softmax(time_output, 1)

        # Prepare target
        time_step = torch.arange(0, self.dtime_shape).unsqueeze(0).repeat(X.size()[0], 1)
        if cfg.CUDA:
            time_step = time_step.cuda()

        mat_time = time.unsqueeze(0).repeat(self.dtime_shape, 1).transpose(0, 1)
        time_step = time_step - mat_time
        time_step = time_step >= 0
        time_step = time_step.float()

        time_step_cens = time_step * (1. - survival_estimate)
        event_cases = event.unsqueeze(1).repeat(1, self.dtime_shape)
        time_step = time_step * event_cases
        time_step_cens = time_step_cens * (event_cases == 0).float()
        cdf_time = time_step + time_step_cens

        ones = torch.ones(X.size()[0], 1)
        if cfg.CUDA:
            ones = ones.cuda()

        cdf_time_ = torch.cat((cdf_time, ones), 1)
        cdf_pred_ = torch.cumsum(time_output, 1)
        loss = self.emd_loss(cdf_pred_, cdf_time_, self.distance_mat)

        rank_output = -torch.mm(cdf_pred_, self.distance_mat.view(-1, 1))

        concat_pred = torch.cat((time.unsqueeze(-1), event.unsqueeze(-1)), 1)
        concat_pred = torch.cat((concat_pred, rank_output), 1)

        return concat_pred, loss
