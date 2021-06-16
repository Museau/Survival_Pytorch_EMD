import torch

from models.networks import MLP
from trainer.trainer_base import TrainerBase
from utils.config import cfg
from utils.loss import cox_loss_basic, cox_loss_ties, rank_loss


class LikelihoodTrainer(TrainerBase):
    def __init__(self, split):
        """
        Likelihood trainer.

        Parameters
        ----------
        split : int
            Split number.
        """
        super(LikelihoodTrainer, self).__init__(split)

        self.model = MLP(input_shape=self.X_train_shape, output_shape=1)
        if cfg.CUDA:
            self.model.cuda()

        if self.loss_type == "rank_loss":
            self.rank_function = cfg.TRAIN.F_RANK

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
        time, event, X, tril, tied_matrix = self.process_batch(batch)

        # Creating tensors
        time = torch.from_numpy(time)
        event = torch.from_numpy(event)
        if cfg.DATA.DEATH_AT_CENSOR_TIME:
            event = torch.ones(event.size())
        X = torch.from_numpy(X)
        if "cox" in self.loss_type:
            tril = torch.from_numpy(tril)
            tied_matrix = torch.from_numpy(tied_matrix)

        if cfg.CUDA:
            time = time.cuda()
            event = event.cuda()
            X = X.cuda()
            if "cox" in self.loss_type:
                tril = tril.cuda()
                tied_matrix = tied_matrix.cuda()

        pred = self.model(X)

        concat_pred = torch.cat((time.unsqueeze(-1), event.unsqueeze(-1)), 1)
        concat_pred = torch.cat((concat_pred, pred), 1)

        if self.loss_type == "cox_loss_ties":
            loss = cox_loss_ties(pred, event, tril, tied_matrix)
        elif self.loss_type == "cox_loss_basic":
            loss = cox_loss_basic(pred, event, tril, tied_matrix)
        elif self.loss_type == "rank_loss":
            loss = rank_loss(pred, time, event, f=self.rank_function)
        else:
            raise ValueError("self.loss_type invalid")

        return concat_pred, loss
