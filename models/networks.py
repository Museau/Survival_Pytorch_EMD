import torch.nn as nn

from torch.nn import ModuleList, Dropout, Linear, ReLU, BatchNorm1d

from utils.config import cfg


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(MLP, self).__init__()
        """
        Constructing the network.

        Parameters
        ----------
        input_shape : tuple
            Input shape (batch_size, num_features).
        output_shape : int
            Output shape.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.drop_input = cfg.TRAIN.DROP_INPUT
        self.drop_hidden = eval(cfg.TRAIN.DROP_HIDDEN)
        self.layer_sizes = eval(cfg.TRAIN.LAYER_SIZES)
        self.batch_norm = cfg.TRAIN.BATCH_NORM

        # Constructing network
        self.layers = []
        prev = self.input_shape[1]
        if self.drop_input > 0.:
            self.layers.append(Dropout(p=self.drop_input, inplace=False))

        for i in range(len(self.layer_sizes)):
            self.layers.append(Linear(prev, self.layer_sizes[i]))
            self.layers.append(ReLU())
            if self.drop_hidden[i] > 0.:
                self.layers.append(Dropout(p=self.drop_hidden[i], inplace=False))
            if self.batch_norm:
                self.layers.append(BatchNorm1d(self.layer_sizes[i]))
            prev = self.layer_sizes[i]

        self.layers = ModuleList(self.layers)
        self.output = Linear(self.layer_sizes[-1], output_shape)

    def forward(self, x):
        """
        x :  torch tensor
            The inputs to the model.
        """
        for i, lay in enumerate(self.layers):
            x = lay(x)

        output = self.output(x)

        return output
