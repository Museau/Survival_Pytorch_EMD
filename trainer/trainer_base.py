import os
import pprint

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from datasets.dataset_loader import load_data
from utils.config import cfg
from utils.utils import mkdir_p, iterate_minibatches, tgt_equal_tgt, tgt_leq_tgt
from visualization.figures import plot_train_val_history


class TrainerBase(object):

    def __init__(self, split):
        """
        Trainer base class object.

        Parameters
        ----------
        split : int
            Split number.
        """

        pprint.pprint(cfg)

        self.verbose = bool(cfg.VERBOSE)
        self.dataset = cfg.DATA.DATASET
        self.split = split

        self.data_path = cfg.DATA.PATH

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_patience = cfg.TRAIN.PATIENCE

        self.num_epochs = cfg.TRAIN.MAX_EPOCH
        self.model_name = cfg.TRAIN.MODEL
        self.loss_type = cfg.TRAIN.LOSS_TYPE

        self.get_data()

        self.model = None

    def process_batch(self, data):
        """
        Process batch of data to extract the time, the event
        and the explanatory variables matrix and, to compute
        the lower triangle and diagonal by block matrix.

        Parameters
        ----------
        data : ndarray
            Data to process.

        Returns
        -------
        time : ndarray
            Time.
        event : ndarray
            Event.
        X : ndarray
            Explanatory variables matrix.
        tril : ndarray
            Lower triangular matrix.
        tied_matrix : ndarray
            Diagonal by block matrix.
        """
        time = data[:, 0]
        event = data[:, 1]
        X = data[:, 2:]

        # Ascending sort by time
        argsort_time = np.argsort(time)

        time = time[argsort_time]
        event = event[argsort_time]
        X = X[argsort_time]

        tril = tgt_leq_tgt(time)
        tied_matrix = tgt_equal_tgt(time)

        return time, event, X, tril, tied_matrix

    def get_data(self):
        """
        Load the data and extract input (and output in the case of EMD) shapes
        to be use by the model.
        """
        data, dict_col = load_data()
        self.X_train_shape = (-1, data.shape[1] - 2)
        if cfg.TRAIN.MODEL == "emd":
            self.time_shape = int(data[:, 0].max()) + 1
        self.data = data
        self.dict_col = dict_col

    def before_train(self, train):
        """
        Place holder for  a function that could
        do additional operation on train set before
        starting training.

        Parameters
        ----------
        train : ndarray
            Training set.
        """
        pass

    def get_pred_loss(self, batch):
        """
        Place holder for the function that compute the model
        loss and prediction.

        Parameters
        ----------
        batch : ndarray
            Batch data.
        """
        raise

    def run(self):
        """
        Run the Training/validation loop and then the test loop.

        Returns
        -------
        scores : dict
            Dictionary containing the scores ('c-index' and 'avg_loss')
            for 'train', 'val' and 'test' sets.
        concat_pred_test : ndarray
            Matrix containing the predictions for the test set.
        """
        train, val, test = self.get_data_random_split()
        self.before_train(train)
        split_id = f"split{self.split}"

        scores = {}

        train_err_history = []
        val_err_history = []
        train_cindex_history = []
        val_cindex_history = []

        patience = 0

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=float(cfg.TRAIN.LR), weight_decay=float(cfg.TRAIN.L2_COEFF))

        best_c_index = -np.inf

        for epoch in range(self.num_epochs):

            concat_pred_train = np.array([]).reshape(0, 3)
            concat_pred_val = np.array([]).reshape(0, 3)

            train_epoch_loss = 0
            val_epoch_loss = 0

            train_iteration = 0
            val_iteration = 0

            for mode in ["train", "val"]:
                if mode == "train":
                    if self.verbose:
                        print(f"\nRunning training epoch {epoch} ...")
                    self.model.train()
                    shuffle_batch = True

                elif mode == "val":
                    if self.verbose:
                        print(f"Running validation epoch {epoch} ...")
                    self.model.eval()
                    shuffle_batch = False

                for batch in iterate_minibatches(train if mode == "train" else val, self.batch_size, shuffle=shuffle_batch):

                    concat_pred, loss = self.get_pred_loss(batch)

                    if mode == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        concat_pred_train = np.concatenate((concat_pred_train, concat_pred.data.cpu().numpy()), axis=0)

                        train_epoch_loss += loss.data.item()
                        train_iteration += 1

                    elif mode == "val":
                        concat_pred_val = np.concatenate((concat_pred_val, concat_pred.data.cpu().numpy()), axis=0)

                        val_epoch_loss += loss.data.item()
                        val_iteration += 1

            # Record and print result after each epoch
            # Train
            train_full_epoch_loss = train_epoch_loss / train_iteration
            train_err_history.append(train_full_epoch_loss)
            if self.verbose:
                print(f"===> Epoch {epoch} Complete: Avg. Train Loss: {train_full_epoch_loss:.4f}")

            # concordance_index(time, pred, event)
            train_c_index = concordance_index(concat_pred_train[:, 0], concat_pred_train[:, 2], concat_pred_train[:, 1])

            if train_c_index < 0.5:
                # concordance_index(time, pred, event)
                train_c_index = concordance_index(concat_pred_train[:, 0], -concat_pred_train[:, 2], concat_pred_train[:, 1])

            train_cindex_history.append(train_c_index)

            if self.verbose:
                print(f"===> Epoch {epoch} Complete: Train C-index: {train_c_index:.4f}")

            # Val
            val_full_epoch_loss = val_epoch_loss / val_iteration
            val_err_history.append(val_full_epoch_loss)
            if self.verbose:
                print(f"===> Epoch {epoch} Complete: Avg. Val Loss: {val_full_epoch_loss:.4f}")

            # concordance_index(time, pred, event)
            val_c_index = concordance_index(concat_pred_val[:, 0], concat_pred_val[:, 2], concat_pred_val[:, 1])

            if val_c_index < 0.5:
                # concordance_index(time, pred, event)
                val_c_index = concordance_index(concat_pred_val[:, 0], -concat_pred_val[:, 2], concat_pred_val[:, 1])

            val_cindex_history.append(val_c_index)
            if self.verbose:
                print(f"===> Epoch {epoch} Complete: Val C-index: {val_c_index:.4f}")

            # Plot training and validation curve
            path = os.path.join(cfg.OUTPUT_DIR, cfg.PARAMS, "Figures/")
            mkdir_p(os.path.dirname(path))

            plot_train_val_history(path, f"error_{split_id}", train_err_history, val_err_history)

            plot_train_val_history(path, f"c_index_{split_id}", train_cindex_history, val_cindex_history)

            if val_c_index > best_c_index:
                if self.verbose:
                    print("Saving best model")
                best_epoch = epoch
                best_c_index = val_c_index
                scores['train'] = {'avg_loss': train_full_epoch_loss, 'c_index': train_c_index}
                scores['val'] = {'avg_loss': val_full_epoch_loss, 'c_index': val_c_index, 'best_epoch': epoch}
                self.save_model("best" + split_id)
                patience = 0
            else:
                patience += 1

            if patience > self.max_patience:
                if self.verbose:
                    print("Max patience reached, ending training")
                break

        if self.verbose:
            print("\nDone training, evaluate on test...")

        concat_pred_test = np.array([]).reshape(0, 3)

        test_epoch_loss = 0
        test_iteration = 0

        # Loading weights of the best model for the test
        if self.verbose:
            print(f"Best epoch: {best_epoch}")
            print(f"Best val C-index: {best_c_index}")

        self.load_model_best(split_id)
        self.model.eval()

        for batch in iterate_minibatches(test, self.batch_size, shuffle=False):

            concat_pred, loss = self.get_pred_loss(batch)

            concat_pred_test = np.concatenate((concat_pred_test, concat_pred.data.cpu().numpy()), axis=0)

            test_epoch_loss += loss.data.item()
            test_iteration += 1

        test_full_epoch_loss = test_epoch_loss / test_iteration
        if self.verbose:
            print(f"===> Avg. Test Loss: {test_full_epoch_loss:.4f}")

        # concordance_index(time, pred, event)
        test_c_index = concordance_index(concat_pred_test[:, 0], concat_pred_test[:, 2], concat_pred_test[:, 1])

        if test_c_index < 0.5:
            # concordance_index(time, pred, event)
            test_c_index = concordance_index(concat_pred_test[:, 0], -concat_pred_test[:, 2], concat_pred_test[:, 1])

        if self.verbose:
            print(f"===> Test C-index: {test_c_index:.4f}")

        scores['test'] = {'avg_loss': test_full_epoch_loss, 'c_index': test_c_index}

        return scores, concat_pred_test

    def get_data_random_split(self):
        """
        Get data split train/val/test from  5Fold cross-validation.

        Returns
        -------
        train : ndarray
            Training set.
        val : ndarray
            Validation set.
        test : ndarray
            Test set.
        """

        kf = KFold(n_splits=5)

        index_train = []
        index_valid = []
        index_test = []
        for train, test in kf.split(self.data):
            index_train.append(train[:int(len(self.data)*0.6)])
            index_valid.append(train[int(len(self.data)*0.6):])
            index_test.append(test)

        if self.split is not None:
            train = self.data[index_train[self.split]]
            val = self.data[index_valid[self.split]]
            test = self.data[index_test[self.split]]
        else:
            raise NotImplementedError()

        # Normalise the data
        col_name = ["time", "event"] + self.dict_col['col']
        df_train = pd.DataFrame(data=train, columns=col_name)
        df_val = pd.DataFrame(data=val, columns=col_name)
        df_test = pd.DataFrame(data=test, columns=col_name)
        scaler = MinMaxScaler()
        df_train[self.dict_col['continuous_keys']] = scaler.fit_transform(df_train[self.dict_col['continuous_keys']])
        df_val[self.dict_col['continuous_keys']] = scaler.transform(df_val[self.dict_col['continuous_keys']])
        df_test[self.dict_col['continuous_keys']] = scaler.transform(df_test[self.dict_col['continuous_keys']])

        train = df_train.to_numpy()
        val = df_val.to_numpy()
        test = df_test.to_numpy()

        if cfg.DATA.ADD_CENS:
            proba = cfg.DATA.PROBA
            cens = train[train[:, 1] == 0]
            non_cens = train[train[:, 1] == 1]

            # Add censure cases in the event feature
            p_ = proba - (cens.shape[0] / float(train.shape[0]))
            p_ = (train.shape[0] * p_) / float(non_cens.shape[0])
            ev_new = np.random.binomial(size=non_cens.shape[0], n=1, p=1-p_)
            non_cens[:, 1] = ev_new

            # Modify target for new censured cases
            new_cens = non_cens[non_cens[:, 1] == 0]
            non_cens = non_cens[non_cens[:, 1] == 1]
            tgt_ = new_cens[:, 0]
            g_rand = lambda x: np.random.randint(x)
            new_tgt = list(map(g_rand, tgt_))
            new_cens[:, 0] = new_tgt

            train = np.concatenate((cens, new_cens), axis=0)
            train = np.concatenate((train, non_cens), axis=0)

        if cfg.TRAIN.MODEL == "emd":
            kmf = KaplanMeierFitter()

            y_train = train[:, 0]
            y_train_cens = train[:, 1]

            kmf.fit(y_train, event_observed=y_train_cens)
            timeline = np.array(kmf.survival_function_.index)
            KM_estimate = kmf.survival_function_.values.flatten()

            survival = []
            for i in range(self.time_shape):
                surv = np.zeros(self.time_shape)
                prob = 1.
                for i in range(self.time_shape):
                    if i in timeline:
                        idx = np.where(timeline == i)[0]
                        prob = KM_estimate[idx]
                    surv[i] = prob
                survival.append(surv)

            self.survival_estimate = np.array(survival)

        if cfg.DATA.NO_CENSORED_DATA:
            train = train[train[:, 1] == 1]

        return train, val, test

    def save_model(self, message):
        """
        Save the model.

        Parameters
        ----------
        message : srt
            Message to include in the path of the model to save.
        """
        path = os.path.join(cfg.OUTPUT_DIR, cfg.PARAMS, "Models/")
        mkdir_p(os.path.dirname(path))

        path_model = f"{path}model_epoch_{message}.pth"
        torch.save(self.model.state_dict(), path_model)

    def load_model_best(self, split_id):
        """
        Load best model for a given split id.

        Parameters
        ----------
        split_id : str
            Split id under the form "split{split#}".
        """
        path = os.path.join(cfg.OUTPUT_DIR, cfg.PARAMS, "Models/")
        path = f"{path}model_epoch_best{split_id}.pth"

        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        print(f"\nLoad from: {path}")
