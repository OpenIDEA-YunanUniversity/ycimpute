from  ..utils.tools import generate_noise
from ..utils.tools import Solver
from ..nn.gainnets import NetD,NetG

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader


def generate_hint(n_rows, n_cols, missing_rate):
    """
    @n_rows: number of rows to generate missing matrix
    @n_cols: number of columns to generate missing matrix
    """


    random_data = np.random.uniform(0., 1., size=[n_rows, n_cols])
    tmp = random_data > missing_rate
    missing_mat = 1. * tmp

    return missing_mat


class SimpleDataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, specify_data, mask):
        """
        """
        self.specify_data = specify_data
        self.mask = mask

    def __len__(self):
        return len(self.specify_data)

    def __getitem__(self, idx):
        data = self.specify_data[idx]
        mask = self.mask[idx]

        return data, mask

class GAIN(Solver):
    def __init__(self,
                 normalizer='min_max',
                 epochs=10,
                 use_cuda=False,
                 batch_size=64,
                 verbose=True,
                 alpha = 0.2,
                 lr = 0.0001,
                 hint_rate=0.2,
                ):
        Solver.__init__(self,
            normalizer=normalizer)

        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.verbose = verbose
        self.hint_rate = hint_rate
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")


    def training(self,training_data,train_mask):
        train_mask = ~train_mask
        train_mask = train_mask.astype(int)

        _, n_cols = training_data.shape
        netD = NetD(feature_dim=n_cols).to(self.device)
        netG = NetG(feature_dim=n_cols).to(self.device)
        optimD = torch.optim.RMSprop(netD.parameters(), lr=self.lr)
        optimG = torch.optim.RMSprop(netG.parameters(), lr=self.lr)

        train_dset = SimpleDataLoader(training_data,train_mask)
        train_loder = DataLoader(train_dset,
                                 batch_size=self.batch_size,
                                 num_workers=1)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean")
        mse_loss = torch.nn.MSELoss(reduction="elementwise_mean")

        for epoch in range(self.epochs):
            for idx, (x, mask) in enumerate(train_loder):
                noise = generate_noise(x.shape[0], x.shape[1])
                hint = generate_hint(x.shape[0], x.shape[1], self.hint_rate)

                x = torch.tensor(x).float().to(self.device)
                noise = torch.tensor(noise).float().to(self.device)
                mask = torch.tensor(mask).float().to(self.device)
                hint = torch.tensor(hint).float().to(self.device)

                hint = mask * hint + 0.5 * (1 - hint)

                # train D
                optimD.zero_grad()
                G_sample = netG(x, noise, mask)

                D_prob = netD(x, mask, G_sample, hint)
                D_loss = bce_loss(D_prob, mask)
                D_loss.backward()
                optimD.step()
                # train G
                optimG.zero_grad()
                G_sample = netG(x, noise, mask)

                D_prob = netD(x, mask, G_sample, hint)

                D_prob.detach_()
                G_loss = ((1 - mask) * (torch.sigmoid(D_prob) + 1e-8).log()).mean() / (1 - mask).sum()+0.001
                G_mse_loss = mse_loss(mask * x, mask * G_sample) / mask.sum()+0.0001
                G_loss = G_loss + self.alpha * G_mse_loss

                G_loss.backward()
                optimG.step()

                G_mse_train = mse_loss((mask) * x, (mask) * G_sample) / (mask).sum()
                if self.verbose:
                    if epoch % 2 == 0:
                        print('Iter:{}\tD_loss: {:.4f}\tG_loss: {:.4f}\tTrain MSE:{:.4f}'. \
                            format(epoch, D_loss, G_loss, np.sqrt(G_mse_train.data.cpu().numpy())))

        return netG


    def solve(self, X,missing_mask):
        complete_rows_index, missing_rows_index = self.detect_complete_part(missing_mask)
        if len(complete_rows_index) == 0:
            raise ValueError('Cant find a completely part for training...')
        model = self.training(training_data=X.copy(),train_mask=missing_mask.copy())
        model.eval()

        missing_mask = ~missing_mask
        missing_mask = missing_mask.astype(int)

        noise = generate_noise(X.shape[0], X.shape[1])
        noise = torch.tensor(noise).float().to(self.device)
        X = torch.tensor(X).float().to(self.device)
        mask = torch.tensor(missing_mask).float().to(self.device)

        filled_data = model(X,noise,mask)
        filled_data = filled_data.cpu().detach().numpy()

        X = X.cpu().detach().numpy()
        X[missing_rows_index] = filled_data[missing_rows_index]

        return X

