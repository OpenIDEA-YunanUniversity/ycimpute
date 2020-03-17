from ..utils.tools import Solver
from ..nn.autoencoder import Autoencoder

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

class MIDA(Solver):
    def __init__(
            self,
            theta=5,
            epochs=300,
            use_cuda=False,
            batch_size=64,
            early_stop=1e-06,
            normalizer='min_max',
            verbose=True):

        Solver.__init__(
            self,
            normalizer=normalizer)

        self.theta = theta
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.verbose = verbose
        self.early_stop = early_stop

        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

    def training(self, training_data):
        n_features = training_data.shape[1]
        training_data = torch.from_numpy(training_data).float()

        train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        model = Autoencoder(dim=n_features,
                            theta=self.theta).to(self.device)
        loss = nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001,momentum=0.8)

        cost_list = []
        early_stop = False
        total_batch = len(training_data) // self.batch_size

        for epoch in range(self.epochs):
            for i, batch_data in enumerate(train_loader):
                batch_data = batch_data.to(self.device)
                reconst_data = model(batch_data)
                cost = loss(reconst_data, batch_data)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                if self.verbose:
                    if (i + 1) % (total_batch // 2) == 0:
                        print('Epoch [%d/%d], lter [%d/%d], Loss: %.6f' %
                              (epoch + 1, self.epochs, i + 1, total_batch, cost.item()))

                # early stopping rule 1 : MSE < 1e-06
                if cost.item() < 1e-06:
                    early_stop = True
                    break

                cost_list.append(cost.item())

            if early_stop:
                break
        return model


    def solve(self, X, missing_mask):
        complete_rows_index, missing_rows_index = self.detect_complete_part(missing_mask)
        if len(complete_rows_index)==0:
            raise ValueError('Cant find a completely part for training...')
        missing_data = X[missing_rows_index]
        training_data = X[complete_rows_index]

        model = self.training(training_data.copy())
        model.eval()

        missing_data = torch.from_numpy(missing_data).float()
        filled_data = model(missing_data.to(self.device))
        filled_data = filled_data.cpu().detach().numpy()
        tmp_mask = missing_mask[missing_rows_index]
        missing_data = missing_data.cpu().numpy()
        filled_data = missing_data * (1 - tmp_mask) + filled_data * (tmp_mask)

        X[missing_rows_index] = filled_data
        X[complete_rows_index] = training_data

        return X