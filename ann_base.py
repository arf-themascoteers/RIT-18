import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, root_mean_squared_error
import utils


class ANNBase(nn.Module):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__()
        self.verbose = True
        self.TEST = False
        self.device = utils.get_device()
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.validation_ds = validation_ds
        self.num_epochs = 1000
        if utils.is_test():
            self.num_epochs = 3
        self.batch_size = 6000
        self.lr = 0.001

    def train_model(self):
        if self.TEST:
            return
        self.train()
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        total_batch = len(dataloader)
        for epoch in range(self.num_epochs):
            for batch_number, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self(x)
                loss = criterion(y_hat, y)

                if self.verbose:
                    _, predicted = torch.max(y_hat, 1)
                    total = y.size(0)
                    train_correct = (predicted == y).sum().item()
                    train_accuracy = train_correct / total

                    y, y_hat = self.evaluate(self.validation_ds)
                    _, predicted = torch.max(y_hat, 1)
                    total = y.size(0)
                    val_correct = (predicted == y).sum().item()
                    val_accuracy = val_correct / total

                    print(f'Epoch:{epoch} (of {self.num_epochs}), Batch: {batch_number+1} of {total_batch}, '
                          f'Loss:{loss.item():.6f}, '
                          f'Train Acc: {train_accuracy:.3f}, Val Acc: \033[91m {val_accuracy:.3f} \033[0m ', end=""
                          )
                    self.verbose_after(self.validation_ds)
                    print("")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            self.after_epoch(epoch)

    def after_epoch(self, epoch):
        pass

    def verbose_after(self, ds):
        pass

    def evaluate(self, ds):
        batch_size = 6000
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        y_all = torch.zeros(0).to(torch.long)
        y_hat_all = torch.zeros(0)

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self(x)

            y = y.detach().cpu()
            y_hat = y_hat.detach().cpu()

            y_all = torch.concatenate((y_all, y))
            y_hat_all = torch.concatenate((y_hat_all, y_hat))

        return y_all, y_hat_all

    def test(self):
        self.eval()
        self.to(self.device)
        y_all, y_hat_all = self.evaluate(self.test_ds)
        return y_hat_all

    def metrics(self, ds):
        self.eval()
        self.to(self.device)
        y_all, y_hat_all = self.evaluate(ds)
        _, predicted = torch.max(y_hat_all, 1)
        total = y_all.size(0)
        val_correct = (predicted == y_all).sum().item()
        val_accuracy = val_correct / total
        pc = self.pc(ds)
        #return val_accuracy, pc
        return val_accuracy

    def run(self):
        self.train_model()
        return self.metrics(self.test_ds)

    def pc(self, ds):
        y_all, y_hat_all = self.evaluate(ds)
        return utils.calculate_pc(y_all, y_hat_all)