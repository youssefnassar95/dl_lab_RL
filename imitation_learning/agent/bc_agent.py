from tkinter.tix import Y_REGION
import torch
from agent.networks import CNN
import numpy as np  # noqa: E402


class BCAgent:

    def __init__(self):
        # TODO: Define network, loss function, optimizer
        self.net = CNN()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=0.001, momentum=0.9)

        pass

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        X_batch = torch.Tensor(X_batch)
        y_batch = torch.LongTensor(y_batch)

        # TODO: forward + backward + optimize
        outputs = self.net(X_batch)
        loss = self.criterion(outputs, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, outputs

    def predict(self, X):
        # TODO: forward pass
        X = torch.Tensor(X)
        outputs = self.net(X)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
