"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import numpy as np

def RParam(*shape):
    r = 2 * (minitorch.rand(shape, requires_grad=True) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # TODO: Implement for Task 2.5.
        x = self.layer1.forward(x).relu()
        x = self.layer2.forward(x).relu()
        x = self.layer3.forward(x).sigmoid()
        return x


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size
        
    def forward(self, x):
        # TODO: Implement for Task 2.5.
        # x should be of in_size, self.weights is a matrix of in_size, out_size that we multiply onto it, using zip? then add self.bias as a broadcast
        # we could do a summed dot product because we don't have matmul implemented yet

        # x has shape )batch_size, in_size and weights has shape (in_size, out_size)
        # so we want to multiply across the in_size dimensions
        batch_size, in_size = x.shape
        x = x.view(batch_size, in_size, 1)
        weights = self.weights.value.view(1, in_size, self.out_size)

        # now we cna perform broadcast multiplication, then reduce  
        p1 = weights * x # nwo of shape (batch-size, in_size, out_size)
        # self.weights.value = self.weights.value.view(*weights_shape) # (1, in_size, out_size)
        p1 = p1.sum(dim=1)
        p1 = p1.view(batch_size, self.out_size)
        # print("aft", x._tensor.shape)
        p1 = p1 + self.bias.value.view(1, self.out_size)
        return p1


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Circle"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
