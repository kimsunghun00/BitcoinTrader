from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, input_size, recurrent_hidden_size, dense_hidden_size, output_size):
        super(Model, self).__init__()

        self.gru = nn.GRU(input_size=input_size, hidden_size=recurrent_hidden_size,
                          bidirectional=True, num_layers=2, batch_first=True)

        self.dense1 = nn.Linear(recurrent_hidden_size * 2, dense_hidden_size, bias=True)
        self.dense2 = nn.Linear(dense_hidden_size, 128, bias=True)
        self.out = nn.Linear(128, output_size, bias=False)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.out(x)
        return x