import torch.nn as nn

class FraudDetector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional):
        super(FraudDetector, self).__init__()
        self.pre_net = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.mult = 2 if bidirectional else 1
        self.post_net = nn.Linear(hidden_size * self.mult, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pre_net(x)
        out, (h, c) = self.lstm(x)
        out = self.post_net(out)

        # return the last value for prediction
        return self.sigmoid(out[:, -1, 0])