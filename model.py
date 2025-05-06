import torch
import torch.nn.functional as F


class LSTM_ASR(torch.nn.Module):
    def __init__(self, feature_type="discrete", input_size=64, hidden_size=256, num_layers=2,
                 output_size=28):
        super().__init__()
        assert feature_type in ['discrete', 'mfcc']
        # Build your own neural network. Play with different hyper-parameters and architectures.
        # === write your code here ===
        self.feature_type = feature_type

        if feature_type == 'discrete':
            self.embedding = torch.nn.Embedding(input_size, hidden_size)
            self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

 
        else:
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, batch_features):
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """
        # === write your code here ===
        if self.feature_type == 'discrete':
            x = self.embedding(batch_features)
            x, _ = self.lstm(x)
            x = self.fc(x)

        else:

            x, _ = self.lstm(batch_features)
            x = self.fc(x)

        x = F.log_softmax(x, dim=-1)
        return x
