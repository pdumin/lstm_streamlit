import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class ConfigRNN:
    vocab_size: int
    device: str
    n_layers: int
    embedding_dim: int
    hidden_size: int
    seq_len: int
    bidirectional: bool or int

net_config = ConfigRNN(
    vocab_size=967 + 1,
    device="cpu",
    n_layers=1,
    embedding_dim=8,
    hidden_size=16,
    seq_len=32,
    bidirectional=False,
)

class LSTMClassifier(nn.Module):
    def __init__(self, rnn_conf=net_config) -> None:
        super().__init__()

        self.embedding_dim = rnn_conf.embedding_dim
        self.hidden_size = rnn_conf.hidden_size
        self.bidirectional = rnn_conf.bidirectional
        self.n_layers = rnn_conf.n_layers

        self.embedding = nn.Embedding(rnn_conf.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.n_layers,
        )
        self.bidirect_factor = 2 if self.bidirectional else 1
        self.clf = nn.Sequential(
            nn.Linear(self.hidden_size * self.bidirect_factor, 32),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(32, 1)
        )

    def model_description(self):
        direction = "bidirect" if self.bidirectional else "onedirect"
        return f"lstm_{direction}_{self.n_layers}"

    def forward(self, x: torch.Tensor):
        embeddings = self.embedding(x)
        out, _ = self.lstm(embeddings)
        # print(out.shape)
        out = out[:, -1, :]  # [все элементы батча, последний h_n, все элементы последнего h_n]
        # print(out.shape)
        out = self.clf(out)
        return out