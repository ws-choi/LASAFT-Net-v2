import torch.nn as nn
from torch.nn.functional import one_hot

class LinearEmbedding(nn.Module):
    def __init__(self, control_input_dim, embedding_dim):
        super().__init__()

        self.control_input_dim = control_input_dim
        self.embedding_dim = embedding_dim

        self.linear = nn.Linear(control_input_dim, embedding_dim, bias=False)

    def forward(self, conditions):
        if conditions.dim() < 2:
            conditions = one_hot(conditions, self.control_input_dim).to(self.linear.weight.dtype)
        return self.linear(conditions)
