import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import DeterministicMixin, CategoricalMixin, Model
import torch.nn.functional as F
import numpy as np
import sys, numpy
torch.set_printoptions(threshold= 10_000)

class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space = None, action_space = None, device= None, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.grid_size = 10
        self.hidden_size = 1024
        self.drop_rate = 0.02

        # Input channels: 1 (grayscale), output channels: 64, kernel size: 3x3
        self.cl1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding = 1, padding_mode="replicate")

        self.input_layer = nn.Linear(6400, 4096)
        self.hl1 = nn.Linear(4096, 4096)
        self.hl2 = nn.Linear(4096, 2048)
        self.hl3 = nn.Linear(2048, self.hidden_size)
        self.hl4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hl5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hl6 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hl7 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.hl8 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.hl9 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hl10 = nn.Linear(self.hidden_size, self.hidden_size)
        self.logits = nn.Linear(self.hidden_size, self.num_actions)

        self.dropout1 = nn.Dropout(p=self.drop_rate)
        self.dropout2 = nn.Dropout(p=self.drop_rate)
        self.dropout3 = nn.Dropout(p=self.drop_rate)
        self.dropout4 = nn.Dropout(p=self.drop_rate)
        self.dropout4 = nn.Dropout(p=self.drop_rate)
        self.dropout5 = nn.Dropout(p=self.drop_rate)
        self.dropout6 = nn.Dropout(p=self.drop_rate)
        self.dropout7 = nn.Dropout(p=self.drop_rate)

    def compute(self, inputs, role):
        if type(inputs) is dict:
            inputs = inputs["states"]
        height_map = inputs[:, 333:].view(inputs.shape[0], 1, 40, 40)
        box = inputs[:, :3].unsqueeze(2).unsqueeze(3)
        box = box.repeat(1, 1, 40, 40)

        input_stack = torch.cat((height_map, box), dim=1)

        ### convolution ###
        x = self.cl1(input_stack)
        x = F.relu(x)
        x = x.view(x.size(0), 6400)
        
        ### feed forward ###
        x = self.input_layer(x)
        x = F.relu(x)
        
        x = self.hl1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.hl2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.hl3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.hl4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.hl5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.hl6(x)
        x = F.relu(x)
        x = self.dropout6(x)
        x = self.hl7(x)
        x = F.relu(x)
        x = self.dropout7(x)
        # x = self.hl8(x)
        # x = F.relu(x)
        # x = self.hl9(x)
        # x = F.relu(x)
        x = self.hl10(x)
        x = F.relu(x)

        x = self.logits(x)
        return x, {}
