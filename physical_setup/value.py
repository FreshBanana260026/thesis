import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import DeterministicMixin, CategoricalMixin, Model
import torch.nn.functional as F

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.drop_rate = 0.02
        self.cl1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding = 1, padding_mode="replicate")
        
        self.input = nn.Linear(6400, 4096)
        
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, 512)
        # self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 256)
        self.fc10 = nn.Linear(256, 128)
        self.fc11 = nn.Linear(128, 1)

        self.dropout1 = nn.Dropout(p=self.drop_rate)
        self.dropout2 = nn.Dropout(p=self.drop_rate)
        self.dropout3 = nn.Dropout(p=self.drop_rate)
        self.dropout4 = nn.Dropout(p=self.drop_rate)
        self.dropout5 = nn.Dropout(p=self.drop_rate)
        self.dropout6 = nn.Dropout(p=self.drop_rate)
        self.dropout7 = nn.Dropout(p=self.drop_rate)
        # self.dropout8 = nn.Dropout(p=self.drop_rate)
        self.dropout9 = nn.Dropout(p=self.drop_rate)

    def compute(self, inputs, role):
        if type(inputs) is dict:
            inputs = inputs["states"]
        # height_map = inputs[:, 333:]
        # box = inputs[:, :3]
        # box = box.repeat(1, 100)
        # input = torch.cat((height_map, box), 1)
        height_map = inputs[:, 333:].view(inputs.shape[0], 1, 40, 40)
        box = inputs[:, :3].unsqueeze(2).unsqueeze(3)
        box = box.repeat(1, 1, 40, 40)

        input_stack = torch.cat((height_map, box), dim=1)

        x = self.cl1(input_stack)
        x = F.relu(x)
        x = x.view(x.size(0), 6400)
        
        x = self.input(x)
        x = F.relu(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.dropout7(x)
        # x = self.fc8(x)
        # x = F.relu(x)
        # x = self.dropout8(x)
        x = self.fc9(x)
        x = F.relu(x)
        x = self.dropout9(x)
        x = self.fc10(x)
        x = F.relu(x)
        
        return self.fc11(x), {}