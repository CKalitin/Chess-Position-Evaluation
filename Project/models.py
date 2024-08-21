import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__() # Makes this class a delegate of torch.nn.
        
        # Input: 1x8x8, Output: 64x5x5
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4,4), stride=1, padding=0)
        self.act1 = nn.Tanh()
        
        # Input: 64x5x5, Output: 64x2x2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(2,2), stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.flat = nn.Flatten(start_dim=0) # Flatten from first dimension, wth was that error?
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.act1(self.conv1(x))
        
        x = self.act2(self.conv2(x))
        x = self.maxpool2(x)
        
        x = self.flat(x)
        x = self.fc3(x)
        
        return x
        
class QTrainer():
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss() # Mean Squared Error Loss
    
    def train_step(self, state, model_output, reward):
        # Tuples to list: 
        state = torch.stack([s.clone().detach() for s in state])
        model_output = [m for m in model_output]
        reward = [r for r in reward]
        
        q_value = torch.zeros(len(state))
        for i in range(len(state)): q_value[i] = self.model(state[i])
        
        target = q_value.clone()
        for i in range(len(target)): target[i] = reward[i] + self.gamma * model_output[i]
        
        self.optimizer.zero_grad()
        loss = self.criterion(q_value, target)
        loss.backward()
        self.optimizer.step()
