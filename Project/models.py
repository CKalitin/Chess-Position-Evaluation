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
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def fake_train_step(self, state, action, reward):
        # Tuples to list: 
        state = torch.stack([s.clone().detach() for s in state])
        action = [m for m in action]
        reward = [r for r in reward]
        
        q_value = torch.zeros(len(state), 3)
        for i in range(len(state)): q_value[i] = self.model(state[i])
        
        target = q_value.clone()
        for i in range(len(target)): target[i][torch.argmax(action[i]).item()] = reward[i]
        
        self.optimizer.zero_grad()
        loss = self.criterion(q_value, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_step(self, state, outcome):
        state = torch.stack(state)
        outcome = torch.tensor(outcome, dtype=torch.float32)
        
        pred_action = torch.tensor([self.model(s) for s in state], dtype=torch.float32, requires_grad=True)
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred_action, outcome)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
