import torch
import torch.nn as nn
import game_loader as gl
import models
import yaml
from collections import deque
import random

class Agent:
    def __init__(self, model, trainer, hyperparameters):
        self.model = model
        self.trainer = trainer
        self.hyperparameters = hyperparameters
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])
        self.memory = deque([], maxlen=self.hyperparameters["training_memory_size"]) # Size in number of games
        self.game_loader = gl.GameLoader()
    
    def train(self):
        print('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Model Parameters: { sum(p.numel() for p in self.model.parameters()) }") # numel returns the num elements in the given tensor
        
        num_games = 0
        num_iters = 0 # Games that have been searched (some discarded due to min elo), this prevents overlap vs. only using num_games which would reuse some games
        epsilon = self.hyperparameters["epsilon_init"]
        training = True
        while training:
            training_games, iters = self.game_loader.load_games(num_iters, self.hyperparameters["training_batch_size"], self.hyperparameters["min_avg_elo"])
            num_games += self.hyperparameters["training_batch_size"]
            num_iters += iters
            
            for i in range(self.hyperparameters["training_batch_size"]):
                fen_as_tensor = self.fen_to_tensor(self.game_loader.moves_to_fen(training_games['moves'][i], self.hyperparameters["training_move"]))
                model_output, type = self.get_epsilon_model_output(epsilon, fen_as_tensor)
                reward = self.get_reward(training_games['result'][i], model_output)
                self.memory.append((fen_as_tensor, model_output, reward))
            
            print(f"Reward: {reward} {type} {num_games} {epsilon} {model_output}")
            state, model_output, reward = zip(*self.memory)
            self.trainer.train_step(state, model_output, reward)
            epsilon = max(epsilon * self.hyperparameters["epsilon_decay"], self.hyperparameters["epsilon_min"])
            
            #if num_games > 0: return
    
    def get_epsilon_model_output(self, epsilon, model_input):
        if random.uniform(0, 1) < epsilon: return random.uniform(-1.5, 1.5), "r"
        else: return self.model(model_input)[0], "m" # Tensor to float
    
    def get_reward(self, game_result, model_output):
        if game_result not in { "1-0": 1, "0-1": -1, "1/2-1/2": 0 }: return 0
        translated_game_result = { "1-0": 1, "0-1": -1, "1/2-1/2": 0 }[game_result]
        return 1 if abs(translated_game_result - model_output) < 0.5 else -1
    
    def fen_to_tensor(self, fen):
        piece_to_value = { 'K': 1, 'Q': 0.8, 'R': 0.5, 'B': 0.4, 'N': 0.3, 'P': 0.1, 'k': -1, 'q': -0.8, 'r': -0.5, 'b': -0.4, 'n': -0.3, 'p': -0.1 }
        tensor = torch.zeros(1, 8, 8)
        pos = 0
        for character in fen:
            if pos >= 64: break
            if character.isdigit(): pos += int(character)
            elif character != "/":
                tensor[0][pos // 8][pos % 8] = piece_to_value[character]
                pos += 1
        return tensor

with open("Project/hyperparameters.yaml", "r") as file: hyperparameters = yaml.safe_load(file)['dqn-1']
model = models.DQN()
trainer = models.QTrainer(model, hyperparameters["learning_rate"], hyperparameters["discount_factor"])
agent = Agent(model, trainer, hyperparameters)
agent.train()