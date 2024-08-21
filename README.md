# Chess-Position-Evaluation
Reinforcement Learning Neural Net Project

Dataset (2024, July): https://database.lichess.org/  
Adjusted to remove time control data, shrunk from 200GB -> 70GB

Goal is to predict which player will win a game of chess given the 10th move (20 half-moves, 2 player "half-moves" = 1 full move).  
1 = White, -1 = Black, 0 = Draw
Reward is the negative squared difference between the predicted outcome and the actual outcome.