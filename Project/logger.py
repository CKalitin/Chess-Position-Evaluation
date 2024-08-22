import matplotlib.pyplot as plt
from IPython import display
from datetime import datetime

class Logger():
    def __init__(self):
        self.losses = []
        self.start_time = datetime.now()
        plt.ion()
    
    def log_loss(self, loss):
        self.losses.append(loss)

    def display_charts(self, num_games, epsilon):
        display.clear_output(wait=True)
        #display.display(plt.gcf())
        plt.clf()
        g = num_games
        e = round(epsilon*1000)/1000
        t = str(datetime.now() - self.start_time).split('.')[0]
        
        plt.title(f"Number of Games: {g}  /  Epsilon: {e}  /  Time: {t}")
        plt.xlabel('Number of Training Batches')
        plt.ylabel('Loss')
        plt.plot(self.losses)
        #plt.ylim(ymin=0)
        plt.show(block=False)
        plt.pause(0.00001)