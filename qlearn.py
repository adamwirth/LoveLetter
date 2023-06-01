from engine.GameEngine import GameEngine
from player.RandomAI import RandomAI
from player.QLearningAI import QLearningAI

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
    # Setup
    num_games = 25000000
    # num_games = 50
    QAI1 = QLearningAI()
    QAI2 = QLearningAI(.05)
    QAI3 = QLearningAI(.2)
    game_stats = np.empty((3, num_games), dtype=bool) # True for AI win, False for other

    RandomW = 0
    for i in tqdm(range(num_games)):
        # AI1's game
        engine = GameEngine()
        engine.addPlayer(QAI1)
        randomAI = RandomAI()
        engine.addPlayer(randomAI)
        winner = engine.runGame()
        game_stats[0, i] = (winner == QAI1)
        
        # AI1's game
        engine = GameEngine()
        engine.addPlayer(QAI2)
        randomAI = RandomAI()
        engine.addPlayer(randomAI)
        winner = engine.runGame()
        game_stats[1, i] = (winner == QAI2)

        # AI1's game
        engine = GameEngine()
        engine.addPlayer(QAI2)
        randomAI = RandomAI()
        engine.addPlayer(randomAI)
        winner = engine.runGame()
        game_stats[2, i] = (winner == QAI2)
        
    print("QAI1 won " + str(game_stats[0].sum()) + "/" + str(num_games) + " games against RandomAI")
    print("QAI2 won " + str(game_stats[1].sum()) + "/" + str(num_games) + " games against RandomAI")
    print("QAI3 won " + str(game_stats[2].sum()) + "/" + str(num_games) + " games against RandomAI")

    # Graph plot
    step_size = 1000000
    # step_size = 10
    num_steps = int(num_games/step_size)
    assert num_games % step_size == 0 # ensure divides evenly

    # Plot QA1
    x = np.linspace(1, num_games, num_steps)
    y1 = []
    y2 = []
    y3 = []
    for i in range(0, num_games, step_size):
        y1.append(game_stats[0, i:i+step_size].sum() / step_size)
        y2.append(game_stats[1, i:i+step_size].sum() / step_size)
        y3.append(game_stats[2, i:i+step_size].sum() / step_size)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.title('')
    plt.xlabel('#game')
    plt.ylabel('winrate')
    plt.show()
    plt.savefig("qlearn.png")

if __name__ == '__main__':
    main()