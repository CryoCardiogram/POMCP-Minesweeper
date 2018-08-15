from problems.minesweeper.board import Board
from problems.minesweeper.player import RandomPlayer, MCPlayer, QPlayer, train_Qplayer
from problems.minesweeper.play import play_minesweeper
import os
import sys
import csv
import traceback
import getopt


class Agent(object):
    def __init__(self, player, name, index=1):
        self.player = player
        self.name = name
        self.index = index

b = Board(9,9,10)
p = RandomPlayer()
monte_carlo = MCPlayer(2000000, 30.0) 
Q = QPlayer("toast", True)
#train_Qplayer(10, Q,2, 5, 3 )
#play_minesweeper(monte_carlo, b, True)
INF = 200000000
PERF = [(1,3,1), (2,5,3), (5,5,10), (5,5,15), (8,8,10), (9,9,10), (16,16,40)]
BSIZE = [(4,4,2), (4,4,3, (4,4,4)), (4,4,5), (4,4,6)]
ALL = PERF + BSIZE
Q_TRAINS = [ 5000, 10000, 50000, 100000]
MC_TIME = [0.5, 1.0, 1.5, 2.0]
AGENTS = {
    # random
    'RND':Agent(RandomPlayer(), 'RND'), 
    # Q-learning without symmetries          
    #'QNS_5K': Agent(QPlayer('QNS_5K', False), 'QNS_5K', 0),
    #'QNS_10K': Agent(QPlayer('QNS_10K', False), 'QNS_10K', 1),
    #'QNS_50K': Agent(QPlayer('QNS_50K', False), 'QNS_50K', 2),
    #'QNS_100K': Agent(QPlayer('QNS_100K', False), 'QNS_100K', 3),
    # Q-learning with symmetries
    #'QS_5K': Agent(QPlayer('QS_5K', False), 'QS_5K', 0),
    #'QS_10K': Agent(QPlayer('QS_10K', False), 'QS_10K', 1),
    'QS_50K': Agent(QPlayer('QS_50K', False), 'QS_50K', 2),
    #'QS_100K': Agent(QPlayer('QS_100K', False), 'QS_100K', 3),
    # Monte-Carlo without pref actions
    #'MCNP_05': Agent(MCPlayer(INF, 0.5 ), 'MCNP_05', 0),
    'MCNP_10': Agent(MCPlayer(INF, 1.0 ), 'MCNP_10', 1),
    #'MCNP_15': Agent(MCPlayer(INF, 1.5 ), 'MCNP_15', 2),
    #'MCNP_20': Agent(MCPlayer(INF, 2.0 ), 'MCNP_20', 3),
    # Monte-Carlo with pref actions
    #'MCP_05': Agent(MCPlayer(INF, 0.5 ), 'MCP_05', 0),
    'MCP_10': Agent(MCPlayer(INF, 1.0 ), 'MCP_10', 1),
    #'MCP_15': Agent(MCPlayer(INF, 1.5 ), 'MCP_15', 2),
    #'MCP_20': Agent(MCPlayer(INF, 2.0 ), 'MCP_20', 3)
}

def experiment(agents, iterations, boards):

    def filename(agent, b):
        return "data/{}_{}x{}m{}.csv".format(agent.name, b[0], b[1], b[2])
    # check if csv file exists and generate it otherwise
    # setup result dict
    res = dict()
    for agent in agents:
        for b in boards:
            res[agent.name] = {b: []}
            fname = filename(agent, b)
            if fname not in os.listdir("data"):
                with open(fname, 'x') as f:
                    cW = csv.writer(f)
                    cW.writerow['win', 'steps']
    
    errors = 0
    def main_loop():
        for i in range(iterations):
            for agent in agents:
                for b in boards:
                    try:
                        res[agent.name][b].append(play_minesweeper(agent.player, Board(b[0], b[1], b[2]), False))
                    except (AssertionError, IndexError):
                        errors += 1
                        with open('err.txt', 'a') as err:
                            err.write("iteration {}\n Agent {}".format(i, agent.name))
                            tb = sys.exc_info()[2]
                            traceback.print_tb(tb, file=err)        
            print("{} error(s)".format(errors))

        try:
            main_loop()
        except:
            print("an error as occured")

    for a, b_res_list in res.items():
        for b, res in b_res_list.items():
            with open(filename(a, b), 'a') as f:
                cW = csv.writer(f)
                cW.writerows(res)



if __name__=='__main__' and False:
    print("UC")
    
        