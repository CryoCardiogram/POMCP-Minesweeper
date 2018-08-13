from problems.minesweeper.board import Board
from problems.minesweeper.player import RandomPlayer, MCPlayer
from problems.minesweeper.play import play_minesweeper
import os
import sys
import csv
import traceback


""" def train_Qplayer(rounds, qplayer):
    assert isinstance(qplayer, QPlayer)
    print("training...")
    for r in range(rounds):
        qplayer.train(Board(4,4,4)) """

b = Board(4,4,3)
p = RandomPlayer()
monte_carlo = MCPlayer(100000, 30.0) 
play_minesweeper(monte_carlo, b, True)


if __name__=='__main__' and False:

    with open('random.csv', 'w+') as o:
        cW = csv.writer(o)
        # headers
        cW.writerow(['win', 'steps'])
    with open('mc.csv', 'w+') as o:
        cW = csv.writer(o)
        # headers
        cW.writerow(['win', 'steps'])

    rr = []
    rmc = []
    errors = 0
    for i in range(5):
        try:
            rmc.append(play_minesweeper(monte_carlo, Board(4,4,5)))
        except:
            errors += 1
            with open('err.txt', 'a') as err:
                err.write("iteration {}\n".format(i))
                tb = sys.exc_info()[2]
                traceback.print_tb(tb, file=err)
        try:
            rr.append(play_minesweeper(p, Board(4,4,5)))
        except:
            errors += 1
            with open('err.txt', 'a') as err:
                err.write("iteration {}: {}\n".format(i, sys.exc_info()))
    print("{} error(s)".format(errors))

    rnd_fd = open("random.csv", 'a')
    mc_fd = open("mc.csv", 'a')
    rnd_w = csv.writer(rnd_fd)
    mc_w = csv.writer(mc_fd)

    rnd_w.writerows(rr)
    rnd_fd.close()
    mc_w.writerows(rmc)
    mc_fd.close()

    
        