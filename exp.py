from problems.minesweeper.board import Board
from problems.minesweeper.player import RandomPlayer, MCPlayer
from problems.minesweeper.play import play_minesweeper


""" def train_Qplayer(rounds, qplayer):
    assert isinstance(qplayer, QPlayer)
    print("training...")
    for r in range(rounds):
        qplayer.train(Board(4,4,4)) """

b = Board(3,2,2)
p = RandomPlayer()
monte_carlo = MCPlayer(10, 0.5)
#q = QPlayer(1)
#train_Qplayer(1000, q)


print("let's play")
w, s = play_minesweeper(monte_carlo, b)
print(w)
print(s)
