from minesweeper.board import Board
from minesweeper.player import RandomPlayer
from minesweeper.play import play_minesweeper


""" def train_Qplayer(rounds, qplayer):
    assert isinstance(qplayer, QPlayer)
    print("training...")
    for r in range(rounds):
        qplayer.train(Board(4,4,4)) """

b = Board(9,9,10)
p = RandomPlayer()
#q = QPlayer(1)
#train_Qplayer(1000, q)


print("let's play")
w, s = play_minesweeper(p, b)
print(w)
print(s)
for l in b.minefield :
    print(l)