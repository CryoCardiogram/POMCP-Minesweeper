from board import Board
from player import RandomPlayer, QPlayer
from play import play_game


def train_Qplayer(rounds, qplayer):
    assert isinstance(qplayer, QPlayer)
    print("training...")
    for r in range(rounds):
        qplayer.train(Board(4,4,4))
        #if r % 50 == 0:
            #print("...")

b = Board(9,9,10)
p = RandomPlayer()
#q = QPlayer(1)
#train_Qplayer(1000, q)


print("let's play")
w, s = play_game(p, b)
print(w)
print(s)
for l in b.reveal :
    print(l)