import pickle

MINE = '*'
UNCOV = "U"
NOTHING = " "
FMOVE = 'S'
ONE = "1"
TWO = "2"
THREE = "3"
FOUR = "4"
FIVE = "5"
SIX = "6"
SEVEN = "7"
EIGHT = "8"

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'w+b') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)