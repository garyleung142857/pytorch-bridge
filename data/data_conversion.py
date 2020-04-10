from torch.utils.data import Dataset
import numpy as np


RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
PLAYERS = ['N', 'E', 'S', 'W']
STRAIN = ['N', 'S', 'H', 'D', 'C']


def line2data(line, dclr, strain):
    '''
    read sol100000.txt
    is_nt: wheter there are trumps
    The first 13 vectors are trump suits (if any)
    Declarer are put at the 0th position, 1st LHO, 2nd partner, 3rd RHO
    '''
    raws, ddtbl = line.split(':')
    dist = [h.split('.') for h in raws.split(' ')]

    # cycle the declarer to the 0th position
    dist = dist[dclr:] + dist[0:dclr]

    is_nt = not strain

    # swap the trump suit (if any) to the front
    if not is_nt:
        for h in dist:
            h[0], h[strain - 1] = h[strain - 1], h[0]

    hot_encode = [[0, 0, 0, 0] for _ in range(52)]
    for suit in range(4):
        for rank in range(13):
            for player in range(4):
                if RANKS[rank] in dist[player][suit]:
                    # hot_encode[13 * suit + rank][player] = 1
                    hot_encode[13 * suit + rank] = player

    c = ddtbl[strain * 4 + dclr]
    d = {
        'A': 10,
        'B': 11,
        'C': 12,
        'D': 13
    }
    c = np.double(d.get(c, c))
    if dclr % 2:
        par_trick = c
    else:
        par_trick = 13 - c
    
    # return [1 * is_nt, 1 - 1 * is_nt] + hot_encode, par_trick
    return [1 * is_nt] + hot_encode, par_trick


class DealParDataset(Dataset):
    ''' DoubleDummy Dataset '''

    def __init__(self, is_train_set=False):
        filename = './data/try10000.txt' if is_train_set else './data/test10000.txt'
        self.rawtable = open(filename, 'r').readlines()
        self.len = len(self.rawtable) * 20  # 5 strains, 4 declarers
    
    def __getitem__(self, index):
        out = line2data(self.rawtable[int(index / 20)], index % 4, index % 5)
        # print(out)
        return out

    def __len__(self):
        return self.len


if __name__ == "__main__":
    with open('data/sol10.txt', 'r') as f:
        a = f.readlines()
        o = line2data(a[23], 3, 3)
        print(o)