import itertools
import random
import time

import numpy as np
from scipy.spatial.distance import hamming

from HelperFunctions.Extras.ProgressbarFunctions import progressBarWithTime


def buildPermutation(amount_of_top_permuts, sort=False):
    """creates a permutation list of permutations with the highest hamming distance between two permutations

    :param amount_of_top_permuts: determines how many top permutations are choosen
    :param sort: boolean value to determine if the permutations should be sorted
    """
    # Build list of all possible permutations
    permuts_list = list(itertools.permutations(range(9)))
    permuts_array = np.array(permuts_list)
    permut_size = len(permuts_list)
    permut_track = []
    starting_time = time.time()

    # Take top x permutations which have max average hamming distance
    top_permuts_list = []

    while True:
        x = random.randint(1, permut_size - 1)
        y = random.randint(1, permut_size - 1)

        permut_1 = permuts_array[x]
        permut_2 = permuts_array[y]
        hd = hamming(permut_1, permut_2)

        if hd > 0.9 and (x not in permut_track) and (y not in permut_track):
            permut_track.append(x)
            permut_track.append(y)
            top_permuts_list.append(permut_1)
            top_permuts_list.append(permut_2)

            if len(top_permuts_list) == amount_of_top_permuts:
                break

        progressBarWithTime(len(top_permuts_list), amount_of_top_permuts, starting_time)

    # Build the array for selected permutation indices above
    top_permuts_array = np.array(top_permuts_list)
    np.save(f'top_{amount_of_top_permuts}_permuts.npy', top_permuts_array)
    if (sort == True):
        sortPermutation(amount_of_top_permuts)


def sortPermutation(amount_of_top_permuts):
    """sorts the permutations and overwrites the file
    opens permutations file by using the parameter amount_of_top_permuts

    :param amount_of_top_permuts: determines how many top permutations are choosen
    """
    top_permuts_array = np.load(f'top_{amount_of_top_permuts}_permuts.npy')
    print(top_permuts_array)
    print("####################\n\n")
    top_permuts_array = top_permuts_array[np.lexsort(np.fliplr(top_permuts_array).T)]
    print(top_permuts_array)
    np.save(f'top_{amount_of_top_permuts}_permuts.npy', top_permuts_array)