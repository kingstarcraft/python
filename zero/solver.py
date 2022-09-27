import random
import numpy as np


def ransac(model, critic, dataset, sample, probablity=0.999):
    size = len(dataset)
    if size < sample:
        return

    dataset = np.array(dataset)

    ids = list(range(len(dataset)))
    numerator = np.log(1 - probablity)
    denominator = 1 / size
    iters = numerator / np.log(1 - denominator ** sample + 1e-16)
    score = 1e10
    best = None
    iter = 0
    while True:
        if iter > iters:
            break
        random.shuffle(ids)
        idx = ids[:sample]
        solution = model(dataset[idx])
        if solution is not None:
            value = critic(solution, dataset)
            if value is not None:
                s, mask = value
                if s < score:
                    score = s
                    iters = numerator / np.log(1 - (np.sum(mask) * denominator)**sample + 1e-16)
                    best = solution, mask
        iter += 1
    return best
