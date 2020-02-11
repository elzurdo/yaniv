from configs import YANIV_LIMIT  #  (ASSAF_PENALTY, END_GAME_SCORE, MAX_ROUNDS, MAX_TURNS, YANIV_LIMIT)

import numpy as np


def _complete(strategy):
    strategy['yaniv_declare'] = 'not_always'
    strategy['prob_successful_yaniv_thresh'] = 0.2


def pile_conservative_vary(seed=None):
    strategy = {}

    min_ = np.max([0, YANIV_LIMIT - 5])
    max_ = YANIV_LIMIT - 3

    np.random.seed(seed)
    strategy['pile_pull'] = {"highest_card_value_to_pull": np.random.randint(min_, max_ + 1)}

    _complete(strategy)

    return strategy


def pile_conservative_constant():
    strategy = {}

    strategy['pile_pull'] = {"highest_card_value_to_pull": 3}

    _complete(strategy)

    return strategy


def pile_always():
    strategy = {}

    strategy['pile_pull'] = {"highest_card_value_to_pull": 10}

    _complete(strategy)

    return strategy
