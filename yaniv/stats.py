from scipy.stats import hypergeom
from itertools import combinations, permutations
from scipy.special import comb
import numpy as np

from cards import cards_to_value_sum, cards_to_values, cards_to_number_jokers
#from game import YANIV_LIMIT
YANIV_LIMIT = 7

# TODO: build on this principle to make card_num_to_max_single_value in a dyanmic fashion, i.e, with knowldege of cards thrown out
def card_number_to_max_card_value_to_declare_yaniv(play_jokers=True):
    '''Returns a mapping between the number of cards in the hand to the max possible single card value f
    or successful Yaniv

    E.g, if no jokers are in play, and someone has 5 cards, in order to successfully declare Yaniv,
    the maximum value any single card may have is 3:
    3, ace, ace, ace, ace

    If there are jokers still in play, and someone has 5 cards, in order to in order to successfully declare Yaniv,
    the maximum value any single card may have is 5:
    5, joker, joker, ace, ace
    4 is aso an option:
    4, joker, joker, ace, 2 (or ace),
    as is 3:
    3, joker, joker, 2, 2 (or 3 and ace)

    but 5 is the ultimate max.

    This is useful to determine heuristics for probabilities.

    Here we assume:
     * a full deck (i.e, no knowledge of what might have been thrown out).
     * Yaniv (win) declare with 7 or less

    :return: dict. keys are int of number of cards in hand. values are int of maximum possible value single card in
    order to declare a successful Yaniv
    '''
    # Evaluating the most extreme values to obtain a Yaniv
    # No assumption about state of deck made (assuming a full deck)

    # making a function for future purposes
    card_num_to_max_single_value = {}

    card_num_to_max_single_value[1] = 7  # if player has 1 card in hand, the max value card for yaniv is 7: 7

    if play_jokers:
        # the first one means: if player has 5 cards in hand, the max value card for yaniv is 5: 5, j, j, ace, ace
        card_num_to_max_single_value[5] = 5  # 5, joker, joker, ace, ace
        card_num_to_max_single_value[4] = 6  # 6, joker, joker, ace
        card_num_to_max_single_value[3] = 7  # 7, joker, joker
        card_num_to_max_single_value[2] = 7  # 7, joker
    else:
        # the first one  means: if player has 5 cards in hand, the max value card for yaniv is 3: 3, ace, ace, ace, ace
        card_num_to_max_single_value[5] = 3
        card_num_to_max_single_value[4] = 4  # 4, ace, ace, ace
        card_num_to_max_single_value[3] = 5  # 5, ace, ace
        card_num_to_max_single_value[2] = 6  # 6, ace

    return card_num_to_max_single_value


def calculate_prob_all_cards_under_thresh(n_cards, smaller_than_thresh_bool, verbose=1):
    n = n_cards  # of n cards in player's hand
    k = n  # we want to verify that all k=n are lower than the thresh

    N = len(smaller_than_thresh_bool)  # from a pool of total N cards
    K = sum(smaller_than_thresh_bool)  # where K of them are known to be less than thresh

    prob_all_cards_under_thresh = hypergeom.pmf(k, N, K, n)

    if verbose >= 1:
        print("Of a total of N={} unknown cards K={} cards are below or equal to thresh".format(N, K))
        print("The probability that all k={} of n={} cards are below thresh is: %{:0.1f}".format(k, n,
                                                                                                 prob_all_cards_under_thresh * 100.))
    return prob_all_cards_under_thresh


def calculate_prob_ht_gt_hi(hand_points_i, card_values, n_cards): # was calculate_prob_above_yaniv_given_all_below_thresh

    permutations_counter = 0
    hj_gt_hi_counter = 0 # was other_above_yaniv_counter

    for permutation in permutations(card_values, n_cards):
        permutations_counter += 1
        hand_points_j = sum(permutation)
        if hand_points_j > hand_points_i:

            hj_gt_hi_counter += 1


    prob_hj_gt_hi = hj_gt_hi_counter * 1. / permutations_counter

    return prob_hj_gt_hi


# TODO: test (OR TAKE OUT IF NOT USED)
def is_smaller_or_equal_binary(value, thresh=None):
    return int(value <= thresh)


# TODO: test
def is_smaller_binary(value, thresh=None):
    return int(value < thresh)


# TODO: create test
def calculate_number_card_combinations(n_cards_total, n_cards_hand, repetition=False):
    return comb(n_cards_total, n_cards_hand, repetition=repetition)


# TODO: create test
def calculate_p_hj_gt_hi_accurate(cards, n_j, h_i, verbose=0):
    if len(cards) == 0:
        return 1

    hj_gt_hi_bool = []

    for cards_j in combinations(cards, n_j):
        bool_ = cards_to_value_sum(cards_j) > h_i
        if verbose > 1:
            print(cards_j, bool_)
        hj_gt_hi_bool.append(bool_)

    if len(hj_gt_hi_bool) == 0: # because len(cards)<n_j
        hj_gt_hi_bool.append( cards_to_value_sum(cards) > h_i   )


    successes = np.sum(hj_gt_hi_bool)
    total = len(hj_gt_hi_bool)

    #print(n_j, cards, hj_gt_hi_bool)
    #assert total > 0

    if verbose:
        print(f'total: {total}\nsuccesses: {successes}')

    return successes / total


def _calculate_thresh_nj(n_j, h_i, n_jokers=0, verbose=0):
    t_nj = h_i - n_j + 2 + n_jokers

    if verbose:
        print(f'n_j={n_j}, h_i={h_i}, n_jokers={n_jokers} yields\nt_nj={t_nj}')

    return t_nj

def _subset_cards_by_thresh(cards, thresh, verbose=0):
    below_tresh_binary = list(map(lambda v: is_smaller_binary(v, thresh), cards_to_values(cards)))
    cards_threshed = list(np.array(cards)[np.array(below_tresh_binary, dtype=bool)])

    if verbose:
        print(f'N: {len(cards)}\n{cards}\nn: {len(cards_threshed)}\n{cards_threshed}')

    return cards_threshed


def calculate_p_hj_gt_hi_conditioned_U(n_j, h_i, cards, play_jokers=True, verbose=0):
    n_jokers = 0
    if play_jokers:
        n_jokers = cards_to_number_jokers(cards)

    thresh = _calculate_thresh_nj(n_j, h_i, n_jokers=n_jokers, verbose=verbose)
    cards_threshed = _subset_cards_by_thresh(cards, thresh, verbose=verbose)

    p_hj_gt_hi_conditioned_U = calculate_p_hj_gt_hi_accurate(cards_threshed, n_j, h_i, verbose=verbose)

    if verbose:
        print(f'P(h_j>h_i|U)={p_hj_gt_hi_conditioned_U:0.3f}')

    return p_hj_gt_hi_conditioned_U


def calculate_p_U(cards, cards_threshed, n_j, verbose=0): #hypergeom_k_equals_n(N, K, n):
    N = len(cards)
    K = len(cards_threshed)

    p_U = comb(K, n_j, repetition=False) / comb(N, n_j, repetition=False)

    if verbose:
        print(f'N={N}, K={K}, n_j={n_j}, yields:\np(U)={p_U:0.3f}')

    return p_U


# TODO might have to be smarter about the h_i/yaniv_limit given `cards` ...
def calculate_p_hj_gt_hi_n_j_prior(n_j, cards, h_i=None, play_jokers=True, verbose=0):
    if h_i is None:
        h_i = YANIV_LIMIT

    n_jokers = 0
    if play_jokers:
        n_jokers = cards_to_number_jokers(cards)

    # --- code from calculate_p_hj_gt_hi_conditioned_U
    thresh = _calculate_thresh_nj(n_j, h_i, n_jokers=n_jokers, verbose=verbose)
    cards_threshed = _subset_cards_by_thresh(cards, thresh, verbose=verbose)

    p_hj_gt_hi_conditioned_U = calculate_p_hj_gt_hi_accurate(cards_threshed, n_j, h_i, verbose=verbose)

    if verbose:
        print(f'P(h_j>h_i|U)={p_hj_gt_hi_conditioned_U:0.3f}')
    # ---

    p_U = calculate_p_U(cards, cards_threshed, n_j, verbose=verbose)

    p_hj_gt_hi = p_hj_gt_hi_conditioned_U * p_U + 1 - p_U

    if verbose:
        print(f'P(h_j>h_i={h_i}|n_j, cards)={p_hj_gt_hi:0.3f}')

    return p_hj_gt_hi

