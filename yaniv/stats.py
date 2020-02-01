from scipy.stats import hypergeom
from itertools import permutations

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
