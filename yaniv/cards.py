# -*- coding: utf-8 -*-

from itertools import groupby
from operator import itemgetter
import numpy as np

SUITE_CHAR_TO_SYMBOL = {'d': '♦', 'h': '♥', 'c': '♣', 's': '♠'}
JOKER_SUITE = 'w'
JOKER_RANK = 'W'
JOKER_STREAK_VALUE = -1  # better than 0, because A has value 1
RANK_TO_STREAK_VALUES = {str(rank): rank for rank in range(2, 11)}
for combos in [('A', 1), ('J', 11), ('Q', 12), ('K', 13), (JOKER_RANK, JOKER_STREAK_VALUE)]:
    RANK_TO_STREAK_VALUES[combos[0]] = combos[1]




# TODO: check if redundant with card_to_score
def rank_to_value(rank):
    if rank == 'A':
        return 1
    elif rank in ['J', 'Q', 'K']:
        return 10
    elif rank == 'W':
        return 0
    else:
        return int(rank)


def card_to_pretty(card):
    return ''.join([SUITE_CHAR_TO_SYMBOL[char] if char in SUITE_CHAR_TO_SYMBOL.keys() else char for char in card])


def card_to_suite(card):
    '''Returns suite of card
    :param card: str
    :return: str. possible values 's' (Spades), 'd' (Diamonds), 'h' (Hearts), 'c' (Clubs) and 'w' (Jokers)
    '''
    if card[-1] in [1, 2]:
        return JOKER_SUITE

    return card[-1]


def card_to_rank(card):
    '''Returns the rank of the card in str

    Note that possible values are 'A', '2', '3', ... '10', 'J', 'Q', 'K' and JOKER_RANK

    :param card: str.
    :return: str.
    '''
    return card[:-1]


# TODO rename card_to_streak_value so not to confuse with function by same name
def define_deck(play_jokers=True):
    '''Return the deck in dict type

    :param play_jokers: bool. True: game played with 2 jokers, False: without
    :return: dict. keys are the sting of the card and the values are the card points
    '''
    suits = ['d', 'h', 'c', 's']   #  ['♦', '♥', '♣', '♠']  # diamonds, hearts, clubs, spades
    ranks = ['A'] + list(map(str, range(2, 11))) + ['J', 'Q', 'K']  # Ace, 2-10, Jack, Queen, King

    values = list(range(1, 11)) + [10, 10, 10]  # notice! J, Q, K all give 10 points
    card_to_value = {"{}{}".format(rank, suit):
                         values[irank] for irank, rank in enumerate(ranks) for suit in suits}

    streak_values = list(range(1, 11)) + [11, 12, 13]  # notice! J, Q, K are valued at 11, 12, 13, respectively
    card_to_streak_value = {"{}{}".format(rank, suit):
                                streak_values[irank] for irank, rank in enumerate(ranks) for suit in suits}

    if play_jokers:
        card_to_value['{}1'.format(JOKER_RANK)] = 0
        card_to_value['{}2'.format(JOKER_RANK)] = 0

        card_to_streak_value['{}1'.format(JOKER_RANK)] = JOKER_STREAK_VALUE
        card_to_streak_value['{}2'.format(JOKER_RANK)] = JOKER_STREAK_VALUE

    return card_to_value, card_to_streak_value


def card_to_streak_value(card):
    return RANK_TO_STREAK_VALUES[card_to_rank(card)]


def sort_cards(cards, return_streak_values=False):
    streak_values = list(map(lambda x: card_to_streak_value(x), cards))
    cards = [card for _, card in sorted(zip(streak_values, cards), key=lambda pair: pair[0])]

    if return_streak_values:
        return cards, sorted(streak_values)
    else:
        return cards


def cards_to_same_rank_combinations(cards):
    same_rank_combinations = []
    ranks = list(map(card_to_rank, cards))

    ranks_set = sorted(set(ranks))  # annoying but not worth effort: puts '10' before '2'

    # we do not care about duplicates of Jokers
    if JOKER_RANK in ranks_set:
        ranks_set.remove(JOKER_RANK)

    for chosen_rank in ranks_set:
        chosen_rank_cards = [card for card, rank in zip(cards, ranks) if rank == chosen_rank]
        if len(chosen_rank_cards) > 1:
            same_rank_combinations.append(chosen_rank_cards)

    return same_rank_combinations


def cards_to_consecutive_combinations(cards):
    cards, streak_values = sort_cards(cards, return_streak_values=True)
    suites = list(map(card_to_suite, cards))

    consecutive_combinations = []

    suites_set, suites_set_counts = np.unique(suites, return_counts=True)

    for this_suite in suites_set[suites_set_counts >= 3]:

        this_suite_card_streakValues = [(card, streak_value) for card, suite, streak_value in zip(cards, suites, streak_values) if suite == this_suite]
        this_suite_cards, this_suite_streakValues = list(zip(*this_suite_card_streakValues))
        # print(this_suite, this_suite_cards, this_suite_streakValues)

        for k, g in groupby(enumerate(this_suite_streakValues), lambda x: x[0 ] -x[1]):
            grouped_values = list(map(itemgetter(1), g))

        if len(grouped_values) >= 3:
            consecutive_combiation = [card for card, streak_value in zip(this_suite_cards, this_suite_streakValues) if streak_value in grouped_values]
            consecutive_combinations.append(consecutive_combiation)

    return consecutive_combinations


def cards_to_valid_throw_combinations(cards):
    # singles
    valid_combinations = [[combination] for combination in cards]

    # same ranks (no point in using joker, though might consider for possible weird ML solutions?)
    same_rank_combinations = cards_to_same_rank_combinations(cards)
    for combination in same_rank_combinations:
        valid_combinations.append(combination)

    # consecutive combinations
    consecutive_combinations = cards_to_consecutive_combinations(cards)
    for combination in consecutive_combinations:
        valid_combinations.append(combination)

    return valid_combinations