# -*- coding: utf-8 -*-

from itertools import groupby
from operator import itemgetter

SUITE_CHAR_TO_SYMBOL = {'d': '♦', 'h': '♥', 'c': '♣', 's': '♠'}
JOKER_SUITE = 'w'
JOKER_RANK = 'W'
JOKER_STREAK_VALUE = -1  # better than 0, because A has value 1


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
