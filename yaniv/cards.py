# -*- coding: utf-8 -*-

from itertools import groupby
from operator import itemgetter
import numpy as np


JOKER_SUITE1 = 'a'
JOKER_SUITE2 = 'b'
JOKER_RANK = 'W'
JOKER_STREAK_VALUE = -1  # better than 0, because A has value 1

SUITE_CHAR_TO_SYMBOL = {'d': '♦', 'h': '♥', 'c': '♣', 's': '♠', JOKER_SUITE1: '☻', JOKER_SUITE2:'☺'}
SUITE_CHAR_TO_COLOR = {'d': 'red', 'h': 'red', 'c': 'black', 's': 'black', JOKER_SUITE1: 'green', JOKER_SUITE2: 'blue'}

RANK_TO_VALUE = {str(rank): rank for rank in range(2, 11)}
for combos in [('A', 1), ('J', 10), ('Q', 10), ('K', 10), (JOKER_RANK, 0)]:
    RANK_TO_VALUE[combos[0]] = combos[1]

RANK_TO_STREAK_VALUES = {str(rank): rank for rank in range(2, 11)}
for combos in [('A', 1), ('J', 11), ('Q', 12), ('K', 13), (JOKER_RANK, JOKER_STREAK_VALUE)]:
    RANK_TO_STREAK_VALUES[combos[0]] = combos[1]


# TODO: create test
def card_is_joker(card):
    return JOKER_RANK in card


def cards_to_number_jokers(cards):
    return sum(list(map(card_is_joker, cards)))


def card_to_color(card):
    return SUITE_CHAR_TO_COLOR[card_to_suite(card)]


def card_to_value(card):
    return RANK_TO_VALUE[card_to_rank(card)]


def cards_to_values(cards):
    return list(map(card_to_value, cards))


def cards_to_value_sum(cards):
    return np.sum(cards_to_values(cards))


def card_to_streak_value(card):
    return RANK_TO_STREAK_VALUES[card_to_rank(card)]


def rank_to_value(rank):
    return RANK_TO_VALUE[rank]


def card_to_pretty(card):
    return ''.join([SUITE_CHAR_TO_SYMBOL[char] if char in SUITE_CHAR_TO_SYMBOL.keys() else char for char in card])


def card_to_suite(card):
    '''Returns suite of card
    :param card: str
    :return: str. possible values 's' (Spades), 'd' (Diamonds), 'h' (Hearts), 'c' (Clubs) and JOKER_SUITE1, JOKER_SUITE2 (Jokers)
    '''

    return card[-1]


def card_to_rank(card):
    '''Returns the rank of the card in str

    Note that possible values are 'A', '2', '3', ... '10', 'J', 'Q', 'K' and JOKER_RANK1, JOKER_RANK2

    :param card: str.
    :return: str.
    '''
    return card[:-1]

def cards_same_rank(these_cards):
    return 1 == len(np.unique(list(map(card_to_rank, these_cards))))


def get_deck(play_jokers=True, shuffle=False, seed=None):
    '''Return the deck in dict type

    :param play_jokers: bool. True: game played with 2 jokers, False: without
    :return: list of cards
    '''

    suits = ['d', 'h', 'c', 's']   #  diamonds, hearts, clubs, spades
    ranks = ['A'] + list(map(str, range(2, 11))) + ['J', 'Q', 'K']  # Ace, 2-10, Jack, Queen, King

    deck = ["{}{}".format(rank, suit) for rank in ranks for suit in suits]

    if play_jokers:
        deck.insert(0, '{}{}'.format(JOKER_RANK, JOKER_SUITE1))
        deck.insert(1, '{}{}'.format(JOKER_RANK, JOKER_SUITE2))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(deck)

    return deck


def sort_cards(cards, return_streak_values=False, descending=False):
    streak_values = list(map(lambda x: card_to_streak_value(x), cards))
    cards = [card for _, card in sorted(zip(streak_values, cards), key=lambda pair: pair[0], reverse=descending)]

    if return_streak_values:
        return cards, sorted(streak_values)
    else:
        return cards


# TODO: might consider ordering of combos that have same sum based on number of cards
def sort_card_combos(card_combos, descending=True, return_sum_values=True):
    card_combo_sums = [cards_to_value_sum(these_cards) for these_cards in card_combos]
    card_combos_ordered = [combos for _, combos in sorted(zip(card_combo_sums, card_combos), key=lambda pair: pair[0], reverse=descending)]

    if return_sum_values:
        return card_combos_ordered, card_combo_sums
    return card_combos_ordered


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


def cards_to_consecutive_combinations(cards, verbose=0):
    cards, streak_values = sort_cards(cards, return_streak_values=True)

    if verbose:
        print(f'cards: {cards}\nstreak_values:{streak_values}')
    suites = list(map(card_to_suite, cards))

    consecutive_combinations = []

    suites_set, suites_set_counts = np.unique(suites, return_counts=True)

    for this_suite in suites_set[suites_set_counts >= 3]:

        this_suite_card_streakValues = [(card, streak_value) for card, suite, streak_value in zip(cards, suites, streak_values) if suite == this_suite]
        this_suite_cards, this_suite_streakValues = list(zip(*this_suite_card_streakValues))

        for k, g in groupby(enumerate(this_suite_streakValues), lambda x: x[0 ] -x[1]):
            grouped_values_one_group = list(map(itemgetter(1), g))
            # print('grouped_values_one_group', grouped_values_one_group)

            len_grouped_values_one_group = len(grouped_values_one_group)
            l_grouped_values = []
            if len_grouped_values_one_group >= 3:
                combo_sizes = range(3, len_grouped_values_one_group + 1)

                for combo_size in combo_sizes:
                    for idx_first in range(0, len_grouped_values_one_group - combo_size + 1):
                        start_ = grouped_values_one_group[idx_first]
                        stop_ = grouped_values_one_group[idx_first + combo_size - 1]
                        l_grouped_values.append(list(range(start_, stop_ + 1)))

                for grouped_values in l_grouped_values:
                    consecutive_combiation = [card for card, streak_value in
                                              zip(this_suite_cards, this_suite_streakValues) if
                                              streak_value in grouped_values]
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


def pile_top_accessible_cards(pile_top_cards):
    pile_top_cards_accessible = pile_top_cards.copy()
    if len(pile_top_cards_accessible) > 2:
        if not cards_same_rank(pile_top_cards_accessible):
            # only outer cards accessible in case of streak
            cards_sorted = sort_cards(pile_top_cards_accessible)
            pile_top_cards_accessible = [cards_sorted[0], cards_sorted[-1]]

    return pile_top_cards_accessible



def pile_cards_plus_player_collective_hypothetical_points(accessible_cards, player_cards):
    """Returns dictionary {card: max_possible_combo_points}
    card is yielded only if max_possible_combo_points comes from a card combination of at least 2 cards including
    one from accessible_cards.

    :param accessible_cards:
    :param player_cards:
    :return:
    """
    pile_card_to_player_max_collective_points = {}


    for card in accessible_cards:
        hypothetical_hand_cards = player_cards + [card]
        hypothetical_valid_combinations = cards_to_valid_throw_combinations(hypothetical_hand_cards)
        hypothetical_valid_combinations_with_card = [combination_ for combination_ in
                                                     hypothetical_valid_combinations
                                                     if (card in combination_) & (len(combination_) > 1)]

        hypothetical_combos_points = list(map(lambda cards: np.sum(cards_to_values(cards)), hypothetical_valid_combinations_with_card))

        if bool(hypothetical_combos_points):
            pile_card_to_player_max_collective_points[card] = max(hypothetical_combos_points)

    return pile_card_to_player_max_collective_points
