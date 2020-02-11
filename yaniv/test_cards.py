# pytest

from unittest import TestCase

from cards import (rank_to_value,
                   card_to_suite,
                   card_to_rank,
                   card_to_value,
                   cards_to_value_sum,
                   cards_to_consecutive_combinations,
                   pile_top_accessible_cards,
                   sort_cards,
                   SUITE_CHAR_TO_SYMBOL,
                   JOKER_RANK, JOKER_SUITE1, JOKER_SUITE2
                   )

RANKS = ['A'] + list(map(str, range(2, 11))) + ['J', 'Q', 'K', JOKER_RANK]


class TestRankToValue(TestCase):
    def test_ace(self):
        self.assertEqual(rank_to_value('A'), 1)

    def test_vals(self):
        for rank in map(str, range(2, 11)):
            self.assertEqual(rank_to_value(rank), int(rank))

    def test_royals(self):
        for rank in ['J', 'Q', 'K']:
            self.assertEqual(rank_to_value(rank), 10)

    def test_joker(self):
        self.assertEqual(rank_to_value('W'), 0)


class TestCardToSuite(TestCase):
    def test_suites(self):
        for suite in SUITE_CHAR_TO_SYMBOL.keys():
            for rank in RANKS:
                print(rank)
                card = '{}{}'.format(rank, suite)
                self.assertEqual(card_to_suite(card), suite)


class TestCardToRank(TestCase):
    def test(self):

        for suite in SUITE_CHAR_TO_SYMBOL.keys():
            for rank in RANKS:
                card = '{}{}'.format(rank, suite)
                self.assertEqual(card_to_rank(card), rank)


class TestCardToValue(TestCase):
    def test(self):

        for suite in SUITE_CHAR_TO_SYMBOL.keys():
            for rank in RANKS:
                card = '{}{}'.format(rank, suite)
                self.assertEqual(card_to_value(card), rank_to_value(rank))


class TestCardsToValueSum(TestCase):
    def test(self):
        self.assertEqual(cards_to_value_sum(['5h', '4s']), 9)

    def test2(self):
        self.assertEqual(cards_to_value_sum(['Jh', '4s']), 14)

    def test3(self):
        self.assertEqual(cards_to_value_sum(['Jh', 'Qs', 'Kc', 'Ad']), 31)

    def test4(self):
        self.assertEqual(cards_to_value_sum(['{}{}'.format(JOKER_RANK, JOKER_SUITE1), '4s']), 4)


class TestCardstoConsecutiveCombinations(TestCase):
    def test(self):
        player_cards = ['As', '2s', '3s']
        result = cards_to_consecutive_combinations(player_cards)
        self.assertIn(player_cards, result)

    def test2(self):
        player_cards = ['As', 'Ks', '2s', '3s']
        result = cards_to_consecutive_combinations(player_cards)
        self.assertIn(['As', '2s', '3s'], result)

    def test3(self):
        player_cards = player_cards = ['8s', '5s', '6s', '7s', '5d']
        result = cards_to_consecutive_combinations(player_cards)
        ligit_combos = [['5s', '6s', '7s'], ['6s', '7s', '8s'], ['5s', '6s', '7s', '8s']]

        for combo in ligit_combos:
            self.assertIn(combo, result)

    def test4(self):
        player_cards = ['10s', 'Js', '9s', 'Qs', 'Ks']
        result = cards_to_consecutive_combinations(player_cards)
        ligit_combos = [['9s', '10s', 'Js'],
                        ['10s', 'Js', 'Qs'],
                        ['Js', 'Qs', 'Ks'],
                        ['9s', '10s', 'Js', 'Qs'],
                        ['10s', 'Js', 'Qs', 'Ks'],
                        ['9s', '10s', 'Js', 'Qs', 'Ks']]

        for combo in ligit_combos:
            self.assertIn(combo, result)


class TestPileTopAccessibleCards(TestCase):
    def test(self):
        pile_cards = ['As', '2s', '3s']
        result = pile_top_accessible_cards(pile_cards)

        self.assertEqual(['As', '3s'], result)

    def test2(self):
        pile_cards = ['As', 'Ad', 'Ah']
        result = pile_top_accessible_cards(pile_cards)

        self.assertEqual(pile_cards, result)

    def test3(self):
        pile_cards = ['9s', '10s', 'Js', 'Qs']
        result = pile_top_accessible_cards(pile_cards)

        self.assertEqual(['9s', 'Qs'], result)


class TesetSortCards(TestCase):
    def test(self):
        the_cards = ['5d', '3h', 'Kh', 'As', 'Qh']

        result = sort_cards(the_cards, descending=True)
        self.assertEqual(['Kh', 'Qh', '5d', '3h', 'As'], result)

        result = sort_cards(the_cards, descending=False)
        self.assertEqual(['As', '3h', '5d', 'Qh', 'Kh'], result)

