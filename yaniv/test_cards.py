from unittest import TestCase

from cards import (rank_to_value,
                  card_to_suite,
                  card_to_rank,
                   card_to_value,
                   cards_to_value_sum,
                  SUITE_CHAR_TO_SYMBOL,
                  JOKER_RANK, JOKER_SUITE1, JOKER_SUITE2
                  )

RANKS = ['A'] + map(str, range(2, 11)) + ['J', 'Q', 'K', JOKER_RANK]


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

