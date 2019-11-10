from unittest import TestCase

from cards import (rank_to_value,
                  card_to_suite,
                  card_to_rank,
                  SUITE_CHAR_TO_SYMBOL,
                  JOKER_SUITE, JOKER_RANK
                  )

RANKS = ['A'] + map(str, range(2, 11)) + ['J', 'Q', 'K', JOKER_RANK]


class TestRankToValue(TestCase):
    def test_ace(self):
        self.assertEqual(rank_to_value('A'), 1)

    def test_vals(self):
        for rank in range(2, 11):
            self.assertEqual(rank_to_value(rank), rank)

    def test_royals(self):
        for rank in ['J', 'Q', 'K']:
            self.assertEqual(rank_to_value(rank), 10)

    def test_joker(self):
        self.assertEqual(rank_to_value('W'), 0)


class TestCardToSuite(TestCase):
    def test_suites(self):
        for suite in SUITE_CHAR_TO_SYMBOL.keys():
            for rank in RANKS:
                card = '{}{}'.format(rank, suite)
                self.assertEqual(card_to_suite(card), suite)

    def test_jokers(self):
        for val in [1, 2]:
            card = '{}{}'.format(val, JOKER_SUITE)
            self.assertEqual(card_to_suite(card), JOKER_SUITE)


class TestCardToRank(TestCase):
    def test(self):

        for suite in SUITE_CHAR_TO_SYMBOL.keys():
            for rank in RANKS:
                card = '{}{}'.format(rank, suite)
                self.assertEqual(card_to_rank(card), rank)
