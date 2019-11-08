from unittest import TestCase

from game import rank_to_value

class TestRankToValue(TestCase):
    def test_ace(self):
        self.assertEqual(rank_to_value('A'), 1)

    def test_vals(self):
        for rank in range(2,11):
            self.assertEqual(rank_to_value(rank), rank)

    def test_royals(self):
        for rank in ['J', 'Q', 'K']:
            self.assertEqual(rank_to_value(rank), 10)

    def test_joker(self):
        self.assertEqual(rank_to_value('W'), 0)
