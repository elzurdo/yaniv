# py.test

from unittest import TestCase

from stats import (card_number_to_max_card_value_to_declare_yaniv,
                   calculate_prob_all_cards_under_thresh,
                   calculate_prob_ht_gt_hi
                   )


class TestCardNumberToMaxCardValueToDeclareYaniv(TestCase):
    def test(self):
        n_cards = 2
        jokers = True
        n_cards_to_max_val = card_number_to_max_card_value_to_declare_yaniv(play_jokers=jokers)
        self.assertEqual(n_cards_to_max_val[n_cards], 7)

class TestCalculateProbAllCardsUnderThresh(TestCase):
    def test(self):
        n_smaller = 5
        n_larger = n_smaller
        smaller_than_thresh_bool = [1] * n_smaller + [0] * n_larger

        l_n_cards = [1, 2, 3]
        l_expected_results = [0.5, 2./9, 1./12]

        for n_cards, expected_result in zip(l_n_cards, l_expected_results):
            result = calculate_prob_all_cards_under_thresh(n_cards, smaller_than_thresh_bool)
            self.assertAlmostEqual(result, expected_result, places=7)


class TestCalculateProbHtGtHi(TestCase):
    def test(self):
        hand_points_i = 7
        n_cards_j = 1

        possible_card_values = [0, 8]

        result = calculate_prob_ht_gt_hi(hand_points_i, possible_card_values, n_cards_j)

        self.assertEqual(result, 0.5)