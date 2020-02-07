import numpy as np
import pandas as pd

from stats import calculate_p_hj_gt_hi_n_j_prior
from cards import cards_to_number_jokers, cards_to_value_sum


def _round_output_turn_keys_to_int(round_output):
    str_keys = ['start', 'winner', 'end']

    turn_keys = []
    for key in round_output.keys():
        if key not in str_keys:
            turn_keys.append(int(key))

    turn_keys = sorted(turn_keys)

    round_output_ = {'start': round_output['start']}

    for turn in turn_keys:
        round_output_[turn] = round_output[f'{turn}']

    if 'winner' in round_output.keys():
        round_output_['winner'] = round_output['winner']
    if 'end' in round_output.keys():
        round_output_['end'] = round_output['end']

    return round_output_


def _game_output_round_keys_to_int(game_output):

    round_keys = []
    for round in game_output.keys():
        round_keys.append(int(round))

        round_keys = sorted(round_keys)

    game_output_ = {}
    for round in round_keys:
        game_output_[round] = game_output[f'{round}']

    return game_output_


def game_output_keys_to_int(game_output):
    game_output = _game_output_round_keys_to_int(game_output)

    for round in game_output.keys():
        game_output[round] = _round_output_turn_keys_to_int(game_output[round])

    return game_output



def round_to_number_of_turns(round_output):
    return np.sum(list(map(lambda x: isinstance(x, int), list(round_output.keys()))))


def game_turns_per_round(game_output):
    turns_per_round = [round_to_number_of_turns(game_output[round_]) for round_ in game_output.keys()]

    return turns_per_round


def round_output_to_player_names(round_output):
    return [key.split('_cards')[0] for key in round_output['start'].keys() if 'cards' in key]


def round_output_to_yaniv_probabilities_two_player(round_output, pov_name=None, verbose=0):
    player_names = round_output_to_player_names(round_output)
    if pov_name is None:
        pov_name = round_output[1]['name']

    opponent_name = list(set(player_names) - set(pov_name))[0]

    n_turns = round_to_number_of_turns(round_output)

    cards_unknown = round_output['start']['deck_ordered'].copy()
    play_jokers = np.sum(cards_to_number_jokers(cards_unknown) > 0, dtype=bool)
    cards_unknown.remove(round_output[1]['pile_top_accessible'][0]) # pile top card

    n_j = len(round_output['start'][f'{opponent_name}_cards']) # no. of cards of opponent

    turn_to_yaniv_probability = {}
    for turn in range(1, n_turns + 1):
        turn_output = round_output[turn]

        if 'yaniv_call' not in turn_output.keys():
            if turn_output['name'] == pov_name:
                for card in turn_output[f'{pov_name}_cards']:
                    if card in cards_unknown:
                        cards_unknown.remove(card)

                h_i = cards_to_value_sum(turn_output[f'{pov_name}_cards'])

                if h_i <= 7:
                    turn_to_yaniv_probability[turn] = calculate_p_hj_gt_hi_n_j_prior(n_j, cards_unknown, h_i=h_i,
                                                                                     play_jokers=play_jokers,
                                                                                     verbose=verbose)

            # elif turn_output['pull_source'] == 'pile': !!! NEED TO ADD THIS INFORMATION !!!
            else:
                n_j = turn_output[f'{opponent_name}_ncards'] - len(turn_output['throws']) + 1
                # print(turn_output['name'], turn_output['pulls'], n_j)

            for card in turn_output['throws']:
                if card in cards_unknown:
                    cards_unknown.remove(card)

        else:
            if turn_output['name'] == pov_name:
                h_i = cards_to_value_sum(turn_output[f'{pov_name}_cards'])
                turn_to_yaniv_probability[turn] = calculate_p_hj_gt_hi_n_j_prior(n_j, cards_unknown, h_i=h_i,
                                                                                 play_jokers=play_jokers, verbose=verbose)

    return turn_to_yaniv_probability



def game_output_to_yaniv_probabilities(game_output, pov_name=None):
    if pov_name is None:
        pov_name = game_output[1][1]['name']

    game_yaniv_probabilities = {}

    for round_ in game_output.keys():

        round_output = game_output[round_].copy()
        probabilities = round_output_to_yaniv_probabilities_two_player(round_output, pov_name=pov_name)

        if len(probabilities) > 0:
            game_yaniv_probabilities[round_] = probabilities

    df_probabilities = pd.DataFrame(game_yaniv_probabilities)
    df_probabilities.index.name = 'turn'
    df_probabilities.columns.name = 'round'

    return df_probabilities


def round_output_to_turns_df(round_output):
    n_turns = round_to_number_of_turns(round_output)
    df = pd.DataFrame({turn: round_output[turn] for turn in np.arange(1, n_turns + 1)}).T

    return df