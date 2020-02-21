import numpy as np
import pandas as pd

import game
from stats import calculate_p_hj_gt_hi_n_j_prior
from cards import cards_to_number_jokers, cards_to_value_sum, sort_cards


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


def update_pov_knowledge(player, opponent_name, turn_output, verbose=1):
    opponent_active = opponent_name == turn_output['name']

    player.other_players_n_cards[opponent_name] = len(turn_output[f'{opponent_name}_cards'])

    if not opponent_active:
        player.cards_in_hand = turn_output[f'{player.name}_cards']
        player.remove_cards_from_unknown(player.cards_in_hand)

    if 'throws' in turn_output.keys():
        cards_thrown_to_pile = turn_output['throws']
        if opponent_active:
            player.remove_cards_from_unknown(cards_thrown_to_pile)
            player.knowledgewise_drop_cards_from_player(opponent_name, cards_thrown_to_pile)
        else:
            player.cards_in_hand = list( set(player.cards_in_hand ) - set(cards_thrown_to_pile))


    cards_top_of_pile = turn_output['pile_top']
    player.remove_cards_from_unknown(cards_top_of_pile)

    if 'yaniv_call' in turn_output.keys():
        player.cards_top_of_pile = cards_top_of_pile
        return 'yaniv'

    card_pile_pulled = None
    if 'pile' == turn_output['pull_source']:
        card_pile_pulled = turn_output['pulls']

    cards_top_of_pile_left = list(set(cards_top_of_pile) - set([card_pile_pulled]))

    if cards_top_of_pile_left:
        player.add_cards_to_out_of_game(cards_top_of_pile_left)

    if opponent_active:
        player.knowledgewise_drop_cards_from_player(opponent_name, cards_top_of_pile)

        if card_pile_pulled:
            player.knowledgewise_assign_card_to_player(opponent_name, card_pile_pulled)

    if verbose:
        print(len(player.unknown_cards), len(player.cards_out_of_game), player.other_players_n_cards[opponent_name],
              len(player.other_players_known_cards[opponent_name]))


def player_to_df_knowledge(player, opponent_name, round_output):
    cards_all = sort_cards(round_output['start']['deck_ordered'].copy())

    df_pov = pd.DataFrame(0, columns=cards_all,
                          index=['cards_in_hand', 'cards_top_of_pile', 'cards_out_of_game', 'opponent_cards',
                                 'unknown_cards']).T
    df_pov.loc[player.cards_in_hand, 'cards_in_hand'] = 1
    df_pov.loc[player.cards_top_of_pile, 'cards_top_of_pile'] = 1
    df_pov.loc[player.cards_out_of_game, 'cards_out_of_game'] = 1

    df_pov.loc[player.unknown_cards, 'unknown_cards'] = 1.
    df_pov.loc[player.other_players_known_cards[opponent_name], 'opponent_cards'] = 1

    # verifying that all cards are accounted for
    #print(df_pov.sum(axis=1).sum(), len(cards_all))
    assert df_pov.sum(axis=1).sum() == len(cards_all)

    n_opponent_unkown_cards = player.other_players_n_cards[opponent_name] - len(
        player.other_players_known_cards[opponent_name])
    df_pov['opponent_cards'] = 0
    df_pov.loc[player.other_players_known_cards[opponent_name], 'opponent_cards'] = 1
    df_pov.loc[player.unknown_cards, 'opponent_cards'] = n_opponent_unkown_cards / len(player.unknown_cards)

    # all fractions accounted for
    assert np.abs(df_pov['opponent_cards'].sum() - player.other_players_n_cards[opponent_name]) < 1.e-5

    del df_pov['unknown_cards']

    return df_pov

# TODO: does not work in all cases. See asserts for edge cases ...
def round_input_to_pov_knowledge(pov_name, opponent_name, round_output):
    """

    :param pov_name:
    :param opponent_name:
    :param round_output:
    :return:

    Example use:
    player_names = ['Albert','Roland'] #, 'Amos','Claude']
    end_game_score = 200

    do_stats = False
    play_jokers = True
    seed = 1

    verbose = 1
    game_ = game.Game(player_names, seed=seed, verbose=verbose, end_game_score=end_game_score,
                      do_stats=do_stats, play_jokers=play_jokers)
    game_output = game_.play()


    round_number = 1
    round_output = game_output[round_number].copy()
    pov_name = 'Albert'
    opponent_name = 'Roland'
    round_input_to_pov_knowledge(pov_name, opponent_name, round_output)
    """
    turn_number_max = round_to_number_of_turns(round_output)
    turn_numbers = range(1, turn_number_max + 1)

    player = game.Player(pov_name)
    player.cards_in_hand = round_output['start'][f'{pov_name}_cards']
    player.other_players_known_cards = {}
    player.other_players_known_cards[opponent_name] = []
    player.other_players_n_cards = {}
    player.add_cards_to_unknown(list(set(round_output['start']['deck_ordered']) - set(player.cards_in_hand)))

    verbose = 0
    for turn_number in turn_numbers:
        turn_output = round_output[turn_number]

        if verbose: print(turn_number)
        update_pov_knowledge(player, opponent_name, turn_output, verbose=verbose)
        if verbose: print('-' * 30)

    return player_to_df_knowledge(player, opponent_name, round_output)






