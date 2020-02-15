import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import game
from cards import (pile_top_accessible_cards,
                   cards_to_value_sum,
                   sort_cards,
                   cards_to_valid_throw_combinations,
                   sort_card_combos, card_to_value
                   )
from configs import YANIV_LIMIT, ASSAF_PENALTY


REWARD_FACTOR_YANIV_INCORRECT_PENALTY = -51
REWARD_FACTOR_TURN_POINT_DIFFERENCE = 0.01
REWARD_FACTOR_TURN_N_CARDS = 0.1
REWARD_NO_WINNER = -ASSAF_PENALTY


# TODO: decide if to leave here or put in cards
def cards_to_best_combination(cards):
    # cards_to_best_combination(['7s', '5s', '6s','5d', '5h'])
    valid_combinations = cards_to_valid_throw_combinations(cards)
    sorted_combinations, sorted_combinations_sums = sort_card_combos(valid_combinations, descending=True,
                                                                     return_sum_values=True)
    cards_to_throw = sorted_combinations[0]

    return cards_to_throw

def _name_to_str_cards(name):
    return f'{name}_cards'


def _name_to_opponent_name_two_players(name, players):
    return list(set(players.keys()) - set([name]))[0]


def new_round(env, name='Albert', seed=None):
    round_ = game.Round(env.players, env.deck, assaf_penalty=env.assaf_penalty,
                   card_num_to_max_value=None, verbose=env.verbose, seed=seed,
                   round_output={}, do_stats=env.do_stats, play_jokers=env.play_jokers)

    # round starts with a full deck
    round_.round_deck = list(round_.deck)

    np.random.seed(round_.seed)
    np.random.shuffle(round_.round_deck)

    round_.distribute_cards()

    round_.players_ordered = round_.get_player_order()

    # each player accounts for other players hands. In beginning no knowledge
    for name_i, player_i in round_.players_ordered.items():
        player_i.other_players_known_cards = {}
        for name_j, player_j in round_.players_ordered.items():
            if name_i != name_j:
                player_i.other_players_known_cards[name_j] = []

    env.round_ = round_

    n_deck = len(round_.round_output['start']['deck_ordered'])
    top_accessible_cards = round_.pile_top_cards
    cards_in_hand = round_.round_output['start'][_name_to_str_cards(name)]

    opponent_name = _name_to_opponent_name_two_players(name, env.players)
    n_cards_opponent = len(round_.round_output['start'][_name_to_str_cards(opponent_name)])

    return name, (n_deck, top_accessible_cards, cards_in_hand, n_cards_opponent)


def next_starter_name(env, pervious_starter=None, previous_winner=None, seed=None, method='alternate'):
    assert method in ['winner', 'alternate', 'random']

    if 'winner' == method:
        return previous_winner
    elif 'alternate' == method:
        return _name_to_opponent_name_two_players(pervious_starter, env.players)
    elif 'random' == method:
        np.random.seed(seed)
        return np.random.choice(list(env.players.keys()))

def step(env, name, action):
    info = {'yaniv_declare_correct': None}

    declare_yaniv = action[0]
    cards_to_throw = action[1]
    pick_from_deck = action[2]

    cards_in_hand_before = env.players[name].cards_in_hand
    n_cards_in_hand_before = len(cards_in_hand_before)
    hand_sum_before = cards_to_value_sum(cards_in_hand_before)

    reward_yaniv, reward_throw_pick = 0, 0
    done = False
    if declare_yaniv:
        done = True
        if hand_sum_before <= YANIV_LIMIT:
            env.round_.round_summary(name)
            info['yaniv_declare_correct'] = True
        else:
            reward_yaniv = REWARD_FACTOR_YANIV_INCORRECT_PENALTY
            info['yaniv_declare_correct'] = False

        n_cards_in_hand_after = None
    else:
        env.round_.throw_cards_to_pile(name, cards_to_throw=cards_to_throw)
        env.round_.pull_card(name, pick_from_deck=pick_from_deck)
        env.round_.pile_top_cards = env.round_.pile_top_cards_this_turn
        env.round_.update_players_knowledge(name)

        cards_in_hand_after = env.players[name].cards_in_hand
        n_cards_in_hand_after = len(cards_in_hand_after)

        reward_throw_pick_points = cards_to_value_sum(cards_in_hand_before) - cards_to_value_sum(cards_in_hand_after)
        reward_throw_pick_points *= REWARD_FACTOR_TURN_POINT_DIFFERENCE

        reward_throw_pick_n_cards = n_cards_in_hand_before - n_cards_in_hand_after
        reward_throw_pick_n_cards *= REWARD_FACTOR_TURN_N_CARDS
        reward_throw_pick = reward_throw_pick_points + reward_throw_pick_n_cards

    reward = reward_yaniv + reward_throw_pick

    # --- setting up opponent ---
    opponent_name = _name_to_opponent_name_two_players(name, env.players)

    n_deck = len(env.round_.round_deck)
    top_accessible_cards = pile_top_accessible_cards(env.round_.pile_top_cards)
    cards_in_hand = env.players[opponent_name].cards_in_hand

    observables = n_deck, top_accessible_cards, cards_in_hand, n_cards_in_hand_after

    return opponent_name, observables, reward, done, info



def basic_policy(observables, turn_number, seed=None, yaniv_thresh=None, throw_out='highest_combination',
                 pickup='random', deck_prob=0.5):
    assert throw_out in ['highest_card', 'highest_combination', 'random_card']
    assert pickup in ['random', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if yaniv_thresh is None:
        yaniv_thresh = YANIV_LIMIT

    n_deck = observables[0]
    top_accessible_cards = observables[1]
    cards_in_hand = observables[2]
    n_cards_opponent = observables[3]

    hand_sum = cards_to_value_sum(cards_in_hand)

    yaniv_call = 0
    if hand_sum <= yaniv_thresh:  # always call Yaniv
        yaniv_call = 1
        # continuing just in case the call was wrong ...

    # throwing out highest value card (not getting rid of cards yet ...)
    cards_in_hand_sorted = sort_cards(cards_in_hand, descending=True)

    np.random.seed(seed)
    if 'highest_card' == throw_out:
        cards_to_throw = [cards_in_hand_sorted[0]]
    elif 'highest_combination':
        cards_to_throw = cards_to_best_combination(cards_in_hand_sorted)
    elif 'random_card' == throw_out:
        cards_to_throw = [np.random.choice(cards_in_hand_sorted)]

    # picking from deck at random
    if 'random' == pickup:
        pick_from_deck = np.random.binomial(1, deck_prob)
    else:
        card_values = [card_to_value(card) for card in top_accessible_cards]
        idx_lowest = np.array(card_values).argmin()

        if pickup >= card_values[idx_lowest]:
            pick_from_deck = 0
        else:
            pick_from_deck = 1

    return (yaniv_call, cards_to_throw, pick_from_deck)


def get_game_setup(player_names, seed=None, play_jokers=False, do_stats=False, verbose=0):
    env = game.Game(player_names, seed=seed, verbose=verbose, do_stats=do_stats, play_jokers=play_jokers)

    env.initiate_players_status()
    env.players = env.get_round_players()

    return env


def get_random_round_environment(player_names, name, seed=1, play_jokers=False, verbose=0, turn_stop=3, do_stats=False):
    player_yaniv_thresh = {'Albert': None, 'Roland': None}  # {'Albert': 7, 'Roland': 7}
    player_throw_out_strategy = {'Albert': 'highest_combination', 'Roland': 'highest_combination'}
    player_pickup_strategy = {'Albert': 4, 'Roland': 4}

    game_setup = get_game_setup(player_names, seed=seed, play_jokers=play_jokers, do_stats=do_stats, verbose=verbose)

    turn_seed = seed + 1
    next_name, observables = new_round(game_setup, name=name, seed=turn_seed)

    for turn in range(1, turn_stop + 1):
        game_setup.round_.turn_output = {}

        turn_seed += 1
        name = next_name

        yaniv_thresh = player_yaniv_thresh[name]
        throw_out_strategy = player_throw_out_strategy[name]
        pickup_strategy = player_pickup_strategy[name]

        action = basic_policy(observables, turn,
                              yaniv_thresh=yaniv_thresh,
                              throw_out=throw_out_strategy,
                              pickup=pickup_strategy,
                              seed=turn_seed)

        next_name, observables, reward, done, info = step(game_setup, name, action)

    return game_setup, observables, done


def run_basic_comparison(players, seed=1, play_jokers=False, verbose=0, n_rounds=20, n_turns_max=100, do_stats=False, start_method='alternate'):
    game_setup = get_game_setup(players, seed=seed, play_jokers=play_jokers, do_stats=do_stats, verbose=verbose)

    player_total_rounds_rewards = {}
    for name in game_setup.players.keys():
        player_total_rounds_rewards[name] = []
    yaniv_declarers = []
    round_winners = []
    player_yaniv_thresh = {'Albert': None, 'Roland': None}  # {'Albert': 7, 'Roland': 7}
    player_throw_out_strategy = {'Albert': 'highest_combination', 'Roland': 'highest_combination'}
    player_pickup_strategy = {'Albert': 4, 'Roland': 4}

    turn_seed = seed + 1
    round_winner = None
    name_start = list(game_setup.players.keys())[:-1][0]

    for round_number in range(1, n_rounds + 1):
        name_start = next_starter_name(game_setup, pervious_starter=name_start, previous_winner=round_winner, seed=turn_seed,
                                       method=start_method)

        next_name, observables = new_round(game_setup, name=name_start, seed=turn_seed)

        game_setup.round_.round_output[round_number] = {}

        player_round_rewards = {}
        for name in game_setup.players.keys():
            player_round_rewards[name] = 0

        yaniv_declarer = None
        round_winner = None

        for turn in range(1, n_turns_max + 1):
            game_setup.round_.turn_output = {}
            turn_seed += 1
            name = next_name

            yaniv_thresh = player_yaniv_thresh[name]
            throw_out_strategy = player_throw_out_strategy[name]
            pickup_strategy = player_pickup_strategy[name]

            #print(observables)
            action = basic_policy(observables, turn,
                                  yaniv_thresh=yaniv_thresh,
                                  throw_out=throw_out_strategy,
                                  pickup=pickup_strategy,
                                  seed=turn_seed)

            next_name, observables, reward, done, info = step(game_setup, name, action)
            player_round_rewards[name] += reward

            game_setup.round_.round_output[round_number][turn] = game_setup.round_.turn_output
            if done:
                yaniv_declarer = name
                if info['yaniv_declare_correct'] == False:
                    print('incorrect sum for Yaniv')

                # TODO - update in case of info['yaniv_declare_correct'] == False
                for name in game_setup.players.keys():
                    player_round_rewards[name] -= game_setup.players[name].hand_points

                round_winner = game_setup.round_.round_output['winner']
                break
        if info['yaniv_declare_correct'] is None:
            for name in game_setup.players.keys():
                player_round_rewards[name] += REWARD_NO_WINNER

        # Testing to verify that all Assafs are different
        # if round_winner != yaniv_declarer:
        #    print(round_number, turn_seed)
        #    print(env.round_.players[yaniv_declarer].cards_in_hand, yaniv_declarer)
        #    print(env.round_.players[round_winner].cards_in_hand, round_winner)
        #    print('-' * 20)

        round_winners.append(round_winner)
        yaniv_declarers.append(yaniv_declarer)

        for name in game_setup.players.keys():
            player_total_rounds_rewards[name].append(player_round_rewards[name])



    df_results0 = pd.DataFrame(player_total_rounds_rewards)
    df_results0['declarer'] = yaniv_declarers
    df_results0['winner'] = round_winners

    return df_results0


def show_rounds_results(df_results):
    player_names = list(df_results.columns)
    player_names.remove('declarer')
    player_names.remove('winner')

    n_rounds_total = len(df_results)
    idx_declaration = df_results[df_results['declarer'].notnull()].index
    n_rounds_declared = len(idx_declaration)

    print(f'declared percentage: {100. * n_rounds_declared/n_rounds_total:0.1f}% of total {n_rounds_total:,} rounds')

    df_results = df_results.loc[idx_declaration]

    # The Assaf rate
    sr_assaf_counts = df_results[df_results['declarer'] != df_results['winner']]['winner'].value_counts(normalize=False)
    print('--- Assaf rate ---')
    print(f'{100. * sr_assaf_counts.sum() / n_rounds_declared:0.1f}%')
    print(sr_assaf_counts)

    # Winner rate
    sr_winner_rate = df_results['winner'].value_counts()
    print('--- Winner rate ---')
    print(sr_winner_rate)
    print(sr_winner_rate / n_rounds_declared)

    for name in player_names:
        plt.hist(df_results[name], histtype='step', linewidth=3, label=name)
    plt.legend(loc='upper left')