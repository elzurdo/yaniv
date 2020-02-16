try:
    import matplotlib.pyplot as plt
except:
    print('did not manage to import matplotlib.pyplot.\ncontinuing without')
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np

assert tf.__version__ >= "2.0"

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
            info['winner'] = env.round_.round_output['winner']
        else:
            reward_yaniv = REWARD_FACTOR_YANIV_INCORRECT_PENALTY
            info['yaniv_declare_correct'] = False
            info['winner'] = None

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
        info['winner'] = None

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

    if seed:
        turn_seed = seed + 1
    else:
        turn_seed = None
    next_name, observables = new_round(game_setup, name=name, seed=turn_seed)

    for turn in range(1, turn_stop + 1):
        game_setup.round_.turn_output = {}

        if turn_seed:
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


def run_basic_comparison(players_names, seed=1, play_jokers=False, verbose=0, n_rounds=20, n_turns_max=100,
                         do_stats=False, start_method='alternate', test_first_to_yaniv=False):
    game_setup = get_game_setup(players_names, seed=seed, play_jokers=play_jokers, do_stats=do_stats, verbose=verbose)

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

        for turn_number in range(1, n_turns_max + 1):
            game_setup.round_.turn_output = {}
            turn_seed += 1
            name = next_name

            yaniv_thresh = player_yaniv_thresh[name]
            throw_out_strategy = player_throw_out_strategy[name]
            pickup_strategy = player_pickup_strategy[name]

            action = basic_policy(observables, turn_number,
                                  yaniv_thresh=yaniv_thresh,
                                  throw_out=throw_out_strategy,
                                  pickup=pickup_strategy,
                                  seed=turn_seed)

            next_name, observables, reward, done, info = step(game_setup, name, action)
            player_round_rewards[name] += reward

            game_setup.round_.round_output[round_number][turn_number] = game_setup.round_.turn_output

            if test_first_to_yaniv & (cards_to_value_sum(observables[2]) <= YANIV_LIMIT):
                reward_for_reaching_yaniv_thresh = 1.5 * YANIV_LIMIT - cards_to_value_sum(observables[2])
                player_round_rewards[name] += reward_for_reaching_yaniv_thresh

                # opponent
                player_round_rewards[next_name] -= game_setup.players[next_name].hand_points

                # for the sake of consistency in what follows
                info['yaniv_declare_correct'] = True  # to ensure not to get REWARD_NO_WINNER
                round_winner = name
                yaniv_declarer = name

                break

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


        round_winners.append(round_winner)
        yaniv_declarers.append(yaniv_declarer)

        for name in game_setup.players.keys():
            player_total_rounds_rewards[name].append(player_round_rewards[name])



    df_results0 = pd.DataFrame(player_total_rounds_rewards)
    df_results0['declarer'] = yaniv_declarers
    df_results0['winner'] = round_winners

    return df_results0


def show_rounds_results(df_results, plot=False):
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
    # print(df_results[df_results['declarer'] != df_results['winner']].head(4))

    # Winner rate
    sr_winner_rate = df_results['winner'].value_counts()
    print('--- Winner rate ---')
    print(sr_winner_rate)
    print(sr_winner_rate / n_rounds_declared)

    if plot:
        for name in player_names:
            plt.hist(df_results[name], histtype='step', linewidth=3, label=name)
        plt.legend(loc='upper left')


def observables_to_model_input(observables, turn_number, reshape=True):
    n_deck = observables[0]
    top_accessible_cards = observables[1]
    cards_in_hand = observables[2]
    n_cards_opponent = observables[3]

    card_values = [card_to_value(card) for card in top_accessible_cards]
    idx_lowest = np.array(card_values).argmin()
    card_value_lowest = card_values[idx_lowest]

    input_ = np.array((n_deck, card_value_lowest, turn_number, len(cards_in_hand), n_cards_opponent))

    if reshape:
        input_ = input_.reshape(1, -1)

    return input_


def ann_policy(model, observables, turn_number, seed=None, yaniv_thresh=None, throw_out='highest_combination'):
    """Takes in one set of obervables

    :param model:
    :param observables:
    :param turn_number:
    :param seed:
    :param yaniv_thresh:
    :param throw_out:
    :return:
    """
    input_ = observables_to_model_input(observables, turn_number)

    deck_prob = model.predict(input_)[0]

    action = basic_policy(observables, turn_number, seed=seed, yaniv_thresh=yaniv_thresh,
                          throw_out=throw_out, pickup='random', deck_prob=deck_prob)

    return action


def run_ann_policy(player_names, model, seed=1, play_jokers=False, n_rounds=20, n_turns_max=100,
                   start_method='alternate', verbose=0, fixed_pickup_strategy=4, do_stats=False,
                   test_first_to_yaniv=False):

    game_setup = get_game_setup(player_names, seed=seed, play_jokers=play_jokers, do_stats=do_stats, verbose=verbose)

    player_total_rounds_rewards = {}
    for name in game_setup.players.keys():
        player_total_rounds_rewards[name] = []
    yaniv_declarers = []
    round_winners = []
    player_yaniv_thresh = {'Albert': 7, 'ANN': 7}  # {'Albert': 7, 'Roland': 7}
    player_throw_out_strategy = {'Albert': 'highest_combination', 'ANN': 'highest_combination'}
    player_pickup_strategy = {'Albert': fixed_pickup_strategy, 'ANN': None}

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

            if 'Albert' == name:
                #print(observables)
                action = basic_policy(observables,
                                      turn,
                                      yaniv_thresh=yaniv_thresh,
                                      throw_out=throw_out_strategy,
                                      pickup=pickup_strategy,
                                      seed=turn_seed)
            elif 'ANN' == name:
                #print(observables)
                action = ann_policy(model, observables,
                                   turn, throw_out=throw_out_strategy,
                                   seed=turn_seed)
                #print(action)


            next_name, observables, reward, done, info = step(game_setup, name, action)
            player_round_rewards[name] += reward

            game_setup.round_.round_output[round_number][turn] = game_setup.round_.turn_output

            if test_first_to_yaniv & (cards_to_value_sum(observables[2]) <= YANIV_LIMIT):
                reward_for_reaching_yaniv_thresh = 1.5 * YANIV_LIMIT - cards_to_value_sum(observables[2])
                player_round_rewards[name] += reward_for_reaching_yaniv_thresh

                # opponent
                player_round_rewards[next_name] -= game_setup.players[next_name].hand_points

                # for the sake of consistency in what follows
                info['yaniv_declare_correct'] = True  # to ensure not to get REWARD_NO_WINNER
                round_winner = name
                yaniv_declarer = name

                break


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


        round_winners.append(round_winner)
        yaniv_declarers.append(yaniv_declarer)

        for name in game_setup.players.keys():
            player_total_rounds_rewards[name].append(player_round_rewards[name])


    df_results0 = pd.DataFrame(player_total_rounds_rewards)
    df_results0['declarer'] = yaniv_declarers
    df_results0['winner'] = round_winners

    return df_results0


def create_deck_pickup_dnn(n_inputs, seed=None, verbose=1, clear_session=True):
    if clear_session:
        keras.backend.clear_session()

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = keras.models.Sequential([
        keras.layers.Dense(n_inputs + 1, activation="elu", input_shape=[n_inputs]),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    if verbose:
        print(model.summary())

    return model


def get_environment_vars(players_names, name, max_turn=40, seed=None, play_jokers=False, verbose=0, do_stats=False):
    """
    Gets environment vars for player `name`.
    Tested for len(players_names) == 2

    :param players_names:
    :param name:
    :param seed:
    :return:
    """
    turns_stopping = list(range(2, max_turn + 1, len(players_names)))  # verifying that only observables for `name` is comine out

    done = True

    while done:
        np.random.seed(seed)
        turn_stop = np.random.choice(turns_stopping)
        game_setup, observables, done = get_random_round_environment(players_names, name, seed=seed,
                                                                     play_jokers=play_jokers,
                                                                     verbose=verbose, turn_stop=turn_stop,
                                                                     do_stats=do_stats)

        if seed:
            seed += 10000

    return game_setup, observables, turn_stop


def get_multiple_environment_vars(n_envs, players_names, name, seed=None):
    l_game_setups = []
    l_observables = []
    l_turn = []

    for ienv in range(n_envs):
        if seed:
            seed_env = ienv + seed
        else:
            seed_env = None

        game_setup, observables, turn_stop = get_environment_vars(players_names, name, seed=seed_env)

        l_game_setups.append(game_setup)
        l_observables.append(observables)
        l_turn.append(turn_stop)

    return l_game_setups, l_observables, l_turn



def step_two_players__ann_then_fixed(game_setup, name, observables, turn_number, deck_prob, max_turn, basic_strategy,
                                     yaniv_thresh=None, throw_out_strategy='highest_combination', seed=None, seed_factor=1,
                                     continue_after_done=True):

    #print(deck_prob)
    #print(observables)
    this_turn_number = int(turn_number)
    ann_action = basic_policy(observables, turn_number, seed=seed,
                          yaniv_thresh=yaniv_thresh, throw_out=throw_out_strategy,
                          pickup='random', deck_prob=deck_prob)
    #print(ann_action)

    opponent_name, opponent_observables, ann_reward, done, info = step(game_setup, name, ann_action)

    # probably should be somewhere else for more generic purposes ...
    if not game_setup.round_.round_deck:
        done = True

    if turn_number > max_turn:
        done = True

    opponent_reward = None
    if done:
        if continue_after_done:
            players_names = list(game_setup.players.keys())
            game_setup, observables, turn_number = get_environment_vars(players_names, name,
                                                                        seed=seed_factor * 100000 + seed)
        else:
            observables = None
            turn_number = None

    else:
        turn_number += 1  # oppenent's turn

        if seed:
            turn_seed = seed + 1
        else:
            turn_seed = None
        opponent_action = basic_policy(opponent_observables, turn_number, seed=turn_seed,
                                       yaniv_thresh=yaniv_thresh, throw_out=throw_out_strategy,
                                       pickup=basic_strategy, deck_prob=None)


        my_name, observables, opponent_reward, done_opponent, info = step(game_setup, opponent_name,
                                                                          opponent_action)
        turn_number += 1  # my turn

        if done_opponent:
            if continue_after_done:
                players_names = list(game_setup.players.keys())
                game_setup, observables, turn_number = get_environment_vars(players_names, name,
                                                                            seed=seed_factor * 100000 + seed)
            else:
                observables = None
                turn_number = None
                done = True


    if continue_after_done:
        return game_setup, observables, turn_number

    else:
        if done:
            #print('done ann_reward:', ann_reward, game_setup.players[name].hand_points, game_setup.players[name].cards_in_hand)
            #print(game_setup.players[name].hand_points, 'hand points')
            #ann_reward -= game_setup.players[name].hand_points

            if info['winner'] == 'Albert':
                ann_reward += 1
            elif info['winner'] == 'Roland':
                ann_reward = -1
            # print(info['winner'], this_turn_number, max_turn)

            #if opponent_reward:
            #    opponent_reward -= game_setup.players[opponent_name].hand_points

            #print(info['winner'], info['yaniv_declare_correct'])
            #round_winner = game_setup.round_.round_output['winner']

            if info['yaniv_declare_correct'] is None:
                REWARD_NO_WINNER = -0.5
                ann_reward += REWARD_NO_WINNER
                if opponent_reward:
                    opponent_reward += REWARD_NO_WINNER
            #print(ann_reward, opponent_reward, 'ann, opponent rewards')
            #print('-' * 30)


        return observables, turn_number, done, ann_reward, opponent_reward, ann_action



def train_deck_pickup_dnn_to_player(n_iters, n_envs, n_inputs,
                                    basic_strategy=4, yaniv_thresh=None, throw_out_strategy='highest_combination',
                                    max_turn = 40, collect_data=False, seed=None
                                    ):
    """
    Assuming simulation of two players, it will train a model to follow the deck pickup basic strategy of the first
    player.


    :param n_iters:
    :param n_envs:
    :param n_inputs:
    :param basic_strategy:
    :param yaniv_thresh:
    :param throw_out_strategy:
    :param max_turn:
    :param seed:
    :return:

    Example usage:

    # --- training ---
    n_environments = 50
    n_iterations = 200
    n_inputs = 5 # [n_deck, card_value_lowest, turn_number, n_cards_i, ncards_j]

    deck_pickup_dnn, l_losses = train_deck_pickup_dnn_to_player(n_iterations, n_environments, n_inputs, seed=2)

    """


    players_names = ['Albert', 'Roland']  # players in simulation
    name = 'Albert'  # player that we are training to match

    print(f'creating {n_envs} environments to train on')
    l_game_setups, l_observables, l_turn = get_multiple_environment_vars(n_envs, players_names, name, seed=seed)

    deck_pickup_dnn = create_deck_pickup_dnn(n_inputs, seed=1)
    optimizer = keras.optimizers.RMSprop()
    loss_fn = keras.losses.binary_crossentropy

    l_losses = []  # for plotting

    if collect_data:
        columns =['n_deck', 'card_lowest', 'turn_number', 'n_cards_i', 'n_cards_j']
        df_data = pd.DataFrame(columns=columns)
        l_targets = []

    for iteration in range(n_iters):
        target_probas = []
        for ienv, (observables, turn_number) in enumerate(zip(l_observables, l_turn)):
            target_action = basic_policy(observables, turn_number, seed=ienv,
                                         yaniv_thresh=yaniv_thresh, throw_out=throw_out_strategy,
                                         pickup=basic_strategy, deck_prob=None)
            target_probas.append(target_action[-1])

        if collect_data:
            l_targets += target_probas
            
        target_probas = np.array(target_probas).reshape(-1, 1)

        inputs = np.zeros([len(l_observables), n_inputs])
        idx = 0
        for observables, turn_number in zip(l_observables, l_turn):
            this_input = observables_to_model_input(observables, turn_number, reshape=False)
            inputs[idx, :] = this_input
            idx += 1

        if collect_data:
            df_data = df_data.append(pd.DataFrame(inputs, columns=columns), ignore_index=True)

        with tf.GradientTape() as tape:
            deck_pick_probas = deck_pickup_dnn(np.array(inputs))
            loss = tf.reduce_mean(loss_fn(target_probas, deck_pick_probas))

        str_ =  f"Iteration: {iteration + 1} of {n_iters}"
        str_ += f" Loss: {loss.numpy():.3f} "
        str_ += f" Turns: ({np.min(l_turn):0.0f}, {np.mean(l_turn):0.1f}, {np.max(l_turn):0.0f})"
        str_ += f" Targets: {target_probas[:4].ravel()} ({np.mean(target_probas):0.1f})"
        print(str_, end="\r")
        grads = tape.gradient(loss, deck_pickup_dnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, deck_pickup_dnn.trainable_variables))
        l_losses.append(float(loss))

        # updating environments
        for ienv, [deck_prob, observables, game_setup, turn_number] in enumerate(zip(deck_pick_probas,
                                                                                     l_observables,
                                                                                     l_game_setups, l_turn
                                                                                     )):

            game_setup, observables, turn_number = step_two_players__ann_then_fixed(game_setup, name, observables,
                                                                                    turn_number, deck_prob, max_turn, basic_strategy,
                                                                                    yaniv_thresh=yaniv_thresh,
                                                                                    throw_out_strategy=throw_out_strategy,
                                                                                    seed=ienv, seed_factor=iteration)

            l_game_setups[ienv] = game_setup
            l_observables[ienv] = observables
            l_turn[ienv] = turn_number


    if collect_data:
        df_data['target_probs'] = l_targets
        return deck_pickup_dnn, l_losses, df_data

    return deck_pickup_dnn, l_losses



def play_one_turn(game_setup, observables, turn_number, model, loss_fn, max_turn, basic_strategy,
                  yaniv_thresh=None, throw_out_strategy='highest_combination', seed=None, seed_factor=1):

    #target_action = basic_policy(observables, turn_number, seed=seed,
    #                             yaniv_thresh=yaniv_thresh, throw_out=throw_out_strategy,
    #                             pickup=basic_strategy, deck_prob=None)

    #target_prob = target_action[-1]
    input_ = observables_to_model_input(observables, turn_number, reshape=True)

    with tf.GradientTape() as tape:
        deck_pick_prob = model(input_)
        action_ = (tf.random.uniform([1, 1]) > deck_pick_prob) # action 0: pile, action 1: deck (might be the opposite of what I want ...)
        y_target = tf.constant([[1.]]) - tf.cast(action_, tf.float32) # y_target is going to agree with deck_pick_prob
        loss = tf.reduce_mean(loss_fn(y_target, deck_pick_prob))


    grads = tape.gradient(loss, model.trainable_variables)
    observables, turn_number, done, ann_reward, opponent_reward, ann_action = step_two_players__ann_then_fixed(
        game_setup, 'Albert', observables, turn_number, deck_pick_prob, max_turn, basic_strategy,
        yaniv_thresh=yaniv_thresh, throw_out_strategy=throw_out_strategy, seed=seed, seed_factor=seed_factor,
        continue_after_done=False)


    return observables, ann_reward, done, grads, ann_action, turn_number


def play_multiple_rounds(n_envs, max_turn_play, model, loss_fn, max_turn_start=10, seed=None,
                         play_jokers=False, verbose=0, do_stats=False):
    basic_strategy = 4

    all_rewards = []
    all_grads = []

    for ienv in range(n_envs):
        current_rewards = []
        current_grads = []

        if seed:
            game_seed = ienv + 1 + seed * 10000  # seed=0 means no seed!
        else:
            game_seed = None
        game_setup, observation, turn_number = get_environment_vars(['Albert', 'Roland'], 'Albert', max_turn=max_turn_start, seed=game_seed,
                                 play_jokers=play_jokers, verbose=verbose, do_stats=do_stats)

        # print('-' * 30)
        # print(turn_number)
        for step in range(max_turn_play):
            # print(turn_number, 'before')

            if seed:
                seed_step = step + 1 + game_seed * 100
            else:
                seed_step = None
            observation, ann_reward, done, grads, ann_action, turn_number = \
                play_one_turn(game_setup, observation, turn_number, model, loss_fn, max_turn_play, basic_strategy,
                              yaniv_thresh=None, throw_out_strategy='highest_combination', seed=seed_step,
                              seed_factor=1)
            # print(turn_number, 'after')
            # print('~' *10)
            # print(ann_reward)
            current_rewards.append(ann_reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
        # print('-' * 30)
    return all_rewards, all_grads


def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]

"""
# use case: 
from rl_utils import play_one_turn, play_multiple_rounds, discount_rewards, discount_and_normalize_rewards
from tensorflow import keras

n_iterations = 200
n_episodes_per_update = 20 #25
n_max_steps = 40
discount_rate = 0.9

learning_rate = 0.001 # 0.01
optimizer = keras.optimizers.Adam(lr=learning_rate)
loss_fn = keras.losses.binary_crossentropy

deck_pickup_dnn_gradients = create_deck_pickup_dnn(n_inputs, seed=1)
total_points_by_env = []

for iteration in range(n_iterations):
    # play_multiple_episodes(n_envs, max_turn, model, loss_fn)
    all_rewards, all_grads = play_multiple_rounds(n_episodes_per_update, n_max_steps, deck_pickup_dnn_gradients, loss_fn, seed=iteration)
    total_rewards = sum(map(sum, all_rewards))                     # Not shown in the book
    total_points_by_env.append(total_rewards / n_episodes_per_update)
    print("\rIteration: {}, mean rewards: {:.1f}".format(          # Not shown
        iteration, total_rewards / n_episodes_per_update), end="") # Not shown
    all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                       discount_rate)
    all_mean_grads = []
    for var_index in range(len(deck_pickup_dnn_gradients.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, deck_pickup_dnn_gradients.trainable_variables))

"""

