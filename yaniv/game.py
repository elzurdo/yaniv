# -*- coding: utf-8 -*-

import numpy as np
import sys

from cards import (get_deck, card_to_pretty,
                   cards_to_valid_throw_combinations,
                   sort_card_combos,
                   card_to_value,
                   cards_to_values,
                   cards_same_rank,
                   sort_cards,
                   pile_cards_plus_player_collective_hypothetical_points
                   )

from stats import (card_number_to_max_card_value_to_declare_yaniv,
                   calculate_prob_all_cards_under_thresh,
                   calculate_prob_ht_gt_hi,
                   is_smaller_or_equal_binary,
                   calculate_p_hj_gt_hi_n_j_prior
                   )


from configs import (ASSAF_PENALTY, END_GAME_SCORE, MAX_ROUNDS, MAX_TURNS, YANIV_LIMIT)

from strategies import (pile_conservative_vary, pile_conservative_constant, pile_always)


# TODO: design and implement different throw_strategy
# TODO: design and implement different yaniv_strategy ('always' (i.e, 7), 'only below 4', 'by statistics')
# TODO: design and implement different pile_pull_strategy
# TODO: figure out and implement how to change strategy as game progresses (so to maximise utility)
class Player():
    def __init__(self, name, agent='bot'):
        assert agent in ['bot', 'human']

        self.name = name
        self.agent = agent

        self.starts_round = False
        self.unknown_cards = []

    def sum_hand_points(self):
        '''Calculates the sum of point in a hand
        '''

        self.hand_points = 0
        for card in self.cards_in_hand:
            self.hand_points += card_to_value(card)

    def add_cards_to_unknown(self, cards):
        self.unknown_cards = self.unknown_cards + list(cards)

    def remove_cards_from_unknown(self, cards):
        self.unknown_cards = list(set(self.unknown_cards) - set(cards))

    def knowledgewise_assign_card_to_player(self, other_player_name, card):
        self.other_players_known_cards[other_player_name].append(card)

    def knowledgewise_drop_cards_from_player(self, other_player_name, cards):
        remaining_cards = list(set(self.other_players_known_cards[other_player_name]) - set(cards))
        self.other_players_known_cards[other_player_name] = remaining_cards

class Game():
    def __init__(self, players, end_game_score=None, assaf_penalty=None, play_jokers=True, do_stats=False,
                 verbose=1, seed=None):
        '''
        verbose:
        0 - display player names, winner his/her score
        1 - in addition to 0, displays Round information
        2 - in addition to 1, displays probability of a Yaniv call being successful
        3 - in addition to 2, displays how the probability was derived

        :param player_names: list of names
        :param end_game_score: int. Score which terminates game for player
        :param assaf_penalty: int. The penalty score for an Assaf
        :param jokers: bool. True to use 2 jokers in the game False without.
        :param verbose: int
        :param seed: int
        '''

        self.seed = seed

        if assaf_penalty:
            self.assaf_penalty = assaf_penalty
        else:
            self.assaf_penalty = ASSAF_PENALTY

        if end_game_score:
            self.end_game_score = end_game_score
        else:
            self.end_game_score = END_GAME_SCORE

        self.play_jokers = play_jokers
        self.verbose = verbose
        self.do_stats = do_stats

        self.generate_players(players)
        self.deck = get_deck(play_jokers=play_jokers, shuffle=False, seed=self.seed)
        if 1:
            print(f'Deck of {len(self.deck)} cards\n{list(map(card_to_pretty, self.deck))}')
        self.game_output = {}

    def play(self):
        self.initiate_players_status()

        self.play_game()

        return self.game_output

    def _seeding(self):
        if self.seed:
            np.random.seed(seed=self.seed)
            self.seed += 1


    # TODO: print all strategies
    def generate_players(self, input_players):
        '''Given a list of players names creates a list of of Player objects

        :param player_names:
        :return:
        '''

        self.all_players = []
        print('generating players and their playing strategies:')

        if isinstance(input_players, list):
            input_players = {name: 'bot' for name in input_players}


        for idx, name in enumerate(input_players):
            self._seeding()
            player = Player(name, agent=input_players[name])
            player.id = idx
            self.all_players.append(player)

            print(f'{name} ({player.agent})')

    def initiate_players_status(self):
        '''Initiates the status of all the players
        In particular:
        * score - int. starts at 0 points
        * in_play - bool. True: plays current round. False: already lossed the game

        :return: None
        '''
        for iplayer, player in enumerate(self.all_players):
            player.game_score = 0 # was player.score ...
            player.in_play = True

            if iplayer == 0:
                player.starts_round = True


    def get_round_players(self): # was _round_player
        '''Returns a dictionary of Player objects only of players that have less the end_game_score.

        This is used to track which players advance to the next round

        :return: dict. keys are player names (str) and values are the corresponding Player objects
        '''
        players = {}

        for player in self.all_players:
            if player.in_play:
                if player.game_score < self.end_game_score:
                    players[player.name] = player
                    if self.verbose >= 3:
                        print(player.name, player.game_score, 'IN')
                else:
                    player.in_play = False
                    if self.verbose >= 2:
                        print(player.name, player.game_score, 'OUT')
                        print('-' * 20)

        return players

    def play_game(self):
        '''Game play

        :return:
        '''

        # card_num_to_max_value is a mapping from the number of cards in hand to the maximum value of single card in hand
        # that can result in a successful Yaniv. This is useful to calculate heuristics of Yaniv success probabilities.
        card_num_to_max_value = card_number_to_max_card_value_to_declare_yaniv(play_jokers=self.play_jokers)

        # round number counter. The first round is value 1, so setting counter to 0.
        round_number = 0

        # players is a dictionary of players that progress to the next round (they have less than max_score)
        players = self.get_round_players()

        while (len(players) > 1) and (round_number < MAX_ROUNDS):  # the game terminates when left with one player
            if self.verbose:
                print('=' * 20)
            round_number += 1
            self.game_output[round_number] = {}
            if self.verbose >= 1:
                print('Round: {:,}'.format(round_number))

            # Declaring Round object and playing a round
            # TODO: make card_num_2_max_value dyanmic

            self.round = Round(players, self.deck, assaf_penalty=self.assaf_penalty,
                               card_num_to_max_value=card_num_to_max_value, verbose=self.verbose, seed=self.seed,
                               round_output=self.game_output[round_number], do_stats=self.do_stats, play_jokers=self.play_jokers)
            self.round.number = round_number
            self.round.play()

            #"""

            if self.seed:
                self.seed += 1
            # ====== player score updating ==============
            self.game_output[round_number]['end'] = {}
            for name, player in players.items():
                # hand points go into score.
                # If Yaniv was successful the caller gets 0 points (otherwise the original hand plus the assaf_penalty
                # All other players get hand_points (including if one was an Assafer)
                self.game_output[round_number]['end'][name] = {}
                self.game_output[round_number]['end'][name]['gained_points'] = player.hand_points

                player.game_score += player.hand_points

                # Jackpot!
                # If a player hits these luck values their score is reduced
                if player.game_score == 100:
                    player.game_score = 50
                    print(f"Round {round_number}:\n\tLucky {name}! Aggregated 100 points reduced to 50")
                    self.game_output[round_number]['end']['lucky_50'] = 1
                elif player.game_score == 200:
                    print(f"Round {round_number}\n\tLucky {name}! Aggregated 200 points reduced to 150")
                    player.game_score = 150
                    self.game_output[round_number]['end']['lucky_150'] = 1

                self.game_output[round_number]['end'][name]['points_conclusion'] = player.game_score


                if self.verbose:
                    print(player.name, player.hand_points, [card_to_pretty(card) for card in player.cards_in_hand], player.game_score)
            # ===========================

            # Round conclusion
            players = self.get_round_players()  # players for next round

            if round_number > MAX_ROUNDS:
                print('breaking at max rounds: {:,}'.format(MAX_ROUNDS))
                break
            #"""

        if len(players) == 1:
            winner = players[list(players.keys())[0]]
            print("{} is the winner with {:,} points".format(winner.name, winner.game_score))
        elif round_number == MAX_ROUNDS:
            print("Round {} reached max rounds. Left standing: {}".format(round_number, ', '.join([player for player in players])))
        else:
            # Case of 0 players left (double-or-more knockout case)
            print("Everybody loses ... ({} players left)".format(len(players)))

class Round():
    def __init__(self, players, deck, card_num_to_max_value=None, assaf_penalty=30, seed=4, verbose=0, do_stats=False,
                 round_output=None, play_jokers=True):
        self.seed = seed
        self.verbose = verbose
        self.assaf_penalty = assaf_penalty
        self.deck = deck
        self.card_num_to_max_value = card_num_to_max_value
        self.cards_thrown = []
        self.do_stats = do_stats

        self.players = players

        #self.meta = {}  # meta data for logging
        self.round_output = round_output
        self.ax = None
        self.play_jokers = play_jokers

    def play(self):
        # round starts with a full deck
        self.round_deck = list(self.deck)

        np.random.seed(self.seed)
        np.random.shuffle(self.round_deck)

        self.distribute_cards()

        self.play_round()

    def _seeding(self):
        if self.seed:
            np.random.seed(seed=self.seed)
            self.seed += 1


    def distribute_cards(self, num_cards=5):
        '''Distributing num_cards to each Player

        :param num_cards: int. Number of cards in each hand at beginning of a round
        :return:
        '''

        full_deck = self.round_deck.copy()

        self.round_output['start'] = {}
        self.round_output['start']['deck_ordered'] = full_deck
        n_players_left = len(self.players)
        for name, player in self.players.items():
            player.add_cards_to_unknown(full_deck)

            # assigning selected cards to Player
            player.cards_in_hand = self.round_deck[::n_players_left][:num_cards]

            self.round_output['start'][f'{name}_cards'] = list(player.cards_in_hand)
            # calculating points in a hand of Player
            player.sum_hand_points()
            player.remove_cards_from_unknown(player.cards_in_hand)

            # Deleting selected cards from the round's deck
            for card in player.cards_in_hand:
                self.round_deck.remove(card)

            n_players_left -= 1

        card = np.random.choice(self.round_deck, size=1, replace=False)
        self.pile_top_cards = card
        self.round_deck.remove(card)
        self.update_players_knowledge(None) # `card` does not belong to anyone

    # TODO: make tidier, hopefully without using pandas
    def get_player_order(self):
        '''Determining the player order

        Basic idea:
        One of the players should have a starts_round=True (based on default or if they won the previous round,
        by means of Yaniv-ing or Assaf-ing).
        Then all the rest of the players are ordered by their indexes (insilico analogy of by "seating order")

        :return:
        '''
        starting_player_name = None

        for idx, (name, player) in enumerate(self.players.items()):
            if player.starts_round == True:

                if starting_player_name:
                    print('Error! {} and {} both have starting status'.format(starting_player_name, name))
                    sys.exit(1)

                starting_player_name = name
                idx_starting = idx

        if starting_player_name is None: # e.g, if the Assafer loses game because of points
            idx_starting = 0
            starting_player_name = list(self.players.keys())[idx_starting]

        if self.verbose:
            print('Player starting the round: {}'.format(starting_player_name))

        player_names = list(self.players.keys())
        idxs = np.arange(len(player_names))
        order_ids = (idxs - idx_starting) % len(self.players)
        player_names_ordered = {order_id: player_names[idx] for idx, order_id in zip(idxs, order_ids)}
        player_names_ordered = [player_names_ordered[order_id] for order_id in sorted(player_names_ordered, reverse=False)]

        players_ordered = {}
        for name in player_names_ordered:
            players_ordered[name] = self.players[name]

        return players_ordered

    # Todo: add probabalistic approaches Issue #3
    # Todo: self.do_stats should be related to game strategy
    def decide_declare_yaniv(self, name):
        '''Returns Player boolean decision if to declare Yaniv

        The decision is based on yaniv_strategy
        (and when we introduce probabilistic approaches also on their prediction of success)

        :param name: str. Name of Player considering declaring
        :return: bool. True: declare Yaniv (end round), False: continue the round play
        '''

        prob_successful_yaniv = None
        if self.do_stats:
            if len(self.players) > 2:
                prob_successful_yaniv = self.prob_lowest_hand(name)
            else:
                prob_successful_yaniv = self.prob_lowest_hand_two_players(name)
            self.turn_output['yaniv_success_prob'] = prob_successful_yaniv

        player = self.players[name]
        if 'bot' == player.agent:
            if ('always' == player.strategy['yaniv_declare']) or (prob_successful_yaniv is None):
                return True
            else:
                if prob_successful_yaniv >= player.strategy['prob_successful_yaniv_thresh']:
                    return True
        else:
            id_to_result = {'1': True, '2': False}
            options = {'1': 'Yes', '2': 'No'}
            option_id = input(f'Yaniv? {options}')

            return id_to_result[option_id]

        return False
        
    # TODO: The Assaf should be the Assafer with the lowest value. If two equally lowest choose randomly
    # TODO: deal with collecting meta data, and flag self.collect_meta
    def round_summary(self, name_yaniv):
        '''Summarising round results

        * Figures out if the Yaniv declaration was successful or if Assafed.
        * Updates hand points accordingly (e.g, if Assafed the hand of declarer increases by assaf_penalty)
        * Determining who starts the next round (if successful Yaniv call it goes to name_yaniv Player,
        otherwise one of the Assafers)

        :param name_yaniv: str. name of person declaring Yaniv
        :return: None
        '''


        assafed = False
        yaniv_player = self.players[name_yaniv]
        # Yaniv caller points. This is the benchmark for Assaf-ing.
        self.yaniv_points = int(yaniv_player.hand_points)

        if self.verbose:
            print('~' * 10)
            print('Round Conclusion')
            print('{} declared Yaniv with {}'.format(name_yaniv, yaniv_player.hand_points))

        """
        if self.collect_meta:
            turn_number = self._get_highest_turn_number()  # turn_number used in self.meta below
            self.meta[turn_number]["declared_yaniv"] = True
        """

        assafers = []  # list of all people who can call Assaf
        for name, player in self.players.items():
            player.starts_round = False  # zero-ing out those that start round

            if name != name_yaniv:

                if player.hand_points <= self.yaniv_points:
                    assafed = True
                    assafers.append(name)


        if assafed:
            assafer_name = assafers[0]  # currently using the first indexed as the Assafer
            self.turn_output['assafed_by'] = assafer_name
            self.round_output['winner'] = assafer_name
            if self.verbose:
                print('ASSAF!')
                print('{} Assafed by: {} (hand of {})'.format(name_yaniv, assafers[0],
                                                              self.players[assafer_name].hand_points))

            """
            if self.collect_meta:
                self.meta[turn_number]["assafed"] = {}
                self.meta[turn_number]["assafed"]["by"] = assafer_name
                self.meta[turn_number]["assafed"]["assafer_points"] = self.players[assafer_name].hand_points
            """
            # The Yaniv declarer is penalised by assaf_penalty because of incorrect call
            self.players[name_yaniv].hand_points += self.assaf_penalty
            # The Assafer gets to start the next round
            self.players[assafer_name].starts_round = True
        else:
            # Yaniv was successful so the caller does not get points
            self.players[name_yaniv].hand_points = 0
            # ... and gets to start the next round.
            yaniv_player.starts_round = True
            self.round_output['winner'] = name_yaniv

        """
        if self.collect_meta:
            self.meta[turn_number]["hand_points_end"] = player.hand_points
        """

    def play_round(self, players_ordered=None, turn=0):
        # flag to finish round. True: keep playing, False: end of round.
        yaniv_declared = False

        if not players_ordered:
            players_ordered = self.get_player_order()

            # each player accounts for other players hands. In beginning no knowledge
            for name_i, player_i in players_ordered.items():
                player_i.other_players_known_cards = {}
                for name_j, player_j in players_ordered.items():
                    if name_i != name_j:
                        player_i.other_players_known_cards[name_j] = []

        #print('playing order: ', ', '.join(list(players_ordered.keys())))

        for name, player in players_ordered.items():
            turn += 1
            self.turn_number = turn
            self.round_output[turn] = {}
            self.turn_output = self.round_output[turn]

            player.strategy = pile_conservative_constant()

            self.turn_output['name'] = name
            self.turn_output['pile_top_accessible'] = list(self.pile_top_accessible_cards())
            #self.turn_output[name] = list(player.cards_in_hand) # this might be redundant information
            for name_j, player_j in players_ordered.items():
                self.turn_output[f'{name_j}_ncards'] = len(list(player_j.cards_in_hand))
                self.turn_output[f'{name_j}_cards'] = list(player_j.cards_in_hand)
            # self._log_meta_data(player)

            # ================
            # player.action(yaniv_declared)
            # ================

            if not yaniv_declared:
                if player.hand_points <= YANIV_LIMIT:
                    # name considers declearing yaniv based on their Player.yaniv_strategy probability of success

                    # ------ return to this ------
                    yaniv_declared = self.decide_declare_yaniv(name)
                # ----------temporary ---------
                ## yaniv_declared = np.random.choice([True, False], size=1, p=[0.4, 0.6])[0]
                # -----------------------------
                if yaniv_declared:
                    self.turn_output['yaniv_call'] = 1
                    # round ends
                    self.round_summary(name)
                    return None

                else:
                    self.throw_cards_to_pile(name)
                    self.pull_card(name)
                    self.pile_top_cards = self.pile_top_cards_this_turn

                    self.update_players_knowledge(name)

        if not yaniv_declared:
            if turn >= MAX_TURNS:
                print(f'Reached {MAX_TURNS} turns, terminating round')
                self.turn_output['max_turns'] = 1
                return None
            # at this stage we did a full "circle around the table",
            # but did not conclude with a Yaniv declaration. We will go for another round
            # perhaps there is a better way of doing this loop.
            self.play_round(players_ordered=players_ordered, turn=turn)

    # TODO: use pile_accessible_cards function from cards.py
    def pile_top_accessible_cards(self):
        pile_top_cards_accessible = self.pile_top_cards
        if len(pile_top_cards_accessible) > 2:
            if not cards_same_rank(pile_top_cards_accessible):
                # only outer cards accessible in case of streak
                cards_sorted = sort_cards(pile_top_cards_accessible)
                pile_top_cards_accessible = [cards_sorted[0], cards_sorted[-1]]

        return pile_top_cards_accessible

    def io_options(self, options, player, option_type='throw'):
        assert option_type in ['throw', 'pull']

        #if self.ax is not None:
        #    self.ax.clear()
        #self.ax = visualise_cards(player.cards_in_hand, ax=self.ax)
        #self.ax = visualise_cards(self.pile_top_cards, ax=self.ax, cards_type='pile')

        print(f'\ncurrent hand {player.cards_in_hand}')
        if 'throw' == option_type:
            print(f'pile cards: {self.pile_top_accessible_cards()}')
        option_id = input(f'{option_type}ing:\n{options}')

        return option_id

    def throw_cards_to_pile(self, name):
        player = self.players[name]
        valid_combinations = cards_to_valid_throw_combinations(player.cards_in_hand)
        sorted_combinations, sorted_combinations_sums = sort_card_combos(valid_combinations, descending=True, return_sum_values=True)

        if 'human' == player.agent:
            options = {f'{idx}': option for idx, option in enumerate(sorted_combinations, 1)}
            option_id = self.io_options(options, player, option_type='throw')
            cards_to_throw = options[option_id]
            print(f'you threw: {cards_to_throw}')
        else:
            # ======= temp, highest combinations =======
            cards_to_throw = sorted_combinations[0]
            # =====================

        # updating player hand cards
        player.cards_in_hand = [this_card for this_card in player.cards_in_hand if this_card not in cards_to_throw]


        self.cards_thrown += cards_to_throw
        self.pile_top_cards_this_turn = cards_to_throw
        self.turn_output['throws'] = cards_to_throw

    # TODO: devise better strategies for pulling cards
    def pull_card(self, name):
        player = self.players[name]
        self.turn_player = player

        self.chosen_from_pile_top = None

        if 'human' == player.agent:
            options = {'1': 'deck'} #, '2': 'pile': }
            accessible_cards = self.pile_top_accessible_cards()
            for option_id, card in enumerate(accessible_cards, 2):
                options[f'{option_id}'] = card
            option_id = self.io_options(options, player, option_type='pull')

            if '1' == option_id:
                this_card = self.pull_card_from_deck()
                source = 'deck'
            else:
                this_card = options[option_id]
                source = 'pile'
            print(f'You chose {this_card} from the {source}')

        else:

            # ======== need to devise better strategy ===========
            #pull_card_function = np.random.choice([self.pull_card_from_deck, self.pull_card_from_pile_top])

            card_values = [card_to_value(card) for card in self.pile_top_cards]
            idx_lowest = np.array(card_values).argmin()

            if player.strategy["pile_pull"]["highest_card_value_to_pull"]  >= card_values[idx_lowest]:
                pull_card_function = self.pull_card_from_pile_top
            else:
                pull_card_function = self.pull_card_from_deck

            # overriding previous decision, because the deck is empty ...
            if len(self.round_deck) == 0:
                pull_card_function = self.pull_card_from_pile_top

            this_card = pull_card_function()
            # ==================================================

        self.turn_output['pulls'] = this_card
        player.cards_in_hand.append(this_card)
        player.sum_hand_points()
        player.remove_cards_from_unknown([this_card])

    # TODO: deal with situation where pile is empty (deck only? might cause infinite loop)
    def pull_card_from_deck(self):
        self.turn_output['pull_source'] = 'deck'
        this_card = np.random.choice(self.round_deck, size=1, replace=False)[0]
        self.round_deck.remove(this_card)


        return this_card

    def pull_card_from_pile_top(self):
        self.turn_output['pull_source'] = 'pile'
        n_cards = len(self.pile_top_cards)

        accessible_cards = self.pile_top_cards
        if 1 == n_cards:
            self.chosen_from_pile_top = self.pile_top_cards[0]
            return self.pile_top_cards[0]
        elif 2 == n_cards:
            # both card are accessible
            pass
        else:
            accessible_cards = self.pile_top_accessible_cards()
        # ======== here we will need to introduce strategy of best card to choose ============
        player_cards = self.turn_player.cards_in_hand
        cards_plus_player_collective_points_ = pile_cards_plus_player_collective_hypothetical_points(accessible_cards, player_cards)

        if bool(cards_plus_player_collective_points_):
            this_card = next(iter(cards_plus_player_collective_points_))
        else:
            if len(accessible_cards) == 1:
                this_card = accessible_cards[0]
            else:
                this_card = sorted(accessible_cards, key=lambda card: card_to_value(card), reverse=False)[0]

        # print(self.number, self.turn_number, self.pile_top_cards, accessible_cards, player_cards, cards_plus_player_collective_points_, this_card)


        # =====================================
        self.pile_top_cards = list(set(self.pile_top_cards) - set([this_card]))
        self.chosen_from_pile_top = this_card

        return this_card

    def update_players_knowledge(self, turn_player_name):
        disposed_cards = set(self.pile_top_cards)

        for player_name, player in self.players.items():
            player.remove_cards_from_unknown(disposed_cards)

            if (player_name != turn_player_name) & (turn_player_name in self.players.keys()):
                player.knowledgewise_drop_cards_from_player(turn_player_name, self.pile_top_cards)
                if self.chosen_from_pile_top:
                    player.knowledgewise_assign_card_to_player(turn_player_name, self.chosen_from_pile_top)

    def prob_lowest_hand_two_players(self, name):
        h_i = self.players[name].hand_points
        cards_unknown = self.players[name].unknown_cards
        other_name = list(set(self.players.keys()) - set([name]))[0]
        n_j = len(self.players[other_name].cards_in_hand)

        return calculate_p_hj_gt_hi_n_j_prior(n_j, cards_unknown, h_i=h_i, play_jokers=self.play_jokers,
                                              verbose=self.verbose)

    def prob_lowest_hand(self, name):
        player_i = self.players[name]
        hand_sum_i = player_i.hand_points  # was hand_points

        cards_unknown = player_i.unknown_cards
        cards_unknown_values = [card_to_value(card) for card in cards_unknown]
        #print(cards_unknown_values)

        prob_lowest = 1.
        for name_j, player_j in self.players.items():
            if name_j != name:
                prob_Hi_lt_Hj = self.calculate_prob_Hi_lt_Hj(hand_sum_i, player_j, cards_unknown_values)
                prob_lowest *= prob_Hi_lt_Hj

        if self.verbose >= 3:
            print('~' * 10)
            print(hand_sum_i, player_i.cards_in_hand)
            print(f"The probability for {name} to make a successful Yaniv decleration is: {100. * prob_lowest:0.1f}%")
        return prob_lowest

    def calculate_prob_Hi_lt_Hj(self, hand_sum_i, player_j, cards_unknown_values): #, hand_points, name_other, cards_unknown_values): # was calculate_prob_yaniv_better_than_other

        n_cards =len(player_j.cards_in_hand) # was n_cards_other
        max_value_to_win = self.card_num_to_max_value[n_cards] # was thresh

        #n_cards_other = len(self.players[name_other].cards_in_hand)  # number of cards of other player
        #thresh = self.card_num_2_max_value[n_cards_other]  # maximum value other can have to declare yaniv


        if self.verbose >= 3:
            print('~' * 10)
            print(f"Given {player_j.name} has {n_cards} cards, the max threshold is {max_value_to_win} (i.e, if has above this value, no chance to Assaf)")


        cards_unknown_smaller_than_max_bool = list(map(lambda x: is_smaller_or_equal_binary(x, thresh=max_value_to_win), cards_unknown_values))

        # Calculating the probability that all cards in other player's hand is smaller than the max thresh possible to Yaniv
        prob_all_cards_under_thresh = calculate_prob_all_cards_under_thresh(n_cards, cards_unknown_smaller_than_max_bool, verbose=self.verbose - 2)

        cards_unknown_values_small = [value for value in cards_unknown_values if value <=max_value_to_win]

        # Given all cards are under the thresh -- what is the probability of NOT Assafing the Yaniv declaration?
        prob_hj_gt_hi = calculate_prob_ht_gt_hi(hand_sum_i, cards_unknown_values_small, n_cards) # was prob_above_yaniv_given_all_below_threshold

        prob_hi_lt_jh = (1. - prob_all_cards_under_thresh) +  prob_hj_gt_hi * prob_all_cards_under_thresh  # was prob_yaniv_better_than_other

        #prob_yaniv_better_than_other = (1 - prob_all_cards_under_thresh) + prob_above_yaniv_given_all_below_threshold * prob_all_cards_under_thresh

        if self.verbose >= 3:
            print("p({} cards sum > yaniv| all {} cards <= {} )=%{:0.1f}".format(player_j.name, player_j.name, max_value_to_win,
                                                                                 100. * prob_hj_gt_hi))
            print(
                "Meaning the probability of Successful Yaniv (=NOT being Assafed by {}) is: %{:0.2f}".format(player_j.name,
                                                                                                             prob_hi_lt_jh * 100.))

        return prob_hj_gt_hi


