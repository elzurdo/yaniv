# -*- coding: utf-8 -*-

import numpy as np
import sys

from cards import (define_deck, card_to_pretty,
                   cards_to_valid_throw_combinations,
                   sort_card_combos,
                   card_to_value,
                   cards_same_rank,
                   sort_cards,
                   )

from stats import (card_number_to_max_card_value_to_declare_yaniv,
                   calculate_prob_all_cards_under_thresh,
                   calculate_prob_ht_gt_hi
                   )


ASSAF_PENALTY = 30
END_GAME_SCORE = 200
MAX_ROUNDS = 100
MAX_TURNS = 100
YANIV_LIMIT = 7  # the value in which one can call Yaniv!

def is_smaller_binary(value, thresh=None):
    return int(value <= thresh)

# TODO: design and implement different throw_strategy
# TODO: design and implement different yaniv_strategy ('always' (i.e, 7), 'only below 4', 'by statistics')
# TODO: design and implement different pile_pull_strategy
# TODO: figure out and implement how to change strategy as game progresses (so to maximise utility)
class Player():
    def __init__(self, name, throw_strategy='highest_card', yaniv_strategy='not_always', prob_success_thresh=0.8, seed=None):
        assert yaniv_strategy in ['always', 'not_always']
        if seed:
            np.random.seed(seed)
        self.name = name

        self.throw_strategy = throw_strategy
        self.pile_pull_strategy = {"highest_card_value_to_pull": np.random.randint(1, 6)}
        self.yaniv_strategy = yaniv_strategy
        self.prob_successful_yaniv_thresh = prob_success_thresh
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
    def __init__(self, player_names, end_game_score=None, assaf_penalty=None, play_jokers=True, verbose=1, seed=None):
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

        self.generate_players(player_names)
        self.deck = define_deck(play_jokers=play_jokers)
        print(self.deck)
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
    def generate_players(self, player_names):
        '''Given a list of players names creates a list of of Player objects

        :param player_names:
        :return:
        '''

        self.all_players = []
        print('generating players and their playing strategies:')

        for idx, name in enumerate(player_names):
            self._seeding()
            player = Player(name, seed=self.seed)
            player.id = idx
            self.all_players.append(player)
            print(name)
            print('Pile pick strategy:\npicks if min pile top min value  <= {}'.format(player.pile_pull_strategy['highest_card_value_to_pull']))
            #print("Highest value card will pick from pile: {}".format(player_.pull_strategy["highest_card_value_to_pull"]))

            print("-" * 10)

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
                               card_num_to_max_value=card_num_to_max_value, verbose=self.verbose, seed=self.seed, round_output=self.game_output[round_number])
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
                    print("Lucky {}! Aggregated 100 points reduced to 50".format(name))
                    self.game_output[round_number]['end']['lucky_50'] = 1
                elif player.game_score == 200:
                    print("Lucky {}! Aggregated 200 points reduced to 150".format(name))
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
            print("Reached max rounds. Left standing: {}".format(', '.join([player for player in players])))
        else:
            # Case of 0 players left (double-or-more knockout case)
            print("Everybody loses ... ({} players left)".format(len(players)))

class Round():
    def __init__(self, players, deck, card_num_to_max_value=None, assaf_penalty=30, seed=4, verbose=0, do_stats=True, round_output=None):
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

    def play(self):
        # round starts with a full deck
        self.round_deck = list(self.deck)

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
        for name, player in self.players.items():
            player.add_cards_to_unknown(full_deck)

            self._seeding()
            # assigning randomised selected cards to Player
            player.cards_in_hand = np.random.choice(self.round_deck, size=num_cards, replace=False)
            self.round_output['start'][name] = list(player.cards_in_hand)
            # calculating points in a hand of Player
            player.sum_hand_points()
            player.remove_cards_from_unknown(player.cards_in_hand)

            # Deleting selected cards from the round's deck
            for card in player.cards_in_hand:
                self.round_deck.remove(card)

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
        if self.do_stats:
            prob_successful_yaniv = self.prob_lowest_hand(name)

        player = self.players[name]
        if 'always' == player.yaniv_strategy:
            return True
        else:
            if player.prob_successful_yaniv_thresh >= prob_successful_yaniv:
                return True

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
            self.round_output[turn] = {}
            self.turn_output = self.round_output[turn]
            self.turn_output['name'] = name
            self.turn_output['pile_top_accessible'] = self.pile_top_accessible_cards()
            self.turn_output[name] = list(player.cards_in_hand) # this might be redundant information
            # self._log_meta_data(player)
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

    def pile_top_accessible_cards(self):
        pile_top_cards_accessible = self.pile_top_cards
        if len(pile_top_cards_accessible) > 2:
            if not cards_same_rank(pile_top_cards_accessible):
                # only outer cards accessible in case of streak
                cards_sorted = sort_cards(pile_top_cards_accessible)
                pile_top_cards_accessible = [cards_sorted[0], cards_sorted[-1]]

        return pile_top_cards_accessible

    def throw_cards_to_pile(self, name):
        player = self.players[name]
        valid_combinations = cards_to_valid_throw_combinations(player.cards_in_hand)
        sorted_combinations, sorted_combinations_sums = sort_card_combos(valid_combinations, descending=True, return_sum_values=True)

        # ======= temp, highest combinations =======
        cards_to_throw = sorted_combinations[0]
        # =====================

        # updating player hand cards
        player.cards_in_hand = [this_card for this_card in player.cards_in_hand if this_card not in cards_to_throw]


        self.cards_thrown += cards_to_throw
        self.pile_top_cards_this_turn = cards_to_throw
        self.turn_output['throws'] = cards_to_throw

    def pull_card(self, name):
        player = self.players[name]

        self.chosen_from_pile_top = None

        # ======== need to devise better strategy ===========
        #pull_card_function = np.random.choice([self.pull_card_from_deck, self.pull_card_from_pile_top])

        card_values = [card_to_value(card) for card in self.pile_top_cards]
        idx_lowest = np.array(card_values).argmin()

        if player.pile_pull_strategy['highest_card_value_to_pull'] >= card_values[idx_lowest]:
            pull_card_function = self.pull_card_from_pile_top
        else:
            pull_card_function = self.pull_card_from_deck

        this_card = pull_card_function()
        # ==================================================

        self.turn_output['pulls'] = this_card
        player.cards_in_hand.append(this_card)
        player.sum_hand_points()
        player.remove_cards_from_unknown([this_card])

    # TODO: deal with situation where pile is empty (deck only? might cause infinite loop)
    def pull_card_from_deck(self):
        self.turn_output['pull_source'] = 'deck'
        if len(self.round_deck) == 0:
            print('Error: The deck is empty, game should end at this stage!')
            DECK_IS_EMPTY_SHOULD_END_ROUND_HERE
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
        this_card = np.random.choice(accessible_cards)
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


        cards_unknown_smaller_than_max_bool = list(map(lambda x: is_smaller_binary(x, thresh=max_value_to_win), cards_unknown_values))

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


