# -*- coding: utf-8 -*-

import numpy as np
import sys

from stats import card_number_to_max_card_value_to_declare_yaniv


ASSAF_PENALTY = 30
END_GAME_SCORE = 200
MAX_ROUNDS = 10
YANIV_LIMIT = 7  # the value in which one can call Yaniv!

SUITE_CHAR_TO_SYMBOL = {'d': '♦', 'h': '♥', 'c': '♣', 's': '♠'}


# TODO: check if redundant with card_to_score
def rank_to_value(rank):
    if rank == 'A':
        return 1
    elif rank in ['J', 'Q', 'K']:
        return 10
    elif rank == 'W':
        return 0
    else:
        return int(rank)


def card_to_pretty(card):
    return ''.join([SUITE_CHAR_TO_SYMBOL[char] if char in SUITE_CHAR_TO_SYMBOL.keys() else char for char in card])

def card_to_suite(card):
    '''Returns suite of card
    :param card: str
    :return: str. possible values 's' (Spades), 'd' (Diamonds), 'h' (Hearts), 'c' (Clubs) and 'w' (Jokers)
    '''
    if card[0] in [1, 2]:
        return 'w'

    return card[0]


def card_to_rank(card):
    '''Returns the face of the card in str

    Note that possible values are 'A', '2', '3', ... '10', 'J', 'Q', 'K' and 'w' for joker

    :param card: str.
    :return: str.
    '''
    return card[1:]

def define_deck(play_jokers=True):
    '''Return the deck in dict type

    :param play_jokers: bool. True: game played with 2 jokers, False: without
    :return: dict. keys are the sting of the card and the values are the card points
    '''
    suits = ['d', 'h', 'c', 's']   #  ['♦', '♥', '♣', '♠']  # diamonds, hearts, clubs, spades
    ranks = ['A'] + list(map(str, range(2, 11))) + ['J', 'Q', 'K']  # Ace, 2-10, Jack, Queen, King

    points = list(range(1, 11)) + [10, 10, 10]  # notice! J, Q, K all give 10 points
    card_to_score = {"{}{}".format(suit, face):
                         points[iface] for iface, face in enumerate(ranks) for suit in suits}

    values = list(range(1, 11)) + [11, 12, 13]  # notice! J, Q, K are valued at 11, 12, 13, respectively
    card_to_streak_value = {"{}{}".format(suit, face):
                                values[iface] for iface, face in enumerate(ranks) for suit in suits}

    if play_jokers:
        card_to_score['1W'] = 0
        card_to_score['2W'] = 0

        card_to_streak_value['1W'] = -1  # better option than 0, because A has value 1
        card_to_streak_value['2W'] = -1

    return card_to_score, card_to_streak_value


# TODO: design and implement different throw_strategy
# TODO: design and implement different yaniv_strategy ('always' (i.e, 7), 'only below 4', 'by statistics')
# TODO: design and implement different pile_pull_strategy
# TODO: figure out and implement how to change strategy as game progresses (so to maximise utility)
class Player():
    def __init__(self, name, throw_strategy='highest_card', yaniv_strategy='always', seed=None):
        if seed:
            np.random.seed(seed)
        self.name = name

        self.throw_strategy = throw_strategy
        self.pile_pull_strategy = {"highest_card_value_to_pull": np.random.randint(1, 6)}
        self.yaniv_strategy = yaniv_strategy
        self.starts_round = False

    def sum_hand_points(self, card_to_score):
        '''Calculates the sum of point in a hand
        '''

        self.hand_points = 0
        for card in self.cards_in_hand:
            self.hand_points += card_to_score[card]

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

        if assaf_penalty:
            self.end_game_score = end_game_score
        else:
            self.end_game_score = END_GAME_SCORE

        self.play_jokers = play_jokers
        self.verbose = verbose

        self.generate_players(player_names)
        self.card_to_score, self.card_to_streak_value = define_deck(play_jokers=play_jokers)

    def play(self):
        self.initiate_players_status()

        self.play_game()

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
            if self.verbose >= 1:
                print('Round: {:,}'.format(round_number))

            # Declaring Round object and playing a round
            # TODO: make card_num_2_max_value dyanmic


            self.round = Round(players, self.card_to_score, assaf_penalty=self.assaf_penalty,
                               card_num_to_max_value=card_num_to_max_value, verbose=self.verbose, seed=self.seed)
            self.round.play()

            #"""

            if self.seed:
                self.seed += 1
            # ====== player score updating ==============
            for name, player in players.items():
                # hand points go into score.
                # If Yaniv was successful the caller gets 0 points (otherwise the original hand plus the assaf_penalty
                # All other players get hand_points (including if one was an Assafer)
                player.game_score += player.hand_points

                # Jackpot!
                # If a player hits these luck values their score is reduced
                if player.game_score == 100:
                    player.game_score = 50
                    print("Lucky {}! Aggregated 100 points reduced to 50".format(name))
                elif player.game_score == 200:
                    print("Lucky {}! Aggregated 200 points reduced to 150".format(name))
                    player.game_score = 150

                if self.verbose:
                    print(player.name, player.hand_points, player.cards_in_hand, player.game_score)
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
    def __init__(self, players, card_to_score, card_num_to_max_value=None, assaf_penalty=30, seed=4, verbose=0, do_stats=False):
        self.seed = seed
        self.verbose = verbose
        self.assaf_penalty = assaf_penalty
        self.card_to_score = card_to_score
        self.card_num_to_max_value = card_num_to_max_value
        self.cards_thrown = []
        self.do_stats = do_stats

        self.players = players

        self.meta = {}  # meta data for logging

    def play(self):
        # round starts with a full deck
        self.round_deck = self.card_to_score.copy()

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

        for name, player in self.players.items():
            self._seeding()
            # assigning randomised selected cards to Player
            player.cards_in_hand = np.random.choice(list(self.round_deck.keys()), size=num_cards, replace=False)
            # calculating points in a hand of Player
            player.sum_hand_points(self.card_to_score)

            # Deleting selected cards from the round's deck
            for card in player.cards_in_hand:
                del self.round_deck[card]

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
    def decide_declare_yaniv(self, name):
        '''Returns Player boolean decision if to declare Yaniv

        The decision is based on yaniv_strategy
        (and when we introduce probabilistic approaches also on their prediction of success)

        :param name: str. Name of Player considering declaring
        :return: bool. True: declare Yaniv (end round), False: continue the round play
        '''
        if (self.verbose > 1) and self.do_stats:
            self.prob_lowest_hand(name)

        player = self.players[name]
        if 'always' == player.yaniv_strategy:
            return True
        
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

        """
        if self.collect_meta:
            self.meta[turn_number]["hand_points_end"] = player.hand_points
        """

    def play_round(self, players_ordered=None):
        # flag to finish round. True: keep playing, False: end of round.
        yaniv_declared = False

        if not players_ordered:
            players_ordered = self.get_player_order()

        # TODO: Consider verbosity !!!
        print('playing order: ', ', '.join(list(players_ordered.keys())))


        for name, player in players_ordered.items():
            # self._log_meta_data(player)
            if not yaniv_declared:
                if player.hand_points <= YANIV_LIMIT:
                    # name considers declearing yaniv based on their Player.yaniv_strategy probability of success

                    # ------ return to this ------
                    yaniv_declared = self.decide_declare_yaniv(name)
                # ----------temporary ---------
                yaniv_declared = np.random.choice([True, False], size=1, p=[0.4, 0.6])[0]
                # -----------------------------
                if yaniv_declared:
                    # round ends
                    self.round_summary(name)
                    return None
                """ !!! continue from here !!!
                else:
                    self.throw_card(name)
                    self.pull_card(name)
                """

        """
        if not yaniv_declared:
            # at this stage we did a full "circle around the table",
            # but did not conclude with a Yaniv declaration. We will go for another round
            # perhaps there is a better way of doing this loop.
            self.play_round(players_ordered=players_ordered)
        """


