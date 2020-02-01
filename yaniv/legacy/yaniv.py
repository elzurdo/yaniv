import numpy as np
import pandas as pd

from scipy.stats import hypergeom
from itertools import permutations

MAX_ROUNDS = 400
YANIV_LIMIT = 7  # the value in which one can call Yaniv!


# ========= card related functions =========


def face_to_value(face):
    if face == 'A':
        return 1
    elif face in ['J', 'Q', 'K']:
        return 10
    elif 'oker' in face:  # purposely 'oker1', 'okder2', and not 'joker'
        return 0
    else:
        return int(face)


def card_to_suite(card):
    '''Returns suite of card
    :param card: str
    :return: str. possible values 's' (Spades), 'd' (Diamonds), 'h' (Hearts), 'c' (Clubs) and 'j' (Jokers)
    '''
    return card[0]


def card_to_face(card):
    '''Returns the face of the card in str

    Note that possible values are 'A', '2', '3', ... '10', 'J', 'Q', 'K'
    and for jokers: 'oker1' or 'oker2'

    :param card: str.
    :return: str.
    '''
    return card[1:]


def deck(jokers=True):
    '''Return the deck in dict type

    :param jokers: bool. True: game played with 2 jokers, False: without
    :return: dict. keys are the sting of the card and the values are the card points
    '''
    suits = ['d', 'h', 'c', 's']  # diamonds, hearts, clubs, spades
    faces = ['A'] + list(map(str, range(2, 11))) + ['J', 'Q', 'K']  # Ace, 2-10, Jack, Queen, King

    points = list(range(1, 11)) + [10, 10, 10]  # notice! J, Q, K all give 10 points
    card_to_score = {"{}{}".format(suit, face):
                         points[iface] for iface, face in enumerate(faces) for suit in suits}

    values = list(range(1, 11)) + [11, 12, 13]  # notice! J, Q, K are valued at 11, 12, 13, respectively
    card_to_streak_value = {"{}{}".format(suit, face):
                                values[iface] for iface, face in enumerate(faces) for suit in suits}

    if jokers:
        card_to_score['joker1'] = 0
        card_to_score['joker2'] = 0

        card_to_streak_value['joker1'] = -1  # better option than 0, because do not want to relate to A (1)
        card_to_streak_value['joker2'] = -1

    return card_to_score, card_to_streak_value


card_to_score_all, card_to_streak_value_all = deck(jokers=True)


def cards_to_df(cards):
    df_cards = pd.DataFrame({'face': list(map(card_to_face, cards)),
                             'suit': list(map(card_to_suite, cards)),
                             'value': list(map(lambda x: card_to_score_all[x], cards))
                             },
                            index=cards)

    return df_cards


def cards_df_to_face_counts_df(df_cards):
    df_face_counts = pd.DataFrame(df_cards['face'].value_counts()).rename(columns={'face': 'counts'})
    df_face_counts.index.name = 'face'
    df_face_counts['value'] = df_face_counts.index.map(face_to_value)
    df_face_counts['total'] = df_face_counts['counts'] * df_face_counts['value']

    return df_face_counts


def cards_to_values(cards):
    values = []
    for card in cards:
        values.append(card_to_score_all[card])

    return values


def cards_to_relevant_to_test_streak(initial_cards):
    '''Returns subset list of cards that might be relevant for streak

    We find cards in the dominant suite in hand and pass any available jokers.
    '''
    df = pd.DataFrame(index=initial_cards)

    df['score'] = df.index.map(lambda x: card_to_score_all[x])
    df['face'] = df.index.map(card_to_face)
    df['suite'] = df.index.map(card_to_suite)
    sr_suite_count = df.groupby("suite").size()

    min_suite_required_for_streak = 3
    # for every joker in hand the minimum required is lower
    # so one would naively do:
    # min_suite_required_for_streak -= sr_suite_count['j']
    # but since using one known card and two jokers does not make sense
    # the thresh should be 2
    if 'j' in sr_suite_count.index:
        min_suite_required_for_streak = 2

    if sr_suite_count.max() >= min_suite_required_for_streak:
        suite = sr_suite_count[sr_suite_count == sr_suite_count.max()].index[0]
        # jokers are also relevant
        cards = df[df['suite'].isin([suite, 'j'])].index.tolist()
    else:
        cards = None

    return cards


# TODO: case of jokers
def find_straight_set(initial_cards):
    '''Returning a set of 3 or more in the cards

    Can only have one set streak (since the maximum cards in hand is 5)

    The basis here is fairly simple:
    (1) cards_to_relevant_to_test_streak limits the initial_cards to cards of the same suit with maximum count
    (2) building a series by going through a sorted list of cars in the same suite


    Jokers not dealt with yet because is non trivial:
    (1) Perhaps more useful to use in less dominant suit
    (2) Need to look across at least two other cards
    (3) Jokers are best to put in the middle where one cannot draw, not on the sides.
    '''
    cards = cards_to_relevant_to_test_streak(initial_cards)

    if not cards:
        return None
    #
    sr_ = pd.Series(list(map(lambda x: card_to_streak_value_all[x], cards)), index=cards).sort_values()

    result = None
    if len(sr_) >= 3:
        idx_previous = sr_.index[0]

        series = [idx_previous]
        for i, idx in enumerate(sr_.index[1:]):
            diff = sr_[idx] - sr_[idx_previous]
            if diff == 1:  # this assumes one deck! if more than one deck need to reexamine
                # adding to series
                series.append(idx)

                if len(series) >= 3:
                    # registering series into result
                    result = list(series)
            else:
                if len(series) >= 3:
                    # registering series into result
                    result = list(series)

                # starting new series
                series = [idx]

            idx_previous = idx

    return result


# ============= auxillary functions
def is_smaller_binary(value, thresh=None):
    return int(value <= thresh)


# ========================================

class Player():
    def __init__(self, name, throw_strategy='highest_card', yaniv_strategy='always', seed=None):
        if seed:
            np.random.seed(seed)
        self.name = name

        self.throw_strategy = throw_strategy
        self.pull_strategy = {"highest_card_value_to_pull": np.random.randint(1, 6)}
        self.yaniv_strategy = yaniv_strategy
        self.starts_round = False

    def _hand_points(self):
        '''Calculates the sum of point in a hand
        '''

        self.hand_points = 0
        for card in self.cards_in_hand:
            self.hand_points += card_to_score_all[card]


class Game():
    def __init__(self, player_names, max_score=200, assaf_penalty=30, jokers=True, verbose=1, seed=4):
        '''
        verbose:
        0 - display player names, winner his/her score
        1 - in addition to 0, displays Round information
        2 - in addition to 1, displays probability of a Yaniv call being successful
        3 - in addition to 2, displays how the probability was derived

        :param player_names: list of names
        :param max_score: int. Score which terminates game for player
        :param assaf_penalty: int. The penalty score for an Assaf
        :param jokers: bool. True to use 2 jokers in the game False without.
        :param verbose: int
        :param seed: int
        '''
        self.seed = seed
        self.assaf_penalty = assaf_penalty
        self.jokers = jokers
        self.max_score = max_score
        self.verbose = verbose

        self._players(player_names)
        self.card_to_score, self.card_to_streak_value = deck(jokers=jokers)

    def play(self):
        self.initiate_players_status()

        self.play_game()

    def _seeding(self):
        if self.seed:
            np.random.seed(seed=self.seed)
            self.seed += 1

    def _players(self, player_names):
        '''Given a list of players names creates a list of of Player objects

        :param player_names:
        :return:
        '''

        self.all_players = []
        print('Players and strategies:')

        for name in player_names:
            self._seeding()
            player_ = Player(name, seed=self.seed)
            self.all_players.append(player_)
            print(name)
            print("Highest value card will pick from pile: {}".format(player_.pull_strategy["highest_card_value_to_pull"]))

            print("-" * 10)

    def initiate_players_status(self):
        '''Initiates the status of all the players
        In particular:
        * score - int. starts at 0 points
        * in_play - bool. True: plays current round. False: already lossed the game

        :return: None
        '''
        for iplayer, player in enumerate(self.all_players):
            player.score = 0
            player.in_play = True

            if iplayer == 0:
                player.starts_round = True

    def _round_players(self):
        '''Returns a dictionary of Player objects only of players that have less the max_score.

        This is used to track which players advance to the next round

        :return: dict. keys are player names (str) and values are the corresponding Player objects
        '''
        players = {}

        for player in self.all_players:
            if player.in_play:

                if player.score < self.max_score:
                    players[player.name] = player
                    if self.verbose >= 3:
                        print(player.name, player.score, 'IN')
                else:
                    player.in_play = False
                    if self.verbose >= 2:
                        print(player.name, player.score, 'OUT')
                        print('-' * 20)

        return players

    def play_game(self):
        '''Game play

        :return:
        '''

        # card_num_2_max_value is a mapping from the number of cards in hand to the maximum value of single card in hand
        # that can result in a successful Yaniv. This is useful to calculate heuristics of Yaniv success probabilities.
        card_num_2_max_value = self.card_number_to_max_card_value_to_declare_yaniv()

        # round number counter. The first round is value 1, so setting counter to 0.
        round_number = 0

        # players is a dictionary of players that progress to the next round (they have less than max_score)
        players = self._round_players()

        while len(players) > 1:  # the game terminates when left with one player
            if self.verbose:
                print('=' * 20)
            round_number += 1
            if self.verbose >= 1:
                print('Round: {:,}'.format(round_number))

            # Declaring Round object and playing a round
            # TODO: make card_num_2_max_value dyanmic
            self.round = Round(players, self.card_to_score, assaf_penalty=self.assaf_penalty,
                               card_num_2_max_value=card_num_2_max_value, verbose=self.verbose, seed=self.seed)
            self.round.play()

            if self.seed:
                self.seed += 1
            # ====== player score updating ==============
            for name, player in players.items():
                # hand points go into score.
                # If Yaniv was successful the caller gets 0 points (otherwise the original hand plus the assaf_penalty
                # All other players get hand_points (including if one was an Assafer)
                player.score += player.hand_points

                # Jackpot!
                # If a player hits these luck values their score is reduced
                if player.score == 100:
                    player.score = 50
                    print("Lucky {}! Aggregated 100 points reduced to 50".format(name))
                elif player.score == 200:
                    print("Lucky {}! Aggregated 200 points reduced to 150".format(name))
                    player.score = 150

                if self.verbose:
                    print(player.name, player.hand_points, player.cards_in_hand, player.score)
            # ===========================

            # Round conclusion
            players = self._round_players()  # players for next round

            if round_number > MAX_ROUNDS:
                print('breaking at max rounds: {:,}'.format(MAX_ROUNDS))
                break

        if len(players) == 1:
            winner = players[list(players.keys())[0]]
            print("The winner is: {} with {:,} points".format(winner.name, winner.score))
        else:
            # Case of 0 players left (double-or-more knockout case)
            print("Everybody loses ... ({} players left)".format(len(players)))

    # TODO: build on this principle to make card_num_to_max_single_value in a dyanmic fashion, i.e, with knowldege of cards thrown out
    def card_number_to_max_card_value_to_declare_yaniv(self):
        '''Returns a mapping between the number of cards in the hand to the max possible single card value f
        or successful Yaniv

        E.g, if no jokers are in play, and someone has 5 cards, in order to successfully declare Yaniv,
        the maximum value any single card may have is 3:
        3, ace, ace, ace, ace

        If there are jokers still in play, and someone has 5 cards, in order to in order to successfully declare Yaniv,
        the maximum value any single card may have is 5:
        5, joker, joker, ace, ace
        4 is aso an option:
        4, joker, joker, ace, 2 (or ace),
        as is 3:
        3, joker, joker, 2, 2 (or 3 and ace)

        but 5 is the ultimate max.

        This is useful to determine heuristics for probabilities.

        Here we assume a full deck (i.e, no knowledge of what might have been thrown out).

        :return: dict. keys are int of number of cards in hand. values are int of maximum possible value single card in
        order to declare a successful Yaniv
        '''
        # Evaluating the most extreme values to obtain a Yaniv
        # No assumption about state of deck made (assuming a full deck)

        # making a function for future purposes
        card_num_to_max_single_value = {}
        card_num_to_max_single_value[
            5] = 3  # this means: if player has 5 cards in hand, the max value card for yaniv is 3: 3, ace, ace, ace, ace
        card_num_to_max_single_value[4] = 4  # 4, ace, ace, ace
        card_num_to_max_single_value[3] = 5  # 5, ace, ace
        card_num_to_max_single_value[2] = 6  # 6, ace
        card_num_to_max_single_value[1] = 7  # if player has 1 card in hand, the max value card for yaniv is 7: 7

        if self.jokers:
            card_num_to_max_single_value[5] = 5  # 5, joker, joker, ace, ace
            card_num_to_max_single_value[4] = 6  # 6, joker, joker, ace
            card_num_to_max_single_value[3] = 7  # 7, joker, joker
            card_num_to_max_single_value[2] = 7  # 7, joker

        return card_num_to_max_single_value


class Round():
    def __init__(self, players, card_to_score, card_num_2_max_value=None, assaf_penalty=30, seed=4, verbose=0):
        self.seed = seed
        self.verbose = verbose
        self.assaf_penalty = assaf_penalty
        self.card_to_score = card_to_score
        self.card_num_2_max_value = card_num_2_max_value
        self.cards_thrown = []

        self.players = players

        self.meta = {} # meta data for logging

    def play(self):
        # round starts with a full deck
        self.round_deck = dict(self.card_to_score)

        self.distribute_cards()

        self.play_round()

    def _seeding(self):
        if self.seed:
            np.random.seed(seed=self.seed)
            self.seed += 1

    # TODO: make tidier, hopefully without using pandas
    def _player_order(self):
        '''Determining the player order

        Basic idea:
        One of the players should have a starts_round=True (based on default or if they won the previous round,
        by means of Yaniv-ing or Assaf-ing).
        Then all the rest of the players are ordered by their indexes (insilico analogy of by "seating order")

        :return:
        '''
        starting_player = None
        for name, player in self.players.items():
            if player.starts_round == True:

                if starting_player:
                    print('Error! {} and {} both have starting status'.format(starting_player, name))
                    sys.exit(1)

                starting_player = name

        if self.verbose:
            print('Player starting the round: {}'.format(starting_player))

        # TODO: this works but quite cumbersome. There is probably a cleaner way to code this
        l_current_player_names = list(self.players.keys())
        sr_current_player_names = pd.Series(l_current_player_names)
        idx_starting_player = sr_current_player_names[(sr_current_player_names == starting_player)].index
        idx_starting_player = idx_starting_player[0]

        l_player_names_order = [starting_player]
        l_player_names_order += l_current_player_names[idx_starting_player + 1:] + l_current_player_names[
                                                                                   0: idx_starting_player]

        player_order = {}
        for name in l_player_names_order:
            player_order[name] = self.players[name]

        return player_order

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
            player._hand_points()

            # Deleting selected cards from the round's deck
            for card in player.cards_in_hand:
                del self.round_deck[card]

    def _get_highest_turn_number(self):
        if not self.meta:
            return 0
        else:
            return max(self.meta.keys())

    def _log_meta_data(self, player):
        turn_number = self._get_highest_turn_number() + 1

        self.meta[turn_number] = {}
        self.meta[turn_number]["name"] = player.name
        self.meta[turn_number]["cards_start"] = list(player.cards_in_hand)
        self.meta[turn_number]["hand_points_start"] = player.hand_points
        # TODO game_points Move to higher hierarchy in json when established for Game level.
        self.meta[turn_number]["game_points"] = player.score

    def play_round(self, player_order=None):
        # flag to finish round. True: keep playing, False: end of round.
        yaniv_declared = False

        if not player_order:
            player_order = self._player_order()

        for name, player in player_order.items():
            self._log_meta_data(player)
            if not yaniv_declared:
                if player.hand_points <= YANIV_LIMIT:
                    # name considers declearing yaniv based on their Player.yaniv_strategy probability of success
                    yaniv_declared = self.decide_declare_yaniv(name)
                if yaniv_declared:
                    # round ends
                    self.round_summary(name)
                    return None
                else:
                    self.throw_card(name)
                    self.pull_card(name)

        if not yaniv_declared:
            # at this stage we did a full "circle around the table",
            # but did not conclude with a Yaniv declaration. We will go for another round
            # perhaps there is a better way of doing this loop.
            self.play_round(player_order=player_order)

    # Todo: add probabalistic approaches Issue #3
    def decide_declare_yaniv(self, name):
        '''Returns Player boolean decision if to declare Yaniv

        The decision is based on yaniv_strategy
        (and when we introduce probabilistic approaches also on their prediction of success)

        :param name: str. Name of Player considering declaring
        :return: bool. True: declare Yaniv (end round), False: continue the round play
        '''
        if self.verbose > 1:
            self.prob_lowest_hand(name)

        player = self.players[name]
        if 'always' == player.yaniv_strategy:
            return True

    # TODO: think more critically about the Assafer that gets to start the next round. (closest neighbor? lowest hand?)
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
        turn_number = self._get_highest_turn_number()  # turn_number used in self.meta below
        self.meta[turn_number]["declared_yaniv"] = True

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

            self.meta[turn_number]["assafed"] = {}
            self.meta[turn_number]["assafed"]["by"] = assafer_name
            self.meta[turn_number]["assafed"]["assafer_points"] = self.players[assafer_name].hand_points
            # The Yaniv declarer is penalised by assaf_penalty because of incorrect call
            self.players[name_yaniv].hand_points += self.assaf_penalty
            # The Assafer gets to start the next round
            self.players[assafer_name].starts_round = True
        else:
            # Yaniv was successful so the caller does not get points
            self.players[name_yaniv].hand_points = 0
            # ... and gets to start the next round.
            yaniv_player.starts_round = True

        self.meta[turn_number]["hand_points_end"] = player.hand_points

    def cards_to_relevant_to_test_streak(initial_cards):
        '''Returns subset list of cards that might be relevant for streak

        We find cards in the dominant suite in hand and pass any available jokers.
        '''
        df = pd.DataFrame(index=initial_cards)

        df['score'] = df.index.map(lambda x: yaniv.card_to_score_all[x])
        df['face'] = df.index.map(yaniv.card_to_face)
        df['suite'] = df.index.map(yaniv.card_to_suite)
        sr_suite_count = df.groupby("suite").size()

        min_suite_required_for_streak = 3
        # for every joker in hand the minimum required is lower
        # so one would naively do:
        # min_suite_required_for_streak -= sr_suite_count['j']
        # but since using one known card and two jokers does not make sense
        # the thresh should be 2
        if 'j' in sr_suite_count.index:
            min_suite_required_for_streak = 2

        if sr_suite_count.max() >= min_suite_required_for_streak:
            suite = sr_suite_count[sr_suite_count == sr_suite_count.max()].index[0]
            # jokers are also relevant
            cards = df[df['suite'].isin([suite, 'j'])].index.tolist()
        else:
            cards = None

        return cards

    def find_straight_set(initial_cards):
        '''Returning a set of 3 or more in the cards

        Can only have one (since the maximum cards in hand is 5)
        '''
        cards = cards_to_relevant_to_test_streak(initial_cards)

        if not cards:
            return None
        #
        sr_ = pd.Series(list(map(lambda x: yaniv.card_to_streak_value_all[x], cards)), index=cards).sort_values()

        print(sr_)
        result = None
        if len(sr_) >= 3:
            idx_previous = sr_.index[0]

            series = [idx_previous]
            for i, idx in enumerate(sr_.index[1:]):
                diff = sr_[idx] - sr_[idx_previous]
                if diff == 1:  # this assumes one deck! if more than one deck need to reexamine
                    # adding to series
                    series.append(idx)

                    if len(series) >= 3:
                        # registering series into result
                        result = list(series)
                else:
                    if len(series) >= 3:
                        # registering series into result
                        result = list(series)

                    # starting new series
                    series = [idx]

                idx_previous = idx

        return result

    def throw_card(self, name):
        '''name throws out card(s)
        Needs to figure out between Same Face (default) or Streak.

        :param name:
        :return:
        '''
        player = self.players[name]

        df_cards = cards_to_df(player.cards_in_hand)

        # ======= Same Face Option =========
        df_face_counts = cards_df_to_face_counts_df(df_cards)
        df_face_counts_max = df_face_counts[df_face_counts['total'] == df_face_counts['total'].max()]

        df_cards_same_face = df_cards[df_cards['face'] == df_face_counts_max.index[0]]
        points_same_face = df_cards_same_face['value'].sum()

        # ======== Streak Option ============
        points_streak = 0  # initial setting

        cards_in_streak = find_straight_set(player.cards_in_hand)
        if cards_in_streak:
            points_streak = df_cards.loc[cards_in_streak, 'value'].sum()

        if points_streak >= points_same_face:
            if self.verbose >= 1:
                print("{} throwing streak {}, {}>={}, {}".format(name, cards_in_streak, points_streak, points_same_face,
                                                                 list(set(df_cards.index) - set(cards_in_streak)))
                      )
            cards_thrown = cards_in_streak
        else:
            cards_thrown = df_cards_same_face.index.tolist()

        turn_number = self._get_highest_turn_number()  # turn_number used in self.meta below
        self.meta[turn_number]["throw_out"] = cards_thrown

        self.cards_thrown += cards_thrown

        self.cards_to_choose_from = cards_thrown

        player.df_cards = df_cards.drop(cards_thrown)
        # TODO: use only df_cards and depricate cards_in_hand
        player.cards_in_hand = player.df_cards.index.tolist()

    def pull_card(self, name):

        # currently only pulling from deck
        player = self.players[name]

        pull_source = None
        if len(self.round_deck) > 0:
            self._seeding()

            highest_value_to_choose = player.pull_strategy["highest_card_value_to_pull"]

            sr_cards_to_choose_from = pd.Series(list(map(lambda x: card_to_score_all[x], self.cards_to_choose_from)),
                                                self.cards_to_choose_from)

            if sr_cards_to_choose_from.min() <= highest_value_to_choose:  # picking up from throw pile
                chosen_card = [sr_cards_to_choose_from.sort_values().index[0]]
                pull_source = "pile"
                # print('pull from pile: {}'.format(chosen_card))
            else:  # picking up from deck
                chosen_card = np.random.choice(list(self.round_deck.keys()), size=1, replace=False)
                # print('deck pile: {}, instead of {}'.format(chosen_card, sr_cards_to_choose_from.min()))
                pull_source = "deck"

            if "deck" == pull_source:
                del self.round_deck[chosen_card[0]]
            player.cards_in_hand = np.append(player.cards_in_hand, chosen_card)
        else:
            if self.verbose >= 2:
                print("Deck is empty")

        turn_number = self._get_highest_turn_number()  # turn_number used in self.meta below
        if pull_source:
            self.meta[turn_number]["pulled"] = chosen_card[0]
            self.meta[turn_number]["pulled_source"] = pull_source

        player._hand_points()
        self.meta[turn_number]["hand_points_end"] = player.hand_points

    def _calculate_stats___OLD(self, name):
        cards_player = list(self.players[name].cards_in_hand)
        cards_unknown = list(set(self.card_to_score.keys()) - set(self.cards_thrown))
        cards_unknown = list(set(cards_unknown) - set(cards_player))

        cards_unknown_values = []
        for card in cards_unknown:
            cards_unknown_values.append(self.card_to_score[card])
        cards_unknown_values = pd.Series(cards_unknown_values).value_counts().sort_index()

        for name_other, player in self.players.items():
            if name_other != name:
                n_cards_other = len(player.cards_in_hand)
                if self.verbose >= 2:
                    print("{} cards: {}".format(n_cards_other, name_other))

    def name_2_cards_unknown(self, name):
        cards_player = self.players[name].cards_in_hand
        cards_unknown = list(set(self.card_to_score.keys()) - set(self.cards_thrown))
        cards_unknown = list(set(cards_unknown) - set(cards_player))

        return cards_unknown

    def prob_lowest_hand(self, name):
        hand_points = self.players[name].hand_points  # michal

        cards_unknown = self.name_2_cards_unknown(name)
        cards_unknown_values = cards_to_values(cards_unknown)

        prob_lowest = 1.
        for name_other, player in self.players.items():
            if name_other != name:
                prob_better_than_other = self.calculate_prob_yaniv_better_than_other(hand_points, name_other,
                                                                                     cards_unknown_values)

                prob_lowest *= prob_better_than_other

        if self.verbose >= 2:
            print('~' * 10)
            print("The probability for {} to make a successful Yaniv decleration is: {:0.1f}%".format(name,
                                                                                                      100. * prob_lowest))

    def calculate_prob_yaniv_better_than_other(self, hand_points, name_other, cards_unknown_values):
        n_cards_other = len(self.players[name_other].cards_in_hand)  # number of cards of other player
        thresh = self.card_num_2_max_value[n_cards_other]  # maximum value other can have to declare yaniv

        if self.verbose >= 3:
            print('~' * 10)
            print(
                "Given {} has {} cards, the max threshold is {} (i.e, if has above this value, no chance to Assaf)".format(
                    name_other, n_cards_other, thresh))

        cards_unknown_smaller_than_thresh_bool = list(
            map(lambda x: is_smaller_binary(x, thresh=thresh), cards_unknown_values))

        # Calculating the probability that all cards in other player's hand is smaller than the max thresh possible to Yaniv
        prob_all_cards_under_thresh = self.calculate_prob_all_cards_under_thresh(n_cards_other,
                                                                                 cards_unknown_smaller_than_thresh_bool)

        # Given all cards are under the thresh -- what is the probability of NOT Assafing the Yaniv declaration?

        cards_unknown_values_small = []
        for card in cards_unknown_values:
            if card <= thresh:
                cards_unknown_values_small.append(card)

        prob_above_yaniv_given_all_below_threshold = self.calculate_prob_above_yaniv_given_all_below_thresh(hand_points,
                                                                                                            cards_unknown_values_small,
                                                                                                            n_cards_other)

        prob_yaniv_better_than_other = (1 - prob_all_cards_under_thresh) + prob_above_yaniv_given_all_below_threshold * prob_all_cards_under_thresh

        if self.verbose >= 3:
            print("p({} cards sum > yaniv| all {} cards <= {} )=%{:0.1f}".format(name_other, name_other, thresh,
                                                                                 100. * prob_above_yaniv_given_all_below_threshold))
            print(
                "Meaning the probability of Successful Yaniv (=NOT being Assafed by {}) is: %{:0.2f}".format(name_other,
                                                                                                             prob_yaniv_better_than_other * 100.))

        return prob_yaniv_better_than_other

    def calculate_prob_all_cards_under_thresh(self, n_cards, smaller_than_thresh_bool):
        n = n_cards  # of n cards in player's hand
        k = n  # we want to verify that all k=n are lower than the thresh

        N = len(smaller_than_thresh_bool)  # from a pool of total N cards
        K = sum(smaller_than_thresh_bool)  # where K of them are known to be less than thresh

        prob_all_cards_under_thresh = hypergeom.pmf(k, N, K, n)

        if self.verbose >= 3:
            print("Of a total of N={} unknown cards K={} card are below or equal to thresh".format(N, K))
            print("The probability that all k={} of n={} cards are below thresh is: %{:0.1f}".format(k, n,
                                                                                                     prob_all_cards_under_thresh * 100.))

        return prob_all_cards_under_thresh

    def calculate_prob_above_yaniv_given_all_below_thresh(self, hand_points, card_values, n_cards):

        l_permutation_sums = []
        other_above_yaniv_counter = 0

        for permutation in permutations(card_values, n_cards):
            sum_ = sum(permutation)
            l_permutation_sums.append(sum_)

            if sum_ > hand_points:  # > self.yaniv_points:
                other_above_yaniv_counter += 1

        prob_above_yaniv_given_below_threshold = other_above_yaniv_counter * 1. / len(l_permutation_sums)

        return prob_above_yaniv_given_below_threshold
