from collections import OrderedDict

import numpy as np
import pandas as pd

class Game():
    """
    Initiates and tracks progress of a game of Yaniv
    """
    def __init__(self, nplayers=3, jokers=True, end=200, verbose=0):
        # general definitions
        self.verbose = verbose

        if self.verbose:
            print "==== New Game ===="

        # may be varied by user
        self.nplayers = nplayers
        self.jokers = jokers
        self.end = end

        # determined by game rules
        self.round_number = 1
        self.nplayers_active = int(self.nplayers)

        if self.verbose > 1:
            print "{} players\njokers: {}\nend: {}".format(self.nplayers, self.jokers, self.end)

        if self.verbose:
            print "==== Player Setup ===="

        self._player_setup()
        if self.verbose:
            print "==== Card Setup ===="
        self._cards()
        if self.verbose > 1:
            print "Number of cards: {}".format(len(self.full_deck))

    def _player_setup(self):
        self.players = OrderedDict()
        for idx in range(self.nplayers):
            self.players[idx] = Player(idx)

            print self.players[idx].name

    def _cards(self):
        # after a threshold of people we should have more than one
        # standard deck. this would mean that full_deck indexes
        # get a number to distinguish which deck each card came from
        suits = ['d', 'h', 'c', 's']  # diamonds, clubs, hearts, spades

        values = range(1, 11) + [10, 10, 10]
        # Ace, 2-10, Jack, Queen, King
        names = ['A'] + map(str, range(2, 11)) + ['J', 'Q', 'K']

        deck = {"{}{}".format(suit, name):
                    values[iname] for iname, name in enumerate(names) for suit in suits}

        if self.jokers:
            deck['Rj'] = 0
            deck['Bj'] = 0

        full_deck = pd.Series(deck.values(), index=deck.keys())

        self.full_deck = full_deck

    def simulate_rounds(self, restart=True):

        if restart:
            self = Game(self.nplayers, jokers=self.jokers, verbose=self.verbose)
            # yields self.round_number = 1
        else:
            self.round_number += 1

        print 'here'
        self.round_ = Round(self)
        print 'now here'
        for idx, _ in self.players.iteritems():
            print idx, self.full_deck.loc[self.players[idx].hand].sum()


    def hand2sum(hand):
        None


l_names = ['John', 'Paul', 'Ringo', 'George']

class Player():
    """
    Initiates and tracks progress of a player
    """

    def __init__(self, idx):
        # constant in game
        self.idx = idx
        self.name = l_names[idx]

        # varies in game
        self.score = 0
        self.active = True  # all players are initially active

        # varies in round
        self.turn = False  # True: turn to pick up or declare, False: Not turn
        self.hand = None

    def hand_total():
        print None

    def check_active(thresh):
        # True: score<thresh, False: score >=tresh
        if self.score < thresh:
            self.active = True
        else:
            self.active = False


class Round():
    """
    Initiates and tracks progress of a round
    """
    def __init__(self, game, nCards_player=5, verbose=0):
        self.verbose = verbose
        self.nCards_player = nCards_player

        #self.players = players

        # counting the number of active players in round
        self.N_active = 0
        for idx in range(len(game.players)):
            if game.players[idx].active:
                self.N_active += 1

        if self.verbose > 1:
            print "Number of players: {}".format(self.N_active)
            l_names = [player.name for idx, player in game.players.iteritems()]
            print ", ".join(l_names)

        self.distribute_cards(game)

    def distribute_cards(self, game):

        if self.verbose:
            print 'Distributing cards'

        for idx, _ in game.players.iteritems():
            game.players[idx].hand = np.random.choice(game.full_deck.index.tolist(), self.nCards_player, replace=False)
            print idx, game.players[idx].hand