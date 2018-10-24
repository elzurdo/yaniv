import numpy as np


class Player():
    def __init__(self, name, throw_strategy='highest_card', yaniv_strategy='always'):
        self.name = name

        self.throw_strategy = throw_strategy
        self.yaniv_strategy = yaniv_strategy
        self.starts_round = False

    def _hand_points(self):
        # The value of points of current hand

        self.hand_points = 0
        for card in self.cards_in_hand:
            self.hand_points += all_card2scores[card]


# start game:
# * round 1


def deck(jokers=True):
    suits = ['d', 'h', 'c', 's']  # diamonds, clubs, hearts, spades

    values = list(range(1, 11)) + [10, 10, 10]
    names = ['A'] + list(map(str, range(2, 11))) + ['J', 'Q', 'K']  # Ace, 2-10, Jack, Queen, King

    card2score = {"{}{}".format(suit, name):
                      values[iname] for iname, name in enumerate(names) for suit in suits}

    if jokers:
        card2score['joker1'] = 0
        card2score['joker2'] = 0

    return card2score

all_card2scores = deck(jokers=True)


MAX_ROUNDS = 400
YANIV_LIMIT = 7  # the value in which one can call Yaniv!

class Game():
    def __init__(self, player_names, max_score=200, assaf_penalty=30, jokers=True, verbose=1, seed=4):
        self.seed = seed
        self._players(player_names)
        self.assaf_penalty = assaf_penalty
        self.card2score = deck(jokers=jokers)

        #self.all_players = players
        self.max_score = max_score

        self.verbose = verbose

    def play(self):
        self.initial_scores()

        self.play_game()

    def _players(self, player_names):

        self.all_players = []

        for name in player_names:
            self.all_players.append(Player(name))
            print(name)

    def initial_scores(self):
        for iplayer, player in enumerate(self.all_players):
            player.score = 0
            player.in_play = True

            if iplayer == 0:
                player.starts_round = True

    def _round_players(self):
        players = {}

        for player in self.all_players:
            if player.in_play:

                if player.score < self.max_score:
                    players[player.name] = player
                    if self.verbose > 1:
                        print(player.name, player.score, 'IN')
                else:
                    player.in_play = False
                    if self.verbose > 0:
                        print(player.name, player.score, 'OUT')
                        print('-' * 20)

        return players

    def play_game(self):

        round_number = 0

        players = self._round_players()

        while len(players) > 1:
            if self.verbose:
                print('=' * 20)
            round_number += 1
            if self.verbose > 0:
                print('Round: {:,}'.format(round_number))

            round_ = Round(players, self.card2score, assaf_penalty=self.assaf_penalty, verbose=self.verbose, seed=self.seed)
            if self.seed:
                self.seed += 1
            # ====== DELETE/COMMENT-OUT: TEST PURPOSES ======
            for name, player in players.items():
                player.score += player.hand_points

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
            print("Everybody loses ... ({} players left)".format(len(players)))


class Round():
    def __init__(self, players, card2score, assaf_penalty=30, seed=4, verbose=0):
        self.seed = seed
        self.verbose = verbose
        self.assaf_penalty = assaf_penalty
        self.card2score = card2score
        self.cards_thrown = []

        self.players = players

        self.round_deck = dict(self.card2score)

        # print(len(self.round_deck))
        self.distribute_cards()

        self.play_round()

    def _seeding(self):
        if self.seed:
            np.random.seed(seed=self.seed)
            self.seed += 1

    def _player_order(self):
        import pandas as pd
        starting_player = None
        for name, player in self.players.items():
            # print(player.name, player.starts_round)
            if player.starts_round == True:

                if starting_player:
                    print('Error! {} and {} both have starting status'.format(starting_player, name))
                    sys.exit(1)

                starting_player = name

        if self.verbose:
            print('Starting player: {}'.format(starting_player))

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

        for name, player in self.players.items():
            self._seeding()
            player.cards_in_hand = np.random.choice(list(self.round_deck.keys()), size=num_cards, replace=False)
            player._hand_points()

            for card in player.cards_in_hand:
                del self.round_deck[card]
            # print(player.name, player.hand_points)

    def play_round(self, player_order=None):

        yaniv_declared = False

        if not player_order:
            player_order = self._player_order()
        for name, player in player_order.items():  # self.players.items():
            # print(name, player.starts_round)
            if not yaniv_declared:
                if player.hand_points <= YANIV_LIMIT:
                    yaniv_declared = self.decide_declare_yaniv(name)
                if yaniv_declared:
                    self.round_summary(name)
                else:
                    self.throw_card(name)
                    self.pull_card(name)

        if not yaniv_declared:
            self.play_round(player_order=player_order)

    def decide_declare_yaniv(self, name):
        player = self.players[name]
        if 'always' == player.yaniv_strategy:
            return True

    def round_summary(self, name_yaniv):
        assafed = False
        yaniv_player = self.players[name_yaniv]

        if self.verbose:
            print('{} declared Yaniv with {}'.format(name_yaniv, yaniv_player.hand_points))

        assafers = []
        for name, player in self.players.items():
            player.starts_round = False  # zero-ing out those that start round

            if name != name_yaniv:
                # print(name, player.hand_points)
                if player.hand_points <= yaniv_player.hand_points:
                    assafed = True
                    assafers.append(name)

        if assafed:
            assafer_name = assafers[0]
            if self.verbose:
                print('ASSAF!')
                print('by: {} (hand of {})'.format(assafers[0], self.players[assafer_name].hand_points))

            self.players[name_yaniv].hand_points += self.assaf_penalty
            self.players[assafer_name].starts_round = True
        else:
            self.players[name_yaniv].hand_points = 0  # Yaniv player does not get points
            yaniv_player.starts_round = True

    def throw_card(self, name):
        player = self.players[name]
        # self.throw_strategy = 'highest_card'
        # print(player.throw_strategy, player.cards_in_hand)

        if 'highest_card' == player.throw_strategy:
            cards_in_hand = {}
            for card in player.cards_in_hand:
                cards_in_hand[card] = self.card2score[card]

            # ========= temp script: figure out how to do this without pandas (find card with highest value)
            import pandas as pd
            cards_thrown = pd.Series(cards_in_hand).sort_values().tail(1).index[0]

            if not isinstance((cards_thrown), list):
                cards_thrown = [cards_thrown]

            # print(pd.Series(cards_in_hand[card])) #.sort_values().tail(1))
            self.cards_thrown += cards_thrown
            for card_thrown in cards_thrown:
                del cards_in_hand[card_thrown]

            player.cards_in_hand = list(cards_in_hand)

    def pull_card(self, name):
        # currently only pulling from deck
        player = self.players[name]

        if len(self.round_deck) > 0:
            self._seeding()
            chosen_card = np.random.choice(list(self.round_deck.keys()), size=1, replace=False)

            del self.round_deck[chosen_card[0]]
            player.cards_in_hand = np.append(player.cards_in_hand, chosen_card)

        player._hand_points()
        # print(name, player.cards_in_hand, player.hand_points)



