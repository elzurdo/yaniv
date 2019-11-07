ASSAF_PENALTY = 30
END_GAME_SCORE = 200


# TODO: check if redundant with card_to_score
def rank_to_value(rank):
    if rank == 'A':
        return 1
    elif rank in ['J', 'Q', 'K']:
        return 10
    elif rank == 'W':  # purposely 'oker1', 'okder2', and not 'joker'
        return 0
    else:
        return int(rank)


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
    suits = ['♦', '♥', '♣', '♠']  # diamonds, hearts, clubs, spades
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
        :param max_score: int. Score which terminates game for player
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
        self.card_to_score, self.card_to_streak_value = define_deck(jokers=play_jokers)

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

        for name in player_names:
            self._seeding()
            player = Player(name, seed=self.seed)
            self.all_players.append(player)
            print(name)
            print('Pile pick strategy:\npicks if min pile card <= {}'.format(player_.pull_strategy['highest_card_value_to_pull'])
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