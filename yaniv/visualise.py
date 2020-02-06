import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from cards import card_to_pretty, pile_top_accessible_cards, card_to_color
from stats import calculate_p_hj_gt_hi_n_j_prior

CARD_WIDTH, CARD_HEIGHT = 1, 1/0.62 * 0.88 * 0.8 #0.62 , 0.88
XLIM, YLIM = 6, 4
MAX_CARDS = 5

FIGSIZE = (10, 10)


def visualise_cards(cards, ax=None, cards_type='hand', show_spines=False):
    """

    :param cards:
    :param ax:
    :param cards_type:
    :param show_spines:
    :return:

    Example usage
    cards_hand = ['As', '7d', 'Kc', 'Wa', 'Wb']
    cards_pile = ['2s', '3s', '4s']

    ax = visualise_cards(cards_hand, ax=None, cards_type='hand', show_spines=False)
    visualise_cards(cards_pile, ax=ax, cards_type='pile', show_spines=False)
    visualise_cards(4, ax=ax, cards_type='opponent', show_spines=False)
    """
    assert cards_type in ['hand', 'pile', 'deck', 'opponent']

    show_card_values = True
    if 'hand' == cards_type:
        bottom = 0.1
    elif 'pile' == cards_type:
        bottom = YLIM / 2 - CARD_HEIGHT / 2
        cards_accessible = pile_top_accessible_cards(cards)
    elif 'opponent' == cards_type:
        bottom = YLIM / 2 + CARD_HEIGHT / 2 + 0.1
        assert isinstance(cards, int)
        cards = [None] * cards
        show_card_values = False


    hatch = None
    if not show_card_values:
        hatch = '/'


    if ax is None:
        fig, ax = plt.subplots(1, figsize=FIGSIZE)

    ncards = len(cards)
    left_most = CARD_WIDTH / 2 + (MAX_CARDS - ncards) * CARD_WIDTH / 2

    left = left_most
    dx, dy = 0.2, -0.1
    card_margin = 0.02
    for card in cards:
        alpha = 1
        if 'pile' == cards_type:
            if card not in cards_accessible:
                alpha = 0.3

        ax.add_patch(Rectangle((left, bottom), CARD_WIDTH, CARD_HEIGHT, fill=False, alpha=alpha, hatch=hatch))


        if show_card_values:
            card_pretty = card_to_pretty(card)
            if 'W' in card_pretty:
                card_pretty = card_pretty[-1]

            ax.annotate(card_pretty,
                        ((left + (left + CARD_WIDTH)) / 2 - dx, (bottom + CARD_HEIGHT / 2.2)),
                        rotation=0,
                        fontsize=30, alpha=alpha, color=card_to_color(card))

        left += CARD_WIDTH + card_margin

    plt.xlim(0., XLIM)
    plt.ylim(0., YLIM)

    if not show_spines:
        spines = ['right', 'top', 'left', 'bottom']
        [ax.spines[spine].set_visible(False) for spine in spines]
        ax.tick_params(bottom=False, left=False)
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(plt.NullFormatter())


    return ax



def add_stats(n_j, cards_unknown, ax, h_i=None, play_jokers=True, verbose=False):
    prob_success = calculate_p_hj_gt_hi_n_j_prior(n_j, cards_unknown, h_i=h_i, play_jokers=play_jokers, verbose=verbose)
    ax.set_title(f'p(success=True)={prob_success:0.3f}', fontsize=16)


