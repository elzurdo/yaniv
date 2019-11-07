
# TODO: build on this principle to make card_num_to_max_single_value in a dyanmic fashion, i.e, with knowldege of cards thrown out
def card_number_to_max_card_value_to_declare_yaniv(play_jokers=True):
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

    if play_jokers:
        card_num_to_max_single_value[5] = 5  # 5, joker, joker, ace, ace
        card_num_to_max_single_value[4] = 6  # 6, joker, joker, ace
        card_num_to_max_single_value[3] = 7  # 7, joker, joker
        card_num_to_max_single_value[2] = 7  # 7, joker

    return card_num_to_max_single_value
