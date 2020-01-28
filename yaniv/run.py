import game

#players = ['Albert','Roland', 'Amos','Claude']
players = {'Eyal': 'human','Roland':'bot', 'Amos':'bot','Claude':'bot'}
end_game_score = 200

do_stats = False

verbose = 0
game_ = game.Game(players, seed=3, verbose=verbose, end_game_score=end_game_score, do_stats=do_stats)
game_output = game_.play()