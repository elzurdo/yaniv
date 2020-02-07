import game

players = ['Albert','Roland', 'Amos','Claude']
#players = {'Eyal': 'human','Roland':'bot', 'Amos':'bot','Claude':'bot'}
#players = ['Albert', 'Roland']
end_game_score = 200

play_jokers = True
do_stats = False
seed = 1

verbose = 0
game_ = game.Game(players, seed=seed, verbose=verbose, end_game_score=end_game_score,
                  do_stats=do_stats, play_jokers=play_jokers)
game_output = game_.play()
