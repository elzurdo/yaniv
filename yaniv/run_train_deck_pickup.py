from rl_utils import train_deck_pickup_dnn_to_player, create_deck_pickup_dnn, run_ann_policy, show_rounds_results

n_environments = 50  # number of environments to train on
n_iterations = 200   # number of iteration to train on
n_inputs = 5  # [n_deck, card_value_lowest, turn_number, n_cards_i, ncards_j]


seed = 1
players_names = ['Albert', 'ANN']
fixed_pickup_strategy = 4
n_rounds = 200
plot = False
# --- results for dummy model ----
dummy_model = create_deck_pickup_dnn(n_inputs, seed=seed)

df_results_dummy = run_ann_policy(players_names, dummy_model, seed=seed, n_rounds=n_rounds,
                             fixed_pickup_strategy=fixed_pickup_strategy)

print('======== Dummy Model Results ========')
show_rounds_results(df_results_dummy, plot=plot)

# --- training DNN to learn fixed_pickup_strategy ---
deck_pickup_dnn, l_losses = train_deck_pickup_dnn_to_player(n_iterations, n_environments, n_inputs,
                                                            seed=seed, basic_strategy=fixed_pickup_strategy)
# --- results for trained model ----
df_results_trained = run_ann_policy(players_names, deck_pickup_dnn, seed=seed, n_rounds=n_rounds,
                             fixed_pickup_strategy=fixed_pickup_strategy)

print('\n======== Trained Model Results ========')
show_rounds_results(df_results_trained, plot=plot)