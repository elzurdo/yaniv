# Yaniv
Yaniv is a highly addictive backpacker's card game. 

![](https://pbs.twimg.com/profile_images/1265531457/kaka_400x400.png)

For rules see the [Wikipedia page](https://en.wikipedia.org/wiki/Yaniv_(card_game)).


The main objectives for this ongoing project:     
* Calculate the game statistics
* Train bots with the end goal to learn strategy
* Make Interactive


In the current format `yaniv` can run a game with `N` players from an overlooker's point of view.  
One nice feature added is calculating the statistics of a Yaniv declaration from the declarer's point of view. 
It is currently correct for two players, and requires more work for `N>2`.

To run a game go to the `yaniv/yaniv` directory and do:
```bash
python run.py
```

This will yield the game results, round for round with a mention of the Yaniv success statistics, e.g:

```
Albert (bot) strategy: picks if min pile top min value  <= 3
----------

Roland (bot) strategy: picks if min pile top min value  <= 5
----------

Amos (bot) strategy: picks if min pile top min value  <= 5
----------

Claude (bot) strategy: picks if min pile top min value  <= 5
----------
Deck of 54 cards
['W☻', 'W☺', 'A♦', 'A♥', 'A♣', 'A♠', '2♦', '2♥', '2♣', '2♠', '3♦', '3♥', '3♣', '3♠', '4♦', '4♥', '4♣', '4♠', '5♦', '5♥', '5♣', '5♠', '6♦', '6♥', '6♣', '6♠', '7♦', '7♥', '7♣', '7♠', '8♦', '8♥', '8♣', '8♠', '9♦', '9♥', '9♣', '9♠', '10♦', '10♥', '10♣', '10♠', 'J♦', 'J♥', 'J♣', 'J♠', 'Q♦', 'Q♥', 'Q♣', 'Q♠', 'K♦', 'K♥', 'K♣', 'K♠']
Round 7:
	Lucky Claude! Aggregated 100 points reduced to 50
Claude is the winner with 194 points
```

For more output information change `verbose` (values 0 to 3).


# Statistics
Within `stats` you will find code that can calculates at a given round:  
Given a player has declared victory (yaniv!), what is the probability 
that they truly have the lowest hand.

For this you will need to update within `run.py`:
```python
do_stats = True
verbose = 1
players = ['Albert','Roland'] # currently limited to two players
```

Example output of a round 

```bash
Round: 23
Player starting the round: Roland
n_j=4, h_i=7, n_jokers=1 yields
t_nj=6
N: 36
['3c', '8c', '3h', 'Kh', 'Qd', 'Ac', '10d', '5d', 'Ks', 'Jd', 'Qc', '3s', 'Qs', 'Kd', '5h', '8h', '5s', 'Wb', '4d', '10c', '2d', 'Ad', '9h', 'As', '4h', 'Jc', 'Kc', '7d', '7c', '4c', '2c', '2h', '6d', '6c', '3d', '10s']
n: 17
['3c', '3h', 'Ac', '5d', '3s', '5h', '5s', 'Wb', '4d', '2d', 'Ad', 'As', '4h', '4c', '2c', '2h', '3d']
total: 2380
successes: 2179
P(h_j>h_i|U)=0.916
N=36, K=17, n_j=4, yields:
p(U)=0.040
P(h_j>h_i=7|n_j, cards)=0.997
~~~~~~~~~~
Round Conclusion
Albert declared Yaniv with 7
Albert 0 ['2♠', 'W☻', '4♠', 'A♥'] 185
Roland 15 ['2♥', '5♥', '3♠', '5♣'] 138
```

From here we learn that:
* Roland started the round,
* Albert declared yaniv
  * with a hand of `h_i=7`
  * was unaware of `N=36` cards 
  * (of which Roland had `n_j=4` cards in hand).
 * Given those facts from Albert's POV the probability of the decleration being successful is `P(h_j>h_i=7|n_j, cards)=0.997`
 * Roland has `h_j=15` from cards ['2♥', '5♥', '3♠', '5♣']
 * Resulting in Albert not getting any points where Roland got 15 (all points are bad).


# Human play

There is some preliminary human playing mode, but still requires more work. 

For a quick peek set one of the players to `human`, like this:
```python
players = {'<Your Name>': 'human','Roland':'bot', 'Amos':'bot','Claude':'bot'}
```


# Requirements 
python 3

* numpy 
* scipy  
* pandas

# Testing
Run bash command 
```bash
py.test
```
(or `pyteset`)


# Credits 
Image credit: Barak Edry from his excellent [App Store game of Yaniv](https://itunes.apple.com/gb/app/yaniv/id397614908?mt=8)
