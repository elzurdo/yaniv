# Yaniv
Yaniv is a highly addictive backpacker's card game. 

![](https://pbs.twimg.com/profile_images/1265531457/kaka_400x400.png)

For rules see the [Wikipedia page](https://en.wikipedia.org/wiki/Yaniv_(card_game)).


The main objectives for this ongoing project:     
(1) Calculate the game statistics (present)    
(2) Write an algorithm that will beat a human (future)     
(3) Interactive Graphics (future)     

In the current format `yaniv` can run a game with `N` players from an overlooker's point of view.  
One nice feature added is calculating the statistics of a Yaniv declaration being successful. 
It is currently corret for two players, and for above, for the sake of simplicity, it (incorrectly) assumes independence of the cards of the potential Assafers. 

To run a game:
```python3
import yaniv

players = ['John','Paul', 'Ringo','George']

verbose = 2
game = yaniv.Game(players, seed=1, verbose=verbose)
game.play()
```

This will yield the game results, round for round with a mention of the Yaniv success statistics, e.g:

```
Players:
John
Paul
Ringo
George
====================
Round: 1
Player starting the round: John
~~~~~~~~~~
The probability for Paul to make a successful Yaniv decleration is: 59.5%
~~~~~~~~~~
Round Conclusion
Paul declared Yaniv with 7
John 12 ['cA' 'dA' 'cJ'] 12
Paul 0 ['c2' 'c3' 'd2'] 0
Ringo 10 ['s2' 'sA' 'hA' 'c6'] 10
George 12 ['h2' 'joker2' 'joker1' 'sJ'] 12
====================
Round: 2
Player starting the round: Paul
~~~~~~~~~~
The probability for Paul to make a successful Yaniv decleration is: 76.0%
~~~~~~~~~~
Round Conclusion
Paul declared Yaniv with 7
John 15 ['s4' 'dA' 'hQ'] 27
Paul 0 ['c2' 'h3' 'h2'] 0
Ringo 10 ['joker2' 'd2' 'cA' 'hA' 'h6'] 20
George 13 ['sA' 'h4' 'c5' 'joker1' 's3'] 25

.
.
.

Round: 23
Player starting the round: George
~~~~~~~~~~
The probability for George to make a successful Yaniv decleration is: 98.5%
~~~~~~~~~~
Round Conclusion
George declared Yaniv with 6
John 20 ['c6' 'h4' 'hK'] 215
George 0 ['d3' 'hA' 'h2'] 194
John 215 OUT
--------------------
The winner is: George with 194 points

```

For more or less information change verbose, where `verbose=0` is minimal information and `verbose=3` is maximum.

# Requirements 
python 3

* numpy 
* scipy  
* pandas


# Credits 
Image credit: Barak Edry from his excellent [App Store game of Yaniv](https://itunes.apple.com/gb/app/yaniv/id397614908?mt=8)
