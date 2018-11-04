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

game = yaniv.Game(['john','paul', 'ringo','george'], seed=1, verbose=0)
game.play()
```

To see stats, set `verbose=2`.
