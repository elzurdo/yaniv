# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
002_display_fps.py

Open a Pygame window and display framerate.
Program terminates by pressing the ESCAPE-Key.

works with python2.7 and python3.4

URL    : http://thepythongamebook.com/en:part2:pygame:step002
Author : horst.jens@spielend-programmieren.at
License: GPL, see http://www.gnu.org/licenses/gpl.html
"""

# the next line is only needed for python2.x and not necessary for python3.x
#from __future__ import print_function, division
import pygame

CARD_WIDTH, CARD_HEIGHT = 222, 323
MAX_CARDS = 5

def display_card(card, screen, location=(100, 100), scale_frac=1., hidden=False):
    card_file = '/Users/ekazin/Work/data_science/projects/playing-cards-assets/png/8_of_diamonds.png'
    card_picture = pygame.image.load(card_file)
    card_picture = card_picture.convert_alpha()

    #111 / 162
    scale_original = card_picture.get_size()
    scale_ = (int(scale_original[0] * scale_frac), int(scale_original[1] * scale_frac))
    card_picture = pygame.transform.scale(card_picture, scale_)

    card_background = card_picture.copy()
    if hidden:
        card_background.fill((128, 0, 128))
        screen.blit(card_background, location)
    else:
        card_background.fill((255, 255, 255))
        screen.blit(card_background, location)
        screen.blit(card_picture, location)


size_factor = 2.
screen_size = ( int(640 * size_factor), int(480 * size_factor))

# Initialize Pygame.
pygame.init()
# Set size of pygame window.
screen = pygame.display.set_mode(screen_size)
# Create empty pygame surface.

# ---- BackGround ----
screen_size = screen.get_size()
print(screen_size)
factor_ = 1.  # 0.5
background = pygame.Surface((factor_ * screen_size[0], factor_* screen_size[1]))
# Fill the background white color.
poker_green = (75, 95, 59)
background.fill(poker_green) # ((0, 255, 255))
# Convert Surface object to make blitting faster.
background = background.convert()
# Copy background to screen (position (0, 0) is upper left corner).
#screen.blit(background, (factor_/2. * screen_size[0], factor_/2. * screen_size[1]))
screen.blit(background, (0,0))

scale_frac = 0.7
card_width = CARD_WIDTH  * scale_frac
# ======= POV Cards =========
bottom = screen_size[1] * 0.75

cards = ['dummy'] * 5

ncards = len(cards)
left_most = card_width / 2 + (MAX_CARDS - ncards) * CARD_WIDTH / 2


for idx, card in enumerate(cards):
    x_location = left_most + (card_width * 1.02) * idx
    display_card(card, screen, location=(x_location, bottom), scale_frac=scale_frac, hidden=False)

# ======= Opponent's Cards ========
bottom = screen_size[1] * 0.05

cards = ['dummy'] * 5

ncards = len(cards)
left_most = card_width / 2 + (MAX_CARDS - ncards) * CARD_WIDTH / 2


for idx, card in enumerate(cards):
    x_location = left_most + (card_width * 1.02) * idx
    display_card(card, screen, location=(x_location, bottom), scale_frac=scale_frac, hidden=True)


# Create Pygame clock object.
clock = pygame.time.Clock()

mainloop = True
# Desired framerate in frames per second. Try out other values.
FPS = 30
# How many seconds the "game" is played.
playtime = 0.0

mouse = [(0,0), 0, 0, 0, 0, 0, 0] #(pos, b1,b2,b3,b4,b5,b6)

while mainloop:
    # Do not go faster than this framerate.
    milliseconds = clock.tick(FPS)
    playtime += milliseconds / 1000.0

    for event in pygame.event.get():
        # User presses QUIT-button.
        if event.type == pygame.QUIT:
            mainloop = False
        elif event.type == pygame.KEYDOWN:
            # User presses ESCAPE-Key
            if event.key == pygame.K_ESCAPE:
                mainloop = False
            elif event.key == pygame.K_z:
                print('I love Zurda!')
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse[event.button] = 1
            mouse[0] = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse[event.button] = 0
            mouse[0] = event.pos
        elif event.type == pygame.MOUSEMOTION:
            mouse[0] = event.pos

    # Print framerate and playtime in titlebar.
    text = "FPS: {0:.2f}   Playtime: {1:.2f}".format(clock.get_fps(), playtime)
    pygame.display.set_caption(text)

    # Update Pygame display.
    pygame.display.flip()

# Finish Pygame.
pygame.quit()

# At the very last:
print("This game was played for {0:.2f} seconds".format(playtime))
