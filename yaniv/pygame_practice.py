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
from __future__ import print_function, division
import pygame

# Initialize Pygame.
pygame.init()
# Set size of pygame window.
screen = pygame.display.set_mode((640, 480))
# Create empty pygame surface.

screen_size = screen.get_size()
factor_ = 1. #0.5
background = pygame.Surface((factor_ * screen_size[0], factor_* screen_size[1]))
# Fill the background white color.

background.fill((255, 255, 255))
# Convert Surface object to make blitting faster.
background = background.convert()
# Copy background to screen (position (0, 0) is upper left corner).
#screen.blit(background, (factor_/2. * screen_size[0], factor_/2. * screen_size[1]))
screen.blit(background, (0,0))

# create a rectangular surface for the ball
ballsurface = pygame.Surface((50,50))
# draw blue filled circle on ball surface
pygame.draw.circle(ballsurface, (0,0,255), (25,25),27)
screen.blit(ballsurface, (50,50))

card_file = '/Users/ekazin/Work/data_science/projects/playing-cards-assets/png/8_of_diamonds.png'
mypicture = pygame.image.load(card_file)

# Create Pygame clock object.
clock = pygame.time.Clock()

mainloop = True
# Desired framerate in frames per second. Try out other values.
FPS = 30
# How many seconds the "game" is played.
playtime = 0.0

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


    # Print framerate and playtime in titlebar.
    text = "FPS: {0:.2f}   Playtime: {1:.2f}".format(clock.get_fps(), playtime)
    pygame.display.set_caption(text)

    # Update Pygame display.
    pygame.display.flip()

# Finish Pygame.
pygame.quit()

# At the very last:
print("This game was played for {0:.2f} seconds".format(playtime))
