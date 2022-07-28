import numpy as np
import pygame
import sys
from pygame import Color
import typing

pygame.init()


class Colors:
    black = Color(0, 0, 0)
    white = Color(255, 255, 255)


FPS: int = 60
UPDATE_TIME: int = round(1 / 60 * 1000)

scale: int = 15

width: int = 64 * scale
height: int = 32 * scale


def scale_rect(pos: typing.List[int]) -> typing.List[int]:
    return [i * scale for i in pos] + [scale, scale]


def draw_from_binary(binary_array: typing.List[int]):
    for y in range(height // scale):
        for x in range(width // scale):
            if binary_array[(width // scale) * y + x] == 1:
                pygame.draw.rect(screen, Colors.white, scale_rect([x, y]))


screen = pygame.display.set_mode((width, height))

pygame.display.set_caption("Chip-8")

clock = pygame.time.Clock()
