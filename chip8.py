import argparse
from ast import arg
import numpy as np
import pygame
import sys
from cpu import CPU
import gpu
from key_map import key_map

parser = argparse.ArgumentParser(description="Chip-8")

parser.add_argument("rom", type=str, help="The path to the rom file")
parser.add_argument(
    "-d",
    "--delay",
    default=1,
    type=int,
    help="the delay time the cpu takes before performing another operation.")
parser.add_argument(
    "-s",
    "--scale",
    default=15,
    type=int,
    help="the delay time the cpu takes before performing another operation.")

args = parser.parse_args()

gpu.scale = args.scale
gpu.delaytime = args.delay
pygame.init()

screen = pygame.display.set_mode(
    (gpu.width * gpu.scale, gpu.height * gpu.scale))

pygame.display.set_caption("Chip-8")

screen.fill(gpu.Colors.black)

clock = pygame.time.Clock()

cpu = CPU(screen)
cpu.load_rom_to_ram(args.rom)

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.KEYDOWN:
            key = pygame.key.get_pressed()
            for k, v in key_map.items():
                if key[v]:
                    cpu.keys[k] = 1

        if event.type == pygame.KEYUP:
            cpu.keys = np.zeros(16, dtype=np.bool_)

    pygame.time.delay(gpu.delaytime)
    cpu.cpu_cycle()
