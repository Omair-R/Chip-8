import numpy as np
import pygame
import sys
from cpu import CPU
import gpu
from key_map import key_map

pygame.init()

screen = pygame.display.set_mode((gpu.width, gpu.height))

pygame.display.set_caption("Chip-8")

screen.fill(gpu.Colors.black)

clock = pygame.time.Clock()

cpu = CPU(screen)
cpu.load_rom_to_ram(
    r'C:\Personal_Files\Projects\Github\Chip-8\Chip-8-RL\roms\Paddles.ch8')

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
