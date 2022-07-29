from dataclasses import dataclass
from unicodedata import decimal
import numpy as np
from fonts import FONTS
import pygame
from gpu import draw_from_np_binary


@dataclass(frozen=True)
class Opcode:

    def __init__(self) -> None:
        self.opcode: np.ushort
        self.NNN: np.ushort
        self.NN: np.ushort
        self.N: np.ubyte
        self.x: np.ubyte
        self.y: np.ubyte

    @classmethod
    def adapt(cls, opcode: np.ushort):
        cls.opcode = opcode
        cls.NNN = opcode & 0x0FFF
        cls.NN = opcode & 0x00FF
        cls.N = opcode & 0x000F
        cls.x = (opcode & 0x0F00) >> 8
        cls.y = (opcode & 0x00F0) >> 4
        return cls


class CPU:

    FIRST_ADDRESS_MEMORY = np.ushort(0x200)
    FONTS_ADDRESS_MEMORY = np.ubyte(0x50)
    WIDTH = np.ubyte(64)
    HEIGHT = np.ubyte(32)

    def __init__(self):
        self.v = np.zeros(16, dtype=np.ubyte)
        self.i: np.ushort = 0
        self.stack = np.zeros(64, dtype=np.ushort)
        self.sp: np.ubyte = 0
        self.dt: np.ubyte = 0
        self.st: np.ubyte = 0
        self.frame_buffer = np.zeros([self.WIDTH, self.HEIGHT], dtype=np.bool_)
        self.pc: np.ushort = self.FIRST_ADDRESS_MEMORY
        self.ram = np.zeros(4096, dtype=np.ubyte)
        self.keys = np.zeros(16, dtype=np.bool_)
        self.ram[self.FONTS_ADDRESS_MEMORY:self.FONTS_ADDRESS_MEMORY +
                 FONTS.shape[0]] = FONTS

    def load_rom_to_ram(self, path: str) -> None:
        with open(path, "rb") as file:
            buffer = file.read()
        buffer_np = np.array(list(buffer), dtype=np.ubyte)
        self.ram[self.pc:self.pc + buffer_np.shape[0]] = buffer_np

    def cpu_cycle(self, screen):
        opcode_temp = (self.ram[self.pc] << 0x8) | self.ram[self.pc + 1]
        opcode = Opcode.adapt(opcode_temp)

        self.screen = screen  #fix later

        self.pc += 2

        self.instruction_look_up(opcode)

        if self.dt > 0:
            self.dt -= 1

        if self.st > 0:
            self.st -= 1

    def instruction_look_up(self, opcode: Opcode):

        zero_look_up_dict = {
            0x0: self.op_cls,
            0xE: self.op_ret,
        }

        Arthicmatics_look_up_dict = {
            0x0: self.op_ld_vx_vy,
            0x1: self.op_or_vx_vy,
            0x2: self.op_and_vx_vy,
            0x3: self.op_xor_vx_vy,
            0x4: self.op_add_vx_vy,
            0x5: self.op_sub_vx_vy,
            0x6: self.op_shr_vx_vy,
            0x7: self.op_subn_vx_vy,
            0xE: self.op_shl_vx_vy,
        }

        FE_look_up_dict = {
            0xA1: self.op_sknp_vx,
            0x9E: self.op_skp_vx,
            0x07: self.op_ld_vx_dt,
            0x0A: self.op_ld_vx_k,
            0x15: self.op_ld_dt_vx,
            0x18: self.op_ld_st_vx,
            0x1E: self.op_add_i_vx,
            0x29: self.op_ld_f_vx,
            0x33: self.op_ld_b_vx,
            0x55: self.op_ld_i_vx,
            0x65: self.op_ld_vx_i,
        }

        def zero_lookup(_):
            zero_look_up_dict[opcode.N](opcode)

        def Arthicmatics_lookup(_):
            Arthicmatics_look_up_dict[opcode.N](opcode)

        def FE_lookup(_):
            FE_look_up_dict[opcode.NN](opcode)

        general_look_up_dict = {
            0x0: zero_lookup,
            0x1: self.op_jp_addr,
            0x2: self.op_call_addr,
            0x3: self.op_se_vx_byte,
            0x4: self.op_sne_vx_byte,
            0x5: self.op_se_vx_vy,
            0x6: self.op_ld_vx_byte,
            0x7: self.op_add_vx_byte,
            0x8: Arthicmatics_lookup,
            0x9: self.op_sne_vx_vy,
            0xA: self.op_ld_i_addr,
            0xB: self.op_jp_v0_addr,
            0xC: self.op_rnd_vx_byte,
            0xD: self.op_drw_vx_vy_nibble,
            0xE: FE_lookup,
            0xF: FE_lookup
        }

        index = (opcode.opcode & 0xF000) >> 12
        general_look_up_dict[index](opcode)

    def op_cls(self, opcode: Opcode):
        # 00E0
        self.frame_buffer = np.zeros([self.WIDTH, self.HEIGHT], dtype=np.bool_)

    def op_ret(self, opcode: Opcode):
        # 00EE
        self.sp -= 1
        self.pc = self.stack[self.sp]

    def op_jp_addr(self, opcode: Opcode):
        # 1nnn
        self.pc = opcode.NNN

    def op_call_addr(self, opcode: Opcode):
        # 2nnn
        self.stack[self.sp] = self.pc
        self.sp += 1
        self.pc = opcode.NNN

    def op_se_vx_byte(self, opcode: Opcode):
        # 3xkk
        if self.v[opcode.x] == opcode.NN:
            self.pc += 2

    def op_sne_vx_byte(self, opcode: Opcode):
        # 4xkk
        if self.v[opcode.x] != opcode.NN:
            self.pc += 2

    def op_se_vx_vy(self, opcode: Opcode):
        # 5xy0
        if self.v[opcode.x] == self.v[opcode.y]:
            self.pc += 2

    def op_ld_vx_byte(self, opcode: Opcode):
        # 6xkk
        self.v[opcode.x] = opcode.NN

    def op_add_vx_byte(self, opcode: Opcode):
        # 7xkk
        self.v[opcode.x] += opcode.NN

    def op_ld_vx_vy(self, opcode: Opcode):
        # 8xy0
        self.v[opcode.x] = self.v[opcode.y]

    def op_or_vx_vy(self, opcode: Opcode):
        # 8xy1
        self.v[opcode.x] |= self.v[opcode.y]

    def op_and_vx_vy(self, opcode: Opcode):
        # 8xy2
        self.v[opcode.x] &= self.v[opcode.y]

    def op_xor_vx_vy(self, opcode: Opcode):
        # 8xy3
        self.v[opcode.x] ^= self.v[opcode.y]

    def op_add_vx_vy(self, opcode: Opcode):
        # 8xy4
        vx = np.ushort(self.v[opcode.x]) + np.ushort(self.v[opcode.y])

        self.v[0xF] = 0

        if vx > 255:
            self.v[0xF] = 1

        self.v[opcode.x] = vx & 0xFF

    def op_sub_vx_vy(self, opcode: Opcode):
        # 8xy5
        self.v[0xF] = 0

        if self.v[opcode.x] > self.v[opcode.y]:
            self.v[0xF] = 1

        self.v[opcode.x] -= self.v[opcode.y]

    def op_shr_vx_vy(self, opcode: Opcode):
        # 8xy6
        self.v[0xF] = self.v[opcode.x] & 0x1
        self.v[opcode.x] >>= 1

    def op_subn_vx_vy(self, opcode: Opcode):
        # 8xy7
        self.v[0xF] = 0

        if self.v[opcode.y] > self.v[opcode.x]:
            self.v[0xF] = 1

        self.v[opcode.x] = self.v[opcode.y] - self.v[opcode.x]

    def op_shl_vx_vy(self, opcode: Opcode):
        # 8xyE
        self.v[0xF] = (self.v[opcode.x] & 0x80) >> 7
        self.v[opcode.x] <<= 1

    def op_sne_vx_vy(self, opcode: Opcode):
        # 9xy0
        if self.v[opcode.x] != self.v[opcode.y]:
            self.pc += 2

    def op_ld_i_addr(self, opcode: Opcode):
        # Annn
        self.i = opcode.NNN

    def op_jp_v0_addr(self, opcode: Opcode):
        # Bnnn
        self.pc = opcode.NNN + self.v[0]

    def op_rnd_vx_byte(self, opcode: Opcode):
        # Cxkk
        self.v[opcode.x] = np.random.randint(255) & opcode.NN

    def op_drw_vx_vy_nibble(self, opcode: Opcode):
        # Dxyn
        wrapped_pos_x = self.v[opcode.x]
        wrapped_pos_y = self.v[opcode.y]
        self.v[0xF] = 0

        for j in range(opcode.N):
            sprite_byte = self.ram[self.i + j]
            for i in range(8):
                sprite_bit = sprite_byte & (0x80 >> i)
                pos_x = (wrapped_pos_x + i) % self.WIDTH
                pos_y = (wrapped_pos_y + j) % self.HEIGHT
                current_bit = self.frame_buffer[pos_x, pos_y]
                if current_bit == 1 and sprite_bit:
                    self.v[0xF] = 1
                    self.frame_buffer[pos_x, pos_y] = 0
                elif current_bit == 0 and sprite_bit:
                    self.frame_buffer[pos_x, pos_y] = 1

        draw_from_np_binary(self.frame_buffer, self.screen)
        pygame.display.update()
        # test this later

    def op_skp_vx(self, opcode: Opcode):
        # Ex9E
        current_key = self.v[opcode.x]
        if self.keys[current_key] == 1:
            self.pc += 2

    def op_sknp_vx(self, opcode: Opcode):
        # ExA1
        current_key = self.v[opcode.x]
        if self.keys[current_key] == 0:
            self.pc += 2

    def op_ld_vx_dt(self, opcode: Opcode):
        # Fx07
        self.v[opcode.x] = self.dt

    def op_ld_vx_k(self, opcode: Opcode):
        # Fx0A
        pressed_keys = np.where(self.keys == 1)[0]
        if np.size(pressed_keys) == 0:
            self.pc -= 2
        else:
            self.v[opcode.x] = pressed_keys[0]

    def op_ld_dt_vx(self, opcode: Opcode):
        # Fx15
        self.dt = self.v[opcode.x]

    def op_ld_st_vx(self, opcode: Opcode):
        # Fx18
        self.st = self.v[opcode.x]

    def op_add_i_vx(self, opcode: Opcode):
        # Fx1E
        self.i += self.v[opcode.x]

    def op_ld_f_vx(self, opcode: Opcode):
        # Fx29
        self.i = self.FONTS_ADDRESS_MEMORY + (5 * self.v[opcode.x])

    def op_ld_b_vx(self, opcode: Opcode):
        # Fx33
        number = self.v[opcode.x]
        self.ram[self.i + 2] = number % 10
        self.ram[self.i + 1] = (number // 10) % 10
        self.ram[self.i] = (number // 100)

    def op_ld_i_vx(self, opcode: Opcode):
        # Fx55
        self.ram[self.i:self.i + opcode.x + 1] = self.v[:opcode.x + 1]

    def op_ld_vx_i(self, opcode: Opcode):
        # Fx65
        self.v[:opcode.x + 1] = self.ram[self.i:self.i + opcode.x + 1]