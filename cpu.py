from dataclasses import dataclass
import numpy as np
from fonts import FONTS


@dataclass(frozen=True)
class Opcode:

    def __init__(self) -> None:
        self.NNN: np.ushort
        self.NN: np.ushort
        self.N: np.ubyte
        self.x: np.ubyte
        self.y: np.ubyte

    @classmethod
    def adapt(cls, opcode: np.ushort):

        cls.NNN = opcode & 0x0FFF
        cls.NN = opcode & 0x00FF
        cls.N = opcode & 0x000F
        cls.x = (opcode & 0x0F00) >> 8
        cls.y = (opcode & 0x00F0) >> 4
        return cls


class CPU:

    FIRST_ADDRESS_MEMORY = np.ushort(0x200)

    def __init__(self):
        self.v = np.zeros(16, dtype=np.ubyte)
        self.i: np.ushort = 0
        self.stack = np.zeros(64, dtype=np.ushort)
        self.sp: np.ubyte = 0
        self.dt: np.ubyte = 0
        self.st: np.ubyte = 0
        self.frame_buffer = np.zeros([64, 32], dtype=np.bool_)
        self.pc: np.ushort = self.FIRST_ADDRESS_MEMORY
        self.ram = np.zeros(4096, dtype=np.ubyte)
        self.keys = np.zeros(16, dtype=np.ubyte)
        self.ram[0x50:0x50 + FONTS.shape[0]] = FONTS

    def load_rom_to_ram(self, path: str) -> None:
        with open(path, "rb") as file:
            buffer = file.read()
        buffer_np = np.array(list(buffer), dtype=np.ubyte)
        self.ram[self.pc:self.pc + buffer_np.shape[0]] = buffer_np

    def op_cls(self, opcode: Opcode):
        # 00E0
        self.frame_buffer = np.zeros([64, 32], dtype=np.bool_)

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
        raise NotImplementedError

    def op_ld_vx_byte(self, opcode: Opcode):
        # 6xkk
        raise NotImplementedError

    def op_add_vx_byte(self, opcode: Opcode):
        # 7xkk
        raise NotImplementedError

    def op_ld_vx_vy(self, opcode: Opcode):
        # 8xy0
        raise NotImplementedError

    def op_or_vx_vy(self, opcode: Opcode):
        # 8xy1
        raise NotImplementedError

    def op_and_vx_vy(self, opcode: Opcode):
        # 8xy2
        raise NotImplementedError

    def op_xor_vx_vy(self, opcode: Opcode):
        # 8xy3
        raise NotImplementedError

    def op_add_vx_vy(self, opcode: Opcode):
        # 8xy4
        raise NotImplementedError

    def op_sub_vx_vy(self, opcode: Opcode):
        # 8xy5
        raise NotImplementedError

    def op_shr_vx_vy(self, opcode: Opcode):
        # 8xy6
        raise NotImplementedError

    def op_subn_vx_vy(self, opcode: Opcode):
        # 8xy7
        raise NotImplementedError

    def op_shl_vx_vy(self, opcode: Opcode):
        # 8xyE
        raise NotImplementedError

    def op_sne_vx_vy(self, opcode: Opcode):
        # 9xy0
        raise NotImplementedError

    def op_ld_i_addr(self, opcode: Opcode):
        # Annn
        raise NotImplementedError

    def op_jp_v0_addr(self, opcode: Opcode):
        # Bnnn
        raise NotImplementedError

    def op_rnd_vx_byte(self, opcode: Opcode):
        # Cxkk
        raise NotImplementedError

    def op_drw_vx_vy_nibble(self, opcode: Opcode):
        # Dxyn
        raise NotImplementedError

    def op_skp_vx(self, opcode: Opcode):
        # Ex9E
        raise NotImplementedError

    def op_sknp_vx(self, opcode: Opcode):
        # ExA1
        raise NotImplementedError

    def op_ld_vx_dt(self, opcode: Opcode):
        # Fx07
        raise NotImplementedError

    def op_ld_vx_k(self, opcode: Opcode):
        # Fx0A
        raise NotImplementedError

    def op_ld_dt_vx(self, opcode: Opcode):
        # Fx15
        raise NotImplementedError

    def op_ld_st_vx(self, opcode: Opcode):
        # Fx18
        raise NotImplementedError

    def op_add_i_vx(self, opcode: Opcode):
        # Fx1E
        raise NotImplementedError

    def op_ld_f_vx(self, opcode: Opcode):
        # Fx29
        raise NotImplementedError

    def op_ld_b_vx(self, opcode: Opcode):
        # Fx33
        raise NotImplementedError

    def op_ld_i_vx(self, opcode: Opcode):
        # Fx55
        raise NotImplementedError

    def op_ld_vx_i(self, opcode: Opcode):
        # Fx65
        raise NotImplementedError