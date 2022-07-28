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
        self.stack = np.zeros(64, dtype=np.ubyte)
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
