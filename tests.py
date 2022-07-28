import unittest
import cpu
from fonts import FONTS
import numpy as np


class TestCPU(unittest.TestCase):

    def test_opcode_segementation(self):
        testopcode = cpu.Opcode.adapt(np.ushort(0xB3A5))

        self.assertEqual(testopcode.NNN, 0x03A5)
        self.assertEqual(testopcode.NN, 0x00A5)
        self.assertEqual(testopcode.N, 0x0005)
        self.assertEqual(testopcode.x, 3)
        self.assertEqual(testopcode.y, 10)

    def test_init_cpu(self):
        testcpu = cpu.CPU()
        test_fonts = testcpu.ram[0x50:0x50 + 80]

        for i in range(FONTS.shape[0]):
            self.assertEqual(FONTS[i], test_fonts[i])

        with open(r'roms\Paddles.ch8', "rb") as file:
            buffer = file.read()

        buffer_np = np.array(list(buffer), dtype=np.ubyte)

        testcpu.load_rom_to_ram(r'roms\Paddles.ch8')
        testing_ram = testcpu.ram[0x200:0x200 + buffer_np.shape[0]]

        for i in range(buffer_np.shape[0]):
            self.assertEqual(buffer_np[i], testing_ram[i])


unittest.main()