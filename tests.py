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

    def test_op_cls(self):
        testcpu = cpu.CPU()

        for _ in range(10):
            testcpu.frame_buffer[np.random.randint(0, 64),
                                 np.random.randint(0, 32)] = 1

        testcpu.op_cls(0x00E0)

        for i in range(testcpu.frame_buffer.shape[0]):
            for j in range(testcpu.frame_buffer.shape[1]):
                self.assertEqual(testcpu.frame_buffer[i, j], 0)

    def test_ret(self):
        testcpu = cpu.CPU()
        testcpu.stack[testcpu.sp] = testcpu.pc
        testcpu.sp += 1

        testcpu.op_ret(0x00EE)
        self.assertEqual(testcpu.pc, 0x200)
        self.assertEqual(testcpu.sp, 0)

    def test_op_jp_addr(self):
        testcpu = cpu.CPU()
        testopcode = cpu.Opcode.adapt(np.ushort(0x13A5))

        testcpu.op_jp_addr(testopcode)
        self.assertEqual(testcpu.pc, 0x3A5)

    def test_op_call_addr(self):
        testcpu = cpu.CPU()
        testopcode = cpu.Opcode.adapt(np.ushort(0x23A5))

        testcpu.op_call_addr(testopcode)
        self.assertEqual(testcpu.pc, 0x3A5)
        self.assertEqual(testcpu.stack[testcpu.sp - 1], 0x200)

        testopcode = cpu.Opcode.adapt(np.ushort(0x24B6))
        testcpu.op_call_addr(testopcode)
        self.assertEqual(testcpu.pc, 0x4B6)
        self.assertEqual(testcpu.stack[testcpu.sp - 1], 0x3A5)

    def test_op_se_vx_byt(self):
        testcpu = cpu.CPU()
        testopcode = cpu.Opcode.adapt(np.ushort(0x33B4))

        testcpu.v[3] = 0xB4
        testcpu.op_se_vx_byte(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

        testcpu.v[3] = 0xB5
        testcpu.op_se_vx_byte(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

    def test_op_sne_vx_byte(self):
        testcpu = cpu.CPU()
        testopcode = cpu.Opcode.adapt(np.ushort(0x33B4))

        testcpu.v[3] = 0xB4
        testcpu.op_sne_vx_byte(testopcode)
        self.assertEqual(testcpu.pc, 0x200)

        testcpu.v[3] = 0xB5
        testcpu.op_sne_vx_byte(testopcode)
        self.assertEqual(testcpu.pc, 0x202)


unittest.main()