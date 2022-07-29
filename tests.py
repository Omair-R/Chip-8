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
        testcpu = cpu.CPU(screen="_")
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

    # def test_op_cls(self):
    #     testcpu = cpu.CPU(screen="_")

    #     for _ in range(10):
    #         testcpu.frame_buffer[np.random.randint(0, 64),
    #                              np.random.randint(0, 32)] = 1

    #     testcpu.op_cls(0x00E0)

    #     for i in range(testcpu.frame_buffer.shape[0]):
    #         for j in range(testcpu.frame_buffer.shape[1]):
    #             self.assertEqual(testcpu.frame_buffer[i, j], 0)

    def test_ret(self):
        testcpu = cpu.CPU(screen="_")
        testcpu.stack[testcpu.sp] = testcpu.pc
        testcpu.sp += 1

        testcpu.op_ret(0x00EE)
        self.assertEqual(testcpu.pc, 0x200)
        self.assertEqual(testcpu.sp, 0)

    def test_op_jp_addr(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x13A5))

        testcpu.op_jp_addr(testopcode)
        self.assertEqual(testcpu.pc, 0x3A5)

    def test_op_call_addr(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x23A5))

        testcpu.op_call_addr(testopcode)
        self.assertEqual(testcpu.pc, 0x3A5)
        self.assertEqual(testcpu.stack[testcpu.sp - 1], 0x200)

        testopcode = cpu.Opcode.adapt(np.ushort(0x24B6))
        testcpu.op_call_addr(testopcode)
        self.assertEqual(testcpu.pc, 0x4B6)
        self.assertEqual(testcpu.stack[testcpu.sp - 1], 0x3A5)

    def test_op_se_vx_byt(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x33B4))

        testcpu.v[3] = 0xB4
        testcpu.op_se_vx_byte(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

        testcpu.v[3] = 0xB5
        testcpu.op_se_vx_byte(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

    def test_op_sne_vx_byte(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x43B4))

        testcpu.v[3] = 0xB4
        testcpu.op_sne_vx_byte(testopcode)
        self.assertEqual(testcpu.pc, 0x200)

        testcpu.v[3] = 0xB5
        testcpu.op_sne_vx_byte(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

    def test_op_se_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x5344))

        testcpu.v[3] = 0xB4
        testcpu.v[4] = 0xB5
        testcpu.op_se_vx_vy(testopcode)
        self.assertEqual(testcpu.pc, 0x200)

        testcpu.v[3] = 0xB4
        testcpu.v[4] = 0xB4
        testcpu.op_se_vx_vy(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

    def test_op_ld_vx_byte(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x6344))

        testcpu.op_ld_vx_byte(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x44)

    def test_op_add_vx_byte(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x6344))

        testcpu.op_add_vx_byte(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x44)

        testcpu.op_add_vx_byte(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x44 + 0x44)

    def test_op_ld_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x7344))
        testcpu.v[4] = 0x68
        testcpu.op_ld_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x68)

        testcpu.v[4] = 0xAA
        testcpu.op_ld_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0xAA)

    def test_op_or_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x8341))

        testcpu.v[4] = 0xAA
        testcpu.op_or_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0xAA)

        testcpu.v[4] = 0x55
        testcpu.op_or_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0xFF)

    def test_op_and_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x8341))

        testcpu.v[3] = 0xAA
        testcpu.v[4] = 0x0F
        testcpu.op_and_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x0A)

    def test_op_xor_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x8341))

        testcpu.v[3] = 0xAF
        testcpu.v[4] = 0x0F
        testcpu.op_xor_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0xA0)

        testcpu.op_xor_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0xAF)

    def test_op_add_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x8341))

        testcpu.v[3] = 0x0F
        testcpu.v[4] = 0x0F
        testcpu.op_add_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x1E)
        self.assertEqual(testcpu.v[0xF], 0)

        testcpu.v[3] = 0xFF
        testcpu.v[4] = 0x0F
        testcpu.op_add_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x0E)
        self.assertEqual(testcpu.v[0xF], 1)

    def test_op_sub_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x8341))

        testcpu.v[3] = 0x0F
        testcpu.v[4] = 0x0F
        testcpu.op_sub_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x00)
        self.assertEqual(testcpu.v[0xF], 0)

        testcpu.v[3] = 0xFF
        testcpu.v[4] = 0x0F
        testcpu.op_sub_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0xF0)
        self.assertEqual(testcpu.v[0xF], 1)

    def test_op_shr_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x8341))

        testcpu.v[3] = 0x0F
        testcpu.op_shr_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x07)
        self.assertEqual(testcpu.v[0xF], 1)

        testcpu.v[3] = 0x0E
        testcpu.op_shr_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x07)
        self.assertEqual(testcpu.v[0xF], 0)

    def test_op_subn_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x8341))

        testcpu.v[3] = 0x0F
        testcpu.v[4] = 0x0F
        testcpu.op_subn_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x00)
        self.assertEqual(testcpu.v[0xF], 0)

        testcpu.v[3] = 0x0F
        testcpu.v[4] = 0xFF
        testcpu.op_subn_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0xF0)
        self.assertEqual(testcpu.v[0xF], 1)

    def test_op_shl_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x8341))

        testcpu.v[3] = 0xFF
        testcpu.op_shl_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0xFE)
        self.assertEqual(testcpu.v[0xF], 1)

        testcpu.v[3] = 0x7F
        testcpu.op_shl_vx_vy(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0xFE)
        self.assertEqual(testcpu.v[0xF], 0)

    def test_op_sne_vx_vy(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0x9340))

        testcpu.v[3] = 0xB4
        testcpu.v[4] = 0xB5
        testcpu.op_sne_vx_vy(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

        testcpu.v[3] = 0xB4
        testcpu.v[4] = 0xB4
        testcpu.op_sne_vx_vy(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

    def test_op_ld_i_addr(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xA340))
        testcpu.op_ld_i_addr(testopcode)
        self.assertEqual(testcpu.i, 0x340)

    def test_op_jp_v0_addr(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xB340))
        testcpu.op_jp_v0_addr(testopcode)
        self.assertEqual(testcpu.pc, 0x340)
        testcpu.v[0] = 1
        testcpu.op_jp_v0_addr(testopcode)
        self.assertEqual(testcpu.pc, 0x341)

    def test_op_rnd_vx_byte(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xc3AA))
        testcpu.op_rnd_vx_byte(testopcode)
        self.assertEqual(testcpu.v[3] & 0x55, 0)

        testopcode = cpu.Opcode.adapt(np.ushort(0xc30F))
        testcpu.op_rnd_vx_byte(testopcode)
        self.assertEqual(testcpu.v[3] & 0xF0, 0)

    def test_op_skp_vx(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xE39E))

        testcpu.v[3] = 4
        testcpu.keys[4] = 1
        testcpu.op_skp_vx(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

        testcpu.v[3] = 5
        testcpu.op_skp_vx(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

    def test_op_sknp_vx(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xE3A1))

        testcpu.v[3] = 4
        testcpu.op_sknp_vx(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

        testcpu.v[3] = 5
        testcpu.keys[5] = 1
        testcpu.op_sknp_vx(testopcode)
        self.assertEqual(testcpu.pc, 0x202)

    def test_op_ld_vx_dt(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xF307))

        testcpu.dt = 0x20
        testcpu.op_ld_vx_dt(testopcode)
        self.assertEqual(testcpu.v[testopcode.x], 0x20)

    def test_op_ld_dt_vx(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xF315))

        testcpu.v[3] = 0x20
        testcpu.op_ld_dt_vx(testopcode)
        self.assertEqual(testcpu.dt, 0x20)

    def test_op_ld_st_vx(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xF318))

        testcpu.v[3] = 0x20
        testcpu.op_ld_st_vx(testopcode)
        self.assertEqual(testcpu.st, 0x20)

    def test_op_add_i_vx(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xF31E))

        testcpu.v[3] = 0x20
        testcpu.op_add_i_vx(testopcode)
        self.assertEqual(testcpu.i, 0x20)

        testcpu.op_add_i_vx(testopcode)
        self.assertEqual(testcpu.i, 0x20 + 0x20)

    def test_op_ld_f_vx(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xF329))

        testcpu.op_ld_f_vx(testopcode)
        self.assertEqual(testcpu.i, 0x50)
        testcpu.v[3] = 3
        testcpu.op_ld_f_vx(testopcode)
        self.assertEqual(testcpu.i, 95)

    def test_op_ld_b_vx(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xF329))

        testcpu.v[3] = 254
        testcpu.op_ld_b_vx(testopcode)
        self.assertEqual(testcpu.ram[testcpu.i], 2)
        self.assertEqual(testcpu.ram[testcpu.i + 1], 5)
        self.assertEqual(testcpu.ram[testcpu.i + 2], 4)

    def test_op_ld_i_vx(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xF355))

        testcpu.v[:4] = [25, 34, 35, 60]

        testcpu.op_ld_i_vx(testopcode)
        for i in range(testcpu.v.shape[0]):
            self.assertEqual(testcpu.ram[testcpu.i + i], testcpu.v[i])

    def test_op_ld_vx_i(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xF355))

        testcpu.ram[testcpu.i:testcpu.i + 4] = [25, 34, 35, 60]

        testcpu.op_ld_vx_i(testopcode)
        for i in range(testcpu.v.shape[0]):
            self.assertEqual(testcpu.ram[testcpu.i + i], testcpu.v[i])

    def test_op_ld_vx_k(self):
        testcpu = cpu.CPU(screen="_")
        testopcode = cpu.Opcode.adapt(np.ushort(0xF355))

        testcpu.op_ld_vx_k(testopcode)

        self.assertNotEqual(testcpu.v[testopcode.x], 2)
        self.assertEqual(testcpu.pc, 510)

        testcpu.keys[5] = 1

        testcpu.op_ld_vx_k(testopcode)

        self.assertEqual(testcpu.v[testopcode.x], 5)

        testcpu.keys[2] = 1

        testcpu.op_ld_vx_k(testopcode)

        self.assertEqual(testcpu.v[testopcode.x], 2)


unittest.main()