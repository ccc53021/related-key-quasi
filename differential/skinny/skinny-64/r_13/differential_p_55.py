
routes = [
[
    0x0000080000280000, # input
    0x0000040000140000, # after sb
    0x0000040040140000, # after tk1
    0x0000000401440000, # after sr

    0x0004014000000004, # after mc
    0x00020a2000000002, # after sb
    0x00020a20a000000a, # after tk2
    0x2000200a000a000a, # after sr

    0x20002000000a0000, # after mc
    0x3000600000060000, # after sb
    0x3000600000060000, # after tk3
    0x0300006000600000, # after sr

    0x0060000000000360, # after mc
    0x0010000000000f30, # after sb
    0x0010000000000000, # after tk4
    0x0001000000000000, # after sr

    0x0000000000000001, # after mc
    0x0000000000000009, # after sb
    0x0000000000000000, # after tk5
    0x0000000000000000, # after sr

    0x0000000000000000, # after mc
    0x0000000000000000, # after sb
    0x0000000000000000, # after tk6
    0x0000000000000000, # after sr

    0x0000000000000000, # after mc
    0x0000000000000000, # after sb
    0x0000000000000a00, # after tk7
    0x0000000000000a00, # after sr

    0x0a0000000a000a00, # after mc
    0x0500000005000600, # after sb
    0x050000000b000000, # after tk8
    0x00500000b0000000, # after sr

    0x0000b00000000050, # after mc
    0x0000c000000000c0, # after sb
    0x0000c000000c00c0, # after tk9
    0x000000c000c000c0, # after sr

    0x0000000000c00000, # after mc
    0x0000000000200000, # after sb
    0x00000000000b0000, # after tk10
    0x0000000000b00000, # after sr

    0x000000b000000000, # after mc
    0x0000001000000000, # after sb
    0x0000001001000000, # after tk11
    0x0000100010000000, # after sr

    0x1000000000001000, # after mc
    0x800000000000a000, # after sb
    0x8000000001000000, # after tk12
    0x0800000010000000, # after sr

    0x0000100000000800, # after mc
    0x0000800000000400, # after sb
    0x0000800000b00400, # after tk13
    0x000000800b000400, # after sr

    0x04800b8004000480, # after mc
],
]

# print(len(routes))