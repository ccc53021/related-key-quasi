
routes = [
[
    0x0020020020000000, # input
    0x0010010010000000, # after sb
    0x0010010010000001, # after tk1
    0x0001000100010001, # after sr

    0x0000000000010001, # after mc
    0x000000000008000b, # after sb
    0x0000000000000000, # after tk2
    0x0000000000000000, # after sr

    0x0000000000000000, # after mc
    0x0000000000000000, # after sb
    0x0000000000000100, # after tk3
    0x0000000000000100, # after sr

    0x0100000001000100, # after mc
    0x0b0000000a000b00, # after sb
    0x0b00000002000000, # after tk3
    0x00b0000020000000, # after sr

    0x00002000000000b0, # after mc
    0x0000100000000010, # after sb
    0x0000100000010010, # after tk4
    0x0000001000100010, # after sr

    0x0000000000100000, # after mc
    0x0000000000800000, # after sb
    0x00000000000b0000, # after tk5
    0x0000000000b00000, # after sr

    0x000000b000000000, # after mc
    0x0000001000000000, # after sb
    0x0000001001000000, # after tk6
    0x0000100010000000, # after sr

    0x1000000000001000, # after mc
    0x8000000000008000, # after sb
    0x800000000b000000, # after tk7
    0x08000000b0000000, # after sr

    0x0000b00000000800, # after mc
    0x0000100000000400, # after sb
    0x0000100000100400, # after tk8
    0x0000001001000400, # after sr

    0x0410011004000410, # after mc
    0x0280088002000280, # after sb
    0x0280088082b00280, # after tk6
    0x002880082b080280, # after sr

    0x8288ab00028082a0, # after mc
],
]

# print(len(routes[0]))