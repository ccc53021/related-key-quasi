
routes = [
    # 1 : 2 7 2
    [
        0x0040004000000040, # after mc
        0x0020007000000020, # after sb
        0x0020007000500020, # after tk8
        0x0002700005000020, # after sr

        0x7020750000207022, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 2 : 2 7 b
    [
        0x0040004000000040, # after mc
        0x00200070000000b0, # after sb
        0x00200070005000b0, # after tk8
        0x00027000050000b0, # after sr

        0x70b0750000b070b2, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 3 : 2 c 2
    [
        0x0040004000000040, # after mc
        0x002000c000000020, # after sb
        0x002000c000500020, # after tk8
        0x0002c00005000020, # after sr

        0xc020c5000020c022, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 4 : 2 c b
    [
        0x0040004000000040, # after mc
        0x002000c0000000b0, # after sb
        0x002000c0005000b0, # after tk8
        0x0002c000050000b0, # after sr

        0xc0b0c50000b0c0b2, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 5 : 2 d 2
    [
        0x0040004000000040, # after mc
        0x002000d000000020, # after sb
        0x002000d000500020, # after tk8
        0x0002d00005000020, # after sr

        0xd020d5000020d022, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 6 : 2 d b
    [
        0x0040004000000040, # after mc
        0x002000d0000000b0, # after sb
        0x002000d0005000b0, # after tk8
        0x0002d000050000b0, # after sr

        0xd0b0d50000b0d0b2, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 7 : 7 7 2
    [
        0x0040004000000040, # after mc
        0x0070007000000020, # after sb
        0x0070007000500020, # after tk8
        0x0007700005000020, # after sr

        0x7020750000207027, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 8 : 7 7 b
    [
        0x0040004000000040, # after mc
        0x00700070000000b0, # after sb
        0x00700070005000b0, # after tk8
        0x00077000050000b0, # after sr

        0x70b0750000b070b7, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 9 : 7 c 2
    [
        0x0040004000000040, # after mc
        0x007000c000000020, # after sb
        0x007000c000500020, # after tk8
        0x0007c00005000020, # after sr

        0xc020c5000020c027, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 10 : 7 c b
    [
        0x0040004000000040, # after mc
        0x007000c0000000b0, # after sb
        0x007000c0005000b0, # after tk8
        0x0007c000050000b0, # after sr

        0xc0b0c50000b0c0b7, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 11 : 7 d 2
    [
        0x0040004000000040, # after mc
        0x007000d000000020, # after sb
        0x007000d000500020, # after tk8
        0x0007d00005000020, # after sr

        0xd020d5000020d027, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 12 : 7 d b
    [
        0x0040004000000040, # after mc
        0x007000d0000000b0, # after sb
        0x007000d0005000b0, # after tk8
        0x0007d000050000b0, # after sr

        0xd0b0d50000b0d0b7, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 13 : b 7 2
    [
        0x0040004000000040, # after mc
        0x00b0007000000020, # after sb
        0x00b0007000500020, # after tk8
        0x000b700005000020, # after sr

        0x702075000020702b, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 14 : b 7 b
    [
        0x0040004000000040, # after mc
        0x00b00070000000b0, # after sb
        0x00b00070005000b0, # after tk8
        0x000b7000050000b0, # after sr

        0x70b0750000b070bb, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 15 : b c 2
    [
        0x0040004000000040, # after mc
        0x00b000c000000020, # after sb
        0x00b000c000500020, # after tk8
        0x000bc00005000020, # after sr

        0xc020c5000020c02b, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 16 : b c b
    [
        0x0040004000000040, # after mc
        0x00b000c0000000b0, # after sb
        0x00b000c0005000b0, # after tk8
        0x000bc000050000b0, # after sr

        0xc0b0c50000b0c0bb, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 17 : b d 2
    [
        0x0040004000000040, # after mc
        0x00b000d000000020, # after sb
        0x00b000d000500020, # after tk8
        0x000bd00005000020, # after sr

        0xd020d5000020d02b, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 18 : b d b
    [
        0x0040004000000040, # after mc
        0x00b000d0000000b0, # after sb
        0x00b000d0005000b0, # after tk8
        0x000bd000050000b0, # after sr

        0xd0b0d50000b0d0bb, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 19 : c 7 2
    [
        0x0040004000000040, # after mc
        0x00c0007000000020, # after sb
        0x00c0007000500020, # after tk8
        0x000c700005000020, # after sr

        0x702075000020702c, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 20 : c 7 b
    [
        0x0040004000000040, # after mc
        0x00c00070000000b0, # after sb
        0x00c00070005000b0, # after tk8
        0x000c7000050000b0, # after sr

        0x70b0750000b070bc, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 21 : c c 2
    [
        0x0040004000000040, # after mc
        0x00c000c000000020, # after sb
        0x00c000c000500020, # after tk8
        0x000cc00005000020, # after sr

        0xc020c5000020c02c, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 22 : c c b
    [
        0x0040004000000040, # after mc
        0x00c000c0000000b0, # after sb
        0x00c000c0005000b0, # after tk8
        0x000cc000050000b0, # after sr

        0xc0b0c50000b0c0bc, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 23 : c d 2
    [
        0x0040004000000040, # after mc
        0x00c000d000000020, # after sb
        0x00c000d000500020, # after tk8
        0x000cd00005000020, # after sr

        0xd020d5000020d02c, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 24 : c d b
    [
        0x0040004000000040, # after mc
        0x00c000d0000000b0, # after sb
        0x00c000d0005000b0, # after tk8
        0x000cd000050000b0, # after sr

        0xd0b0d50000b0d0bc, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 25 : d 7 2
    [
        0x0040004000000040, # after mc
        0x00d0007000000020, # after sb
        0x00d0007000500020, # after tk8
        0x000d700005000020, # after sr

        0x702075000020702d, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 26 : d 7 b
    [
        0x0040004000000040, # after mc
        0x00d00070000000b0, # after sb
        0x00d00070005000b0, # after tk8
        0x000d7000050000b0, # after sr

        0x70b0750000b070bd, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 27 : d c 2
    [
        0x0040004000000040, # after mc
        0x00d000c000000020, # after sb
        0x00d000c000500020, # after tk8
        0x000dc00005000020, # after sr

        0xc020c5000020c02d, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 28 : d c b
    [
        0x0040004000000040, # after mc
        0x00d000c0000000b0, # after sb
        0x00d000c0005000b0, # after tk8
        0x000dc000050000b0, # after sr

        0xc0b0c50000b0c0bd, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 29 : d d 2
    [
        0x0040004000000040, # after mc
        0x00d000d000000020, # after sb
        0x00d000d000500020, # after tk8
        0x000dd00005000020, # after sr

        0xd020d5000020d02d, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
    # 30 : d d b
    [
        0x0040004000000040, # after mc
        0x00d000d0000000b0, # after sb
        0x00d000d0005000b0, # after tk8
        0x000dd000050000b0, # after sr

        0xd0b0d50000b0d0bd, # after mc
        0x0b00b000000b0050,  # after sb
        0x0b00b000000b0000,  # after tk1
        0x00b000b000b00000,  # after sr

        0x00b0000000000000,  # after mc
    ],
]