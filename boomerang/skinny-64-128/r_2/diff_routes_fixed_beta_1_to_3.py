
routes = [
    # 1 : 8 8 8
    [
        0x0010001000000010, # after mc
        0x0080008000000080, # after sb
        0x0080008000c00080, # after tk8
        0x000880000c000080, # after sr

        0x80808c0000808088, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 2 : 8 8 9
    [
        0x0010001000000010, # after mc
        0x0080008000000090, # after sb
        0x0080008000c00090, # after tk8
        0x000880000c000090, # after sr

        0x80908c0000908098, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 3 : 8 8 a
    [
        0x0010001000000010, # after mc
        0x00800080000000a0, # after sb
        0x0080008000c000a0, # after tk8
        0x000880000c0000a0, # after sr

        0x80a08c0000a080a8, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 4 : 8 8 b
    [
        0x0010001000000010, # after mc
        0x00800080000000b0, # after sb
        0x0080008000c000b0, # after tk8
        0x000880000c0000b0, # after sr

        0x80b08c0000b080b8, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 5 : 8 9 8
    [
        0x0010001000000010, # after mc
        0x0080009000000080, # after sb
        0x0080009000c00080, # after tk8
        0x000890000c000080, # after sr

        0x90809c0000809088, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 6 : 8 9 9
    [
        0x0010001000000010, # after mc
        0x0080009000000090, # after sb
        0x0080009000c00090, # after tk8
        0x000890000c000090, # after sr

        0x90909c0000909098, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 7 : 8 9 a
    [
        0x0010001000000010, # after mc
        0x00800090000000a0, # after sb
        0x0080009000c000a0, # after tk8
        0x000890000c0000a0, # after sr

        0x90a09c0000a090a8, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 8 : 8 9 b
    [
        0x0010001000000010, # after mc
        0x00800090000000b0, # after sb
        0x0080009000c000b0, # after tk8
        0x000890000c0000b0, # after sr

        0x90b09c0000b090b8, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 9 : 8 a 8
    [
        0x0010001000000010, # after mc
        0x008000a000000080, # after sb
        0x008000a000c00080, # after tk8
        0x0008a0000c000080, # after sr

        0xa080ac000080a088, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 10 : 8 a 9
    [
        0x0010001000000010, # after mc
        0x008000a000000090, # after sb
        0x008000a000c00090, # after tk8
        0x0008a0000c000090, # after sr

        0xa090ac000090a098, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 11 : 8 a a
    [
        0x0010001000000010, # after mc
        0x008000a0000000a0, # after sb
        0x008000a000c000a0, # after tk8
        0x0008a0000c0000a0, # after sr

        0xa0a0ac0000a0a0a8, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 12 : 8 a b
    [
        0x0010001000000010, # after mc
        0x008000a0000000b0, # after sb
        0x008000a000c000b0, # after tk8
        0x0008a0000c0000b0, # after sr

        0xa0b0ac0000b0a0b8, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 13 : 8 b 8
    [
        0x0010001000000010, # after mc
        0x008000b000000080, # after sb
        0x008000b000c00080, # after tk8
        0x0008b0000c000080, # after sr

        0xb080bc000080b088, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 14 : 8 b 9
    [
        0x0010001000000010, # after mc
        0x008000b000000090, # after sb
        0x008000b000c00090, # after tk8
        0x0008b0000c000090, # after sr

        0xb090bc000090b098, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 15 : 8 b a
    [
        0x0010001000000010, # after mc
        0x008000b0000000a0, # after sb
        0x008000b000c000a0, # after tk8
        0x0008b0000c0000a0, # after sr

        0xb0a0bc0000a0b0a8, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 16 : 8 b b
    [
        0x0010001000000010, # after mc
        0x008000b0000000b0, # after sb
        0x008000b000c000b0, # after tk8
        0x0008b0000c0000b0, # after sr

        0xb0b0bc0000b0b0b8, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 17 : 9 8 8
    [
        0x0010001000000010, # after mc
        0x0090008000000080, # after sb
        0x0090008000c00080, # after tk8
        0x000980000c000080, # after sr

        0x80808c0000808089, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 18 : 9 8 9
    [
        0x0010001000000010, # after mc
        0x0090008000000090, # after sb
        0x0090008000c00090, # after tk8
        0x000980000c000090, # after sr

        0x80908c0000908099, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 19 : 9 8 a
    [
        0x0010001000000010, # after mc
        0x00900080000000a0, # after sb
        0x0090008000c000a0, # after tk8
        0x000980000c0000a0, # after sr

        0x80a08c0000a080a9, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 20 : 9 8 b
    [
        0x0010001000000010, # after mc
        0x00900080000000b0, # after sb
        0x0090008000c000b0, # after tk8
        0x000980000c0000b0, # after sr

        0x80b08c0000b080b9, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 21 : 9 9 8
    [
        0x0010001000000010, # after mc
        0x0090009000000080, # after sb
        0x0090009000c00080, # after tk8
        0x000990000c000080, # after sr

        0x90809c0000809089, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 22 : 9 9 9
    [
        0x0010001000000010, # after mc
        0x0090009000000090, # after sb
        0x0090009000c00090, # after tk8
        0x000990000c000090, # after sr

        0x90909c0000909099, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 23 : 9 9 a
    [
        0x0010001000000010, # after mc
        0x00900090000000a0, # after sb
        0x0090009000c000a0, # after tk8
        0x000990000c0000a0, # after sr

        0x90a09c0000a090a9, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 24 : 9 9 b
    [
        0x0010001000000010, # after mc
        0x00900090000000b0, # after sb
        0x0090009000c000b0, # after tk8
        0x000990000c0000b0, # after sr

        0x90b09c0000b090b9, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 25 : 9 a 8
    [
        0x0010001000000010, # after mc
        0x009000a000000080, # after sb
        0x009000a000c00080, # after tk8
        0x0009a0000c000080, # after sr

        0xa080ac000080a089, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 26 : 9 a 9
    [
        0x0010001000000010, # after mc
        0x009000a000000090, # after sb
        0x009000a000c00090, # after tk8
        0x0009a0000c000090, # after sr

        0xa090ac000090a099, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 27 : 9 a a
    [
        0x0010001000000010, # after mc
        0x009000a0000000a0, # after sb
        0x009000a000c000a0, # after tk8
        0x0009a0000c0000a0, # after sr

        0xa0a0ac0000a0a0a9, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 28 : 9 a b
    [
        0x0010001000000010, # after mc
        0x009000a0000000b0, # after sb
        0x009000a000c000b0, # after tk8
        0x0009a0000c0000b0, # after sr

        0xa0b0ac0000b0a0b9, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 29 : 9 b 8
    [
        0x0010001000000010, # after mc
        0x009000b000000080, # after sb
        0x009000b000c00080, # after tk8
        0x0009b0000c000080, # after sr

        0xb080bc000080b089, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 30 : 9 b 9
    [
        0x0010001000000010, # after mc
        0x009000b000000090, # after sb
        0x009000b000c00090, # after tk8
        0x0009b0000c000090, # after sr

        0xb090bc000090b099, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 31 : 9 b a
    [
        0x0010001000000010, # after mc
        0x009000b0000000a0, # after sb
        0x009000b000c000a0, # after tk8
        0x0009b0000c0000a0, # after sr

        0xb0a0bc0000a0b0a9, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 32 : 9 b b
    [
        0x0010001000000010, # after mc
        0x009000b0000000b0, # after sb
        0x009000b000c000b0, # after tk8
        0x0009b0000c0000b0, # after sr

        0xb0b0bc0000b0b0b9, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 33 : a 8 8
    [
        0x0010001000000010, # after mc
        0x00a0008000000080, # after sb
        0x00a0008000c00080, # after tk8
        0x000a80000c000080, # after sr

        0x80808c000080808a, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 34 : a 8 9
    [
        0x0010001000000010, # after mc
        0x00a0008000000090, # after sb
        0x00a0008000c00090, # after tk8
        0x000a80000c000090, # after sr

        0x80908c000090809a, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 35 : a 8 a
    [
        0x0010001000000010, # after mc
        0x00a00080000000a0, # after sb
        0x00a0008000c000a0, # after tk8
        0x000a80000c0000a0, # after sr

        0x80a08c0000a080aa, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 36 : a 8 b
    [
        0x0010001000000010, # after mc
        0x00a00080000000b0, # after sb
        0x00a0008000c000b0, # after tk8
        0x000a80000c0000b0, # after sr

        0x80b08c0000b080ba, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 37 : a 9 8
    [
        0x0010001000000010, # after mc
        0x00a0009000000080, # after sb
        0x00a0009000c00080, # after tk8
        0x000a90000c000080, # after sr

        0x90809c000080908a, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 38 : a 9 9
    [
        0x0010001000000010, # after mc
        0x00a0009000000090, # after sb
        0x00a0009000c00090, # after tk8
        0x000a90000c000090, # after sr

        0x90909c000090909a, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 39 : a 9 a
    [
        0x0010001000000010, # after mc
        0x00a00090000000a0, # after sb
        0x00a0009000c000a0, # after tk8
        0x000a90000c0000a0, # after sr

        0x90a09c0000a090aa, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 40 : a 9 b
    [
        0x0010001000000010, # after mc
        0x00a00090000000b0, # after sb
        0x00a0009000c000b0, # after tk8
        0x000a90000c0000b0, # after sr

        0x90b09c0000b090ba, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 41 : a a 8
    [
        0x0010001000000010, # after mc
        0x00a000a000000080, # after sb
        0x00a000a000c00080, # after tk8
        0x000aa0000c000080, # after sr

        0xa080ac000080a08a, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 42 : a a 9
    [
        0x0010001000000010, # after mc
        0x00a000a000000090, # after sb
        0x00a000a000c00090, # after tk8
        0x000aa0000c000090, # after sr

        0xa090ac000090a09a, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 43 : a a a
    [
        0x0010001000000010, # after mc
        0x00a000a0000000a0, # after sb
        0x00a000a000c000a0, # after tk8
        0x000aa0000c0000a0, # after sr

        0xa0a0ac0000a0a0aa, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 44 : a a b
    [
        0x0010001000000010, # after mc
        0x00a000a0000000b0, # after sb
        0x00a000a000c000b0, # after tk8
        0x000aa0000c0000b0, # after sr

        0xa0b0ac0000b0a0ba, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 45 : a b 8
    [
        0x0010001000000010, # after mc
        0x00a000b000000080, # after sb
        0x00a000b000c00080, # after tk8
        0x000ab0000c000080, # after sr

        0xb080bc000080b08a, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 46 : a b 9
    [
        0x0010001000000010, # after mc
        0x00a000b000000090, # after sb
        0x00a000b000c00090, # after tk8
        0x000ab0000c000090, # after sr

        0xb090bc000090b09a, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 47 : a b a
    [
        0x0010001000000010, # after mc
        0x00a000b0000000a0, # after sb
        0x00a000b000c000a0, # after tk8
        0x000ab0000c0000a0, # after sr

        0xb0a0bc0000a0b0aa, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 48 : a b b
    [
        0x0010001000000010, # after mc
        0x00a000b0000000b0, # after sb
        0x00a000b000c000b0, # after tk8
        0x000ab0000c0000b0, # after sr

        0xb0b0bc0000b0b0ba, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 49 : b 8 8
    [
        0x0010001000000010, # after mc
        0x00b0008000000080, # after sb
        0x00b0008000c00080, # after tk8
        0x000b80000c000080, # after sr

        0x80808c000080808b, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 50 : b 8 9
    [
        0x0010001000000010, # after mc
        0x00b0008000000090, # after sb
        0x00b0008000c00090, # after tk8
        0x000b80000c000090, # after sr

        0x80908c000090809b, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 51 : b 8 a
    [
        0x0010001000000010, # after mc
        0x00b00080000000a0, # after sb
        0x00b0008000c000a0, # after tk8
        0x000b80000c0000a0, # after sr

        0x80a08c0000a080ab, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 52 : b 8 b
    [
        0x0010001000000010, # after mc
        0x00b00080000000b0, # after sb
        0x00b0008000c000b0, # after tk8
        0x000b80000c0000b0, # after sr

        0x80b08c0000b080bb, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 53 : b 9 8
    [
        0x0010001000000010, # after mc
        0x00b0009000000080, # after sb
        0x00b0009000c00080, # after tk8
        0x000b90000c000080, # after sr

        0x90809c000080908b, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 54 : b 9 9
    [
        0x0010001000000010, # after mc
        0x00b0009000000090, # after sb
        0x00b0009000c00090, # after tk8
        0x000b90000c000090, # after sr

        0x90909c000090909b, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 55 : b 9 a
    [
        0x0010001000000010, # after mc
        0x00b00090000000a0, # after sb
        0x00b0009000c000a0, # after tk8
        0x000b90000c0000a0, # after sr

        0x90a09c0000a090ab, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 56 : b 9 b
    [
        0x0010001000000010, # after mc
        0x00b00090000000b0, # after sb
        0x00b0009000c000b0, # after tk8
        0x000b90000c0000b0, # after sr

        0x90b09c0000b090bb, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 57 : b a 8
    [
        0x0010001000000010, # after mc
        0x00b000a000000080, # after sb
        0x00b000a000c00080, # after tk8
        0x000ba0000c000080, # after sr

        0xa080ac000080a08b, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 58 : b a 9
    [
        0x0010001000000010, # after mc
        0x00b000a000000090, # after sb
        0x00b000a000c00090, # after tk8
        0x000ba0000c000090, # after sr

        0xa090ac000090a09b, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 59 : b a a
    [
        0x0010001000000010, # after mc
        0x00b000a0000000a0, # after sb
        0x00b000a000c000a0, # after tk8
        0x000ba0000c0000a0, # after sr

        0xa0a0ac0000a0a0ab, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 60 : b a b
    [
        0x0010001000000010, # after mc
        0x00b000a0000000b0, # after sb
        0x00b000a000c000b0, # after tk8
        0x000ba0000c0000b0, # after sr

        0xa0b0ac0000b0a0bb, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 61 : b b 8
    [
        0x0010001000000010, # after mc
        0x00b000b000000080, # after sb
        0x00b000b000c00080, # after tk8
        0x000bb0000c000080, # after sr

        0xb080bc000080b08b, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 62 : b b 9
    [
        0x0010001000000010, # after mc
        0x00b000b000000090, # after sb
        0x00b000b000c00090, # after tk8
        0x000bb0000c000090, # after sr

        0xb090bc000090b09b, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 63 : b b a
    [
        0x0010001000000010, # after mc
        0x00b000b0000000a0, # after sb
        0x00b000b000c000a0, # after tk8
        0x000bb0000c0000a0, # after sr

        0xb0a0bc0000a0b0ab, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
    # 64 : b b b
    [
        0x0010001000000010, # after mc
        0x00b000b0000000b0, # after sb
        0x00b000b000c000b0, # after tk8
        0x000bb0000c0000b0, # after sr

        0xb0b0bc0000b0b0bb, # after mc
        0x3000000000000000,  # after sb
        0x3000000000000300,  # after tk1
        0x0300000000000300,  # after sr

        0x0300000003000000,  # after mc
    ],
]

print(len(routes))