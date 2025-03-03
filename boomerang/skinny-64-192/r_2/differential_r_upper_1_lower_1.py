
routes_upper = [
[
    0x0040004000000040,  # after mc
    0x0020002000000020,  # after sb
    0x0020002000500020,  # after tk11
    0x0002200005000020,  # after sr

    0x2020250000202022,  # after mc

],
]

routes_lower = [
[
    0x0100100000010020,  # input
    0x0b00b000000b0050,  # after sb
    0x0b00b000000b0000,  # after tk1
    0x00b000b000b00000,  # after sr

    0x00b0000000000000,  # after mc

],
]

print(len(routes_upper))