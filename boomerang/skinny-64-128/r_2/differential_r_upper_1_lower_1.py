
routes_upper = [
[
    0x0010001000000010, # after mc
    0x0080008000000080, # after sb
    0x0080008000c00080, # after tk8
    0x000880000c000080, # after sr

    0x80808c0000808088, # after mc

],
]

routes_lower = [
[
    0x2000000000000000,  # input
    0x3000000000000000,  # after sb
    0x3000000000000300,  # after tk1
    0x0300000000000300,  # after sr

    0x0300000003000000,  # after mc

],
]

print(len(routes_upper))