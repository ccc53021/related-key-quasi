
import numpy, pyboolector

SKINNY_128_SBOX = [
    0x65,0x4c,0x6a,0x42,0x4b,0x63,0x43,0x6b,0x55,0x75,0x5a,0x7a,0x53,0x73,0x5b,0x7b,
    0x35,0x8c,0x3a,0x81,0x89,0x33,0x80,0x3b,0x95,0x25,0x98,0x2a,0x90,0x23,0x99,0x2b,
    0xe5,0xcc,0xe8,0xc1,0xc9,0xe0,0xc0,0xe9,0xd5,0xf5,0xd8,0xf8,0xd0,0xf0,0xd9,0xf9,
    0xa5,0x1c,0xa8,0x12,0x1b,0xa0,0x13,0xa9,0x05,0xb5,0x0a,0xb8,0x03,0xb0,0x0b,0xb9,
    0x32,0x88,0x3c,0x85,0x8d,0x34,0x84,0x3d,0x91,0x22,0x9c,0x2c,0x94,0x24,0x9d,0x2d,
    0x62,0x4a,0x6c,0x45,0x4d,0x64,0x44,0x6d,0x52,0x72,0x5c,0x7c,0x54,0x74,0x5d,0x7d,
    0xa1,0x1a,0xac,0x15,0x1d,0xa4,0x14,0xad,0x02,0xb1,0x0c,0xbc,0x04,0xb4,0x0d,0xbd,
    0xe1,0xc8,0xec,0xc5,0xcd,0xe4,0xc4,0xed,0xd1,0xf1,0xdc,0xfc,0xd4,0xf4,0xdd,0xfd,
    0x36,0x8e,0x38,0x82,0x8b,0x30,0x83,0x39,0x96,0x26,0x9a,0x28,0x93,0x20,0x9b,0x29,
    0x66,0x4e,0x68,0x41,0x49,0x60,0x40,0x69,0x56,0x76,0x58,0x78,0x50,0x70,0x59,0x79,
    0xa6,0x1e,0xaa,0x11,0x19,0xa3,0x10,0xab,0x06,0xb6,0x08,0xba,0x00,0xb3,0x09,0xbb,
    0xe6,0xce,0xea,0xc2,0xcb,0xe3,0xc3,0xeb,0xd6,0xf6,0xda,0xfa,0xd3,0xf3,0xdb,0xfb,
    0x31,0x8a,0x3e,0x86,0x8f,0x37,0x87,0x3f,0x92,0x21,0x9e,0x2e,0x97,0x27,0x9f,0x2f,
    0x61,0x48,0x6e,0x46,0x4f,0x67,0x47,0x6f,0x51,0x71,0x5e,0x7e,0x57,0x77,0x5f,0x7f,
    0xa2,0x18,0xae,0x16,0x1f,0xa7,0x17,0xaf,0x01,0xb2,0x0e,0xbe,0x07,0xb7,0x0f,0xbf,
    0xe2,0xca,0xee,0xc6,0xcf,0xe7,0xc7,0xef,0xd2,0xf2,0xde,0xfe,0xd7,0xf7,0xdf,0xff
]
n = 8
m = 8
sbox = SKINNY_128_SBOX
state_bits = 128
state_words = 16
sbox_bits = 8

def get_sr_table_by_n(bit_n):
    table = []
    for i in range(4 * n):
        table.append(i)
    for i in range(7 * n, 8 * n):
        table.append(i)
    for i in range(4 * n, 7 * n):
        table.append(i)
    for i in range(10 * n, 12 * n):
        table.append(i)
    for i in range(8 * n, 10 * n):
        table.append(i)
    for i in range(13 * n, 16 * n):
        table.append(i)
    for i in range(12 * n, 13 * n):
        table.append(i)

    return table

sr_bits_table = get_sr_table_by_n(sbox_bits)

def get_matrix_t_table_by_n(n):
    table = []
    for i in range(4 * n):
        one_bit_table = [i, i + 4 * n, i + 12 * n]
        table.append(one_bit_table)
    for i in range(4 * n, 8 * n):
        one_bit_table = [i + 4 * n]
        table.append(one_bit_table)
    for i in range(8 * n, 12 * n):
        one_bit_table = [i - 8 * n, i, i + 4 * n]
        table.append(one_bit_table)
    for i in range(12 * n, 16 * n):
        one_bit_table = [i - 12 * n]
        table.append(one_bit_table)

    return table
mc_t_bits_table = get_matrix_t_table_by_n(sbox_bits)

def get_DDT(sbox, n, m):
    DDT = numpy.zeros((2 ** n, 2 ** m))

    for input_d in range(2 ** n):
        for output_d in range(2 ** m):
            for x in range(2 ** n):
                y = x ^ input_d
                sx = sbox[x]
                sy = sbox[y]
                if (sx ^ sy) == output_d:
                    DDT[input_d, output_d] += 1

    return DDT


DDT = get_DDT(sbox, n, m)


def vector_inner_product(u, v, x, fx, n, m):
    left = 0
    for i in range(n):
        left += ((u >> i) & 0x1) * ((x >> i) & 0x1)
    left = left % 2
    right = 0
    for j in range(m):
        right += ((v >> j) & 0x1) * ((fx >> j) & 0x1)
    right = right % 2

    return left ^ right


def get_correlation_by_fixed_difference(a, b):
    IN_a_to_b = []
    for x in range(2 ** n):
        y = x ^ a
        sx = sbox[x]
        sy = sbox[y]
        if (sx ^ sy) == b:
            IN_a_to_b.append(x)

    LAT = numpy.zeros((2 ** n, 2 ** m))

    for u in range(2 ** n):
        for v in range(2 ** m):
            count_x = 0
            for x in IN_a_to_b:
                c = vector_inner_product(u, v, x, sbox[x], n, m)
                count_x += (-1) ** c
            LAT[u, v] = count_x

    return LAT


def get_quasidifferentials_by_one_difference_route(difference_route, route_number, min_weight, max_weight):
    def search_quasidifferential(one_difference_characteristic, f):
        differential_characteristic_rounds = (len(one_difference_characteristic) - 1) // 4

        btor = pyboolector.Boolector()
        btor.Set_opt(pyboolector.BTOR_OPT_MODEL_GEN, 1)

        mask_a = [btor.Var(btor.BitVecSort(state_bits), "mask_a%d" % i) for i in range(differential_characteristic_rounds + 1)]
        mask_b = [btor.Var(btor.BitVecSort(state_bits), "mask_b%d" % i) for i in range(differential_characteristic_rounds)]
        mask_c = [btor.Var(btor.BitVecSort(state_bits), "mask_c%d" % i) for i in range(differential_characteristic_rounds)]

        def permute_bits_sr(x, y):
            for i in range(state_bits):
                btor.Assert(y[i] == x[sr_bits_table[i]])


        def mix_columns(x, y):
            for i in range(state_bits):
                one_bit_table = mc_t_bits_table[i]
                temp = btor.Const(0)
                for j in range(len(one_bit_table)):
                    temp ^= y[one_bit_table[j]]
                btor.Assert(x[i] == temp)

        def get_lat_weight(lat, flag):
            lat_weight = numpy.zeros((2 ** n, 2 ** m))
            if flag == 0:
                p = numpy.log2(abs(lat[0, 0]))
                for i in range(2 ** n):
                    for j in range(2 ** m):
                        c = lat[i, j]
                        if c != 0:
                            if (abs(c) == 8) or (abs(c) == 16) or (abs(c) == 32) or (abs(c) == 64) or (abs(c) == 128):
                                w = p - numpy.log2(abs(c))
                                lat_weight[i, j] = w
                            elif (abs(c) > 4) and (abs(c) < 8):
                                w = p - numpy.log2(abs(4))
                                lat_weight[i, j] = w
                            elif (abs(c) > 8) and (abs(c) < 16):
                                w = p - numpy.log2(abs(8))
                                lat_weight[i, j] = w
                            elif (abs(c) > 16) and (abs(c) < 32):
                                w = p - numpy.log2(abs(16))
                                lat_weight[i, j] = w
                            elif (abs(c) > 32) and (abs(c) < 64):
                                w = p - numpy.log2(abs(32))
                                lat_weight[i, j] = w
                            elif (abs(c) > 64) and (abs(c) < 128):
                                w = p - numpy.log2(abs(64))
                                lat_weight[i, j] = w
                            elif (abs(c) > 128) and (abs(c) < 256):
                                w = p - numpy.log2(abs(128))
                                lat_weight[i, j] = w

            else:
                if flag == 1:
                    if lat[0, 0] != 0:
                        p = numpy.log2(abs(lat[0, 0]))
                        for i in range(2 ** n):
                            for j in range(2 ** m):
                                c = lat[i, j]
                                if c != 0:
                                    w = p - numpy.log2(abs(c))
                                    lat_weight[i, j] = w
                                else:
                                    if c == 0:
                                        lat_weight[i, j] = 1
                    else:
                        print("# error! a = {} b = {}".format(a, b), file=f)

            return lat_weight

        def get_one_words_weight(a, b, u, v):
            if a == b == 0:
                lat_weight = get_lat_weight(get_correlation_by_fixed_difference(0, 0), 0)
                weight0 = (u == 0) & (v == 0)

                weight1 = btor.Const(0)
                for x in range(2 ** n):
                    for y in range(2 ** m):
                        if lat_weight[x, y] == 1:
                            weight1 |= (u == x) & (v == y)

                weight2 = btor.Const(0)
                for x in range(2 ** n):
                    for y in range(2 ** m):
                        if lat_weight[x, y] == 2:
                            weight2 |= (u == x) & (v == y)

                weight3 = btor.Const(0)
                for x in range(2 ** n):
                    for y in range(2 ** m):
                        if lat_weight[x, y] == 3:
                            weight3 |= (u == x) & (v == y)

                weight4 = btor.Const(0)
                for x in range(2 ** n):
                    for y in range(2 ** m):
                        if lat_weight[x, y] == 4:
                            weight4 |= (u == x) & (v == y)

                weight5 = btor.Const(0)
                for x in range(2 ** n):
                    for y in range(2 ** m):
                        if lat_weight[x, y] == 5:
                            weight5 |= (u == x) & (v == y)

                btor.Assert(weight0 | weight1 | weight2 | weight3 | weight4 | weight5)
                return btor.Cond(
                    weight5,
                    btor.Const(5, state_words),
                    btor.Cond(
                        weight4,
                        btor.Const(4, state_words),
                        btor.Cond(
                            weight3,
                            btor.Const(3, state_words),
                            btor.Cond(
                                weight2,
                                btor.Const(2, state_words),
                                btor.Cond(
                                    weight1,
                                    btor.Const(1, state_words),
                                    btor.Const(0, state_words)
                                )
                            )
                        )
                    )

                )
            else:
                lat_weight = get_lat_weight(get_correlation_by_fixed_difference(a, b), 1)
                allowed = btor.Const(0)
                for x in range(2 ** n):
                    for y in range(2 ** m):
                        if lat_weight[x, y] == 0:
                            allowed |= (u == x) & (v == y)
                btor.Assert(allowed)
                return btor.Const(0, state_words)

        cost = btor.Const(0, state_words)
        for i in range(differential_characteristic_rounds):
            for j in range(0, state_bits, sbox_bits):
                u = mask_a[i][j + sbox_bits - 1:j]
                v = mask_b[i][j + sbox_bits - 1:j]
                a = (one_difference_characteristic[4 * i] >> j) & (2 ** sbox_bits - 1)
                b = (one_difference_characteristic[4 * i + 1] >> j) & (2 ** sbox_bits - 1)
                w = get_one_words_weight(a, b, u, v)
                cost += w
            permute_bits_sr(mask_b[i], mask_c[i])
            mix_columns(mask_c[i], mask_a[i + 1])

        btor.Assert(mask_a[0] == 0)
        btor.Assert(mask_a[differential_characteristic_rounds] == 0)
        btor.Set_opt(pyboolector.BTOR_OPT_INCREMENTAL, 1)

        def compute_sign(diffs, trail):
            correlation_sign = 0
            for round in range(differential_characteristic_rounds):
                one_round_difference_input = diffs[4 * round]
                one_round_difference_s = diffs[4 * round + 1]
                one_round_mask_input = trail[3 * round]
                one_round_mask_s = trail[3 * round + 1]

                for i in range(0, state_words):
                    a = (one_round_difference_input >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
                    b = (one_round_difference_s >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
                    u = (one_round_mask_input >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
                    v = (one_round_mask_s >> (sbox_bits * i)) & (2 ** sbox_bits - 1)

                    if a != 0:
                        c = 1
                        if u == v == 0:
                            c = DDT[a, b]
                        else:
                            c = get_correlation_by_fixed_difference(a, b)[u, v]
                        if c < 0:
                            correlation_sign += 1
            return (-1) ** correlation_sign

        print("# differential : 0x{:016x} -> 0x{:016x}".format(one_difference_characteristic[0],
                                                               one_difference_characteristic[
                                                                   len(one_difference_characteristic) - 1]), file=f)
        quasidifferentials = []
        for target in range(min_weight, max_weight):
            mask_strings = []
            previous = []
            print("# Solution: of weight {}".format(target), file=f)
            print("[", file=f)

            count_negative = 0
            count_positive = 0

            while True:
                btor.Assume(cost == target)
                distinct = btor.Const(1)
                for _, ws in previous:
                    temp = btor.Const(0)
                    for i in range(1, differential_characteristic_rounds):
                        temp |= (mask_a[i] != btor.Const(ws[i - 1], state_bits))
                    distinct &= temp
                btor.Assume(distinct)

                r = btor.Sat()
                if r == btor.SAT:
                    print("    # Solution: [#{} of weight {}]".format(len(previous) + 1, target), file=f)
                    print("    [", file=f)
                    trail = []
                    for i in range(differential_characteristic_rounds):
                        ma = int(mask_a[i].assignment, base=2)
                        mb = int(mask_b[i].assignment, base=2)
                        mc = int(mask_c[i].assignment, base=2)
                        print("     0x{:032x},".format(ma), file=f)
                        print("     0x{:032x},".format(mb), file=f)
                        print("     0x{:032x},".format(mc), file=f)
                        trail.append(ma)
                        trail.append(mb)
                        trail.append(mc)
                    s = compute_sign(one_difference_characteristic, trail)
                    if s < 0:
                        count_negative += 1
                    else:
                        count_positive += 1
                    trail.append(int(mask_a[differential_characteristic_rounds].assignment, base=2))
                    mask_strings.append(trail)
                    print("     0x{:032x},".format(int(mask_a[differential_characteristic_rounds].assignment, base=2)),
                          file=f)
                    print("     # Sign: {}".format(s), file=f)
                    print("    ],", file=f)
                    previous.append((s, [mask_a[i].assignment for i in range(1, differential_characteristic_rounds)]))
                else:
                    print("     # No trails with weight equal to {}.".format(target), file=f)
                    print("     # sign = -1: {}".format(count_negative), file=f)
                    print("     # sign = +1: {}".format(count_positive), file=f)
                    break
            print("],", file=f)
            quasidifferentials.append(mask_strings)

        return quasidifferentials

    difference_characteristic_round = (len(difference_route) - 1) // 4
    filename = "result/quasidifferentials_r_{}_differential_route_{}_w_{}_to_{}.txt".format(difference_characteristic_round, route_number, min_weight, max_weight)
    f = open(filename, "w")
    print("# --------------------------------- route {} ---------------------------------------".format(route_number),
          file=f)
    print("trails = [".format(), file=f)
    quasidifferentials = search_quasidifferential(difference_route, f)
    print("],", file=f)
    print("", file=f)
    f.close()
    return quasidifferentials


import differential_p_110 as differentials
routes = differentials.routes
print(len(routes))

min_weight = 0
max_weight = 21

for rr in range(len(routes)):
    print("")
    difference_characteristic = routes[rr]
    quasidifferentials = get_quasidifferentials_by_one_difference_route(difference_characteristic, rr, min_weight, max_weight)
