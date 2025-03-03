
import numpy, pyboolector

GIFT_SBOX = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
n = 4
m = 4
sbox = GIFT_SBOX
state_bits = 64
state_words = 16
sbox_bits = 4

permutation_bits_table_64 = [
    0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63,
    12, 1, 6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59,
    8, 13, 2, 7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55,
    4, 9, 14, 3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51
]


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


def get_quasidifferentials_by_one_difference_route(difference_route, key_route, route_number, min_weight, max_weight):
    def search_quasidifferential(one_difference_characteristic, f):
        differential_characteristic_rounds = (len(one_difference_characteristic) - 1) // 3

        btor = pyboolector.Boolector()
        btor.Set_opt(pyboolector.BTOR_OPT_MODEL_GEN, 1)

        us = [btor.Var(btor.BitVecSort(state_bits), "u%d" % i) for i in range(differential_characteristic_rounds + 1)]
        vs = [btor.Var(btor.BitVecSort(state_bits), "v%d" % i) for i in range(differential_characteristic_rounds)]

        def permute_bits(x, y):
            for i in range(state_bits):
                btor.Assert(y[i] == x[permutation_bits_table_64[i]])

        def get_lat_weight(lat, flag):
            lat_weight = numpy.zeros((2 ** n, 2 ** m))
            if flag == 0:
                p = numpy.log2(abs(lat[0, 0]))
                for i in range(2 ** n):
                    for j in range(2 ** m):
                        c = lat[i, j]
                        if c != 0:
                            w = p - numpy.log2(abs(c))
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

                btor.Assert(weight0 | weight1 | weight2)
                return btor.Cond(
                    weight1,
                    btor.Const(1, state_words),
                    btor.Cond(weight2, btor.Const(2, state_words), btor.Const(0, state_words))
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
            permute_bits(vs[i], us[i + 1])
            for j in range(0, state_bits, 4):
                u = us[i][j + 3:j]
                v = vs[i][j + 3:j]
                a = (one_difference_characteristic[3 * i] >> j) & 0xf
                b = (one_difference_characteristic[3 * i + 1] >> j) & 0xf
                w = get_one_words_weight(a, b, u, v)
                cost += w

        btor.Assert(us[0] == 0)
        btor.Assert(us[differential_characteristic_rounds] == 0)
        btor.Set_opt(pyboolector.BTOR_OPT_INCREMENTAL, 1)

        def compute_sign(diffs, trail):
            correlation_sign = 0
            for round in range(differential_characteristic_rounds):
                one_round_difference_input = diffs[3 * round]
                one_round_difference_s = diffs[3 * round + 1]
                one_round_mask_input = trail[2 * round]
                one_round_mask_s = trail[2 * round + 1]

                for i in range(0, state_words):
                    a = (one_round_difference_input >> (4 * i)) & 0xf
                    b = (one_round_difference_s >> (4 * i)) & 0xf
                    u = (one_round_mask_input >> (4 * i)) & 0xf
                    v = (one_round_mask_s >> (4 * i)) & 0xf

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
                        temp |= (us[i] != btor.Const(ws[i - 1], state_bits))
                    distinct &= temp
                btor.Assume(distinct)

                r = btor.Sat()
                if r == btor.SAT:
                    print("    # Solution: [#{} of weight {}]".format(len(previous) + 1, target), file=f)
                    print("    [", file=f)
                    trail = []
                    for i in range(differential_characteristic_rounds):
                        u = int(us[i].assignment, base=2)
                        v = int(vs[i].assignment, base=2)
                        print("     0x{:016x},".format(u), file=f)
                        print("     0x{:016x},".format(v), file=f)
                        trail.append(u)
                        trail.append(v)
                    s = compute_sign(one_difference_characteristic, trail)
                    if s < 0:
                        count_negative += 1
                    else:
                        count_positive += 1
                    trail.append(int(us[differential_characteristic_rounds].assignment, base=2))
                    mask_strings.append(trail)
                    print("     0x{:016x},".format(int(us[differential_characteristic_rounds].assignment, base=2)),
                          file=f)
                    print("     # Sign: {}".format(s), file=f)
                    print("    ],", file=f)
                    previous.append((s, [us[i].assignment for i in range(1, differential_characteristic_rounds)]))
                else:
                    print("     # No trails with weight equal to {}.".format(target), file=f)
                    print("     # sign = -1: {}".format(count_negative), file=f)
                    print("     # sign = +1: {}".format(count_positive), file=f)
                    break
            print("],", file=f)
            quasidifferentials.append(mask_strings)

        return quasidifferentials

    difference_characteristic_round = (len(difference_route) - 1) // 3
    filename = "result/quasidifferentials_r_{}_differential_route_{}.txt".format(difference_characteristic_round, route_number)
    f = open(filename, "w")
    print("# --------------------------------- route {} ---------------------------------------".format(route_number),
          file=f)
    print("routes = [".format(), file=f)
    quasidifferentials = search_quasidifferential(difference_route, f)
    print("],", file=f)
    print("", file=f)
    f.close()
    return quasidifferentials

import differential_p_58 as differentials
routes = differentials.routes
print(len(routes))
routes_mkds = [
    0x00000000000000000000000080020000
]

min_weight = 0
max_weight = 20

for rr in range(len(routes)):
        print("")
        difference_characteristic = routes[rr]
        route_key = routes_mkds[rr]
        quasidifferentials = get_quasidifferentials_by_one_difference_route(difference_characteristic, route_key, rr, min_weight, max_weight)
