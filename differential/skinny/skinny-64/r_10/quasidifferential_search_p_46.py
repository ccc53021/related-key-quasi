
import numpy, pyboolector

SKINNY_64_SBOX = [0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf]

n = 4
m = 4
sbox = SKINNY_64_SBOX
state_bits = 64
state_words = 16
sbox_bits = 4

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
                            if (abs(c) == 4) or (abs(c) == 8) or (abs(c) == 16) or (abs(c) == 32):
                                w = p - numpy.log2(abs(c))
                                lat_weight[i, j] = w
                            elif abs(c) == 12:
                                w = p - numpy.log2(abs(8))
                                lat_weight[i, j] = w
                            elif (abs(c) == 20) or (abs(c) == 24) or (abs(c) == 24) or (abs(c) == 28):
                                w = p - numpy.log2(abs(16))
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
                    weight2,
                    btor.Const(2, state_words),
                    btor.Cond(
                        weight1,
                        btor.Const(1, state_words),
                        btor.Const(0, state_words))

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
                        print("     0x{:016x},".format(ma), file=f)
                        print("     0x{:016x},".format(mb), file=f)
                        print("     0x{:016x},".format(mc), file=f)
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
                    print("     0x{:016x},".format(int(mask_a[differential_characteristic_rounds].assignment, base=2)),
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
    filename = "result/quasidifferentials_r_{}_differential_route_{}.txt".format(difference_characteristic_round, route_number)
    f = open(filename, "w")
    print("# --------------------------------- route {} ---------------------------------------".format(route_number),
          file=f)
    print("trails = [".format(), file=f)
    quasidifferentials = search_quasidifferential(difference_route, f)
    print("],", file=f)
    print("", file=f)
    f.close()
    return quasidifferentials

import differential_p_46 as differentials
routes = differentials.routes
print(len(routes))

min_weight = 0
max_weight = 21

for rr in range(len(routes)):
    print("")
    difference_characteristic = routes[rr]
    quasidifferentials = get_quasidifferentials_by_one_difference_route(difference_characteristic, rr, min_weight, max_weight)
