
import numpy, pyboolector

AES_SBOX = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]
n = 8
m = 8
sbox = AES_SBOX
state_bits = 128
state_words = 32
sbox_bits = 8

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
        differential_characteristic_rounds = len(one_difference_characteristic) // 2

        btor = pyboolector.Boolector()
        btor.Set_opt(pyboolector.BTOR_OPT_MODEL_GEN, 1)

        c0 = [btor.Var(btor.BitVecSort(32), "c0_%d" % i) for i in range(differential_characteristic_rounds + 1)]
        c1 = [btor.Var(btor.BitVecSort(32), "c1_%d" % i) for i in range(differential_characteristic_rounds + 1)]
        c2 = [btor.Var(btor.BitVecSort(32), "c2_%d" % i) for i in range(differential_characteristic_rounds + 1)]
        c3 = [btor.Var(btor.BitVecSort(32), "c3_%d" % i) for i in range(differential_characteristic_rounds + 1)]
        c30 = [btor.Var(btor.BitVecSort(32), "c30_%d" % i) for i in range(differential_characteristic_rounds)]
        c31 = [btor.Var(btor.BitVecSort(32), "c31_%d" % i) for i in range(differential_characteristic_rounds)]
        c30_1 = [btor.Var(btor.BitVecSort(32), "c30_1_%d" % i) for i in range(differential_characteristic_rounds)]
        c30_2 = [btor.Var(btor.BitVecSort(32), "c30_2_%d" % i) for i in range(differential_characteristic_rounds)]
        d0 = [btor.Var(btor.BitVecSort(32), "d0_%d" % i) for i in range(differential_characteristic_rounds)]
        d00 = [btor.Var(btor.BitVecSort(32), "d00_%d" % i) for i in range(differential_characteristic_rounds)]
        d01 = [btor.Var(btor.BitVecSort(32), "d01_%d" % i) for i in range(differential_characteristic_rounds)]
        d1 = [btor.Var(btor.BitVecSort(32), "d1_%d" % i) for i in range(differential_characteristic_rounds)]
        d10 = [btor.Var(btor.BitVecSort(32), "d10_%d" % i) for i in range(differential_characteristic_rounds)]
        d11 = [btor.Var(btor.BitVecSort(32), "d11_%d" % i) for i in range(differential_characteristic_rounds)]
        d2 = [btor.Var(btor.BitVecSort(32), "d2_%d" % i) for i in range(differential_characteristic_rounds)]
        d20 = [btor.Var(btor.BitVecSort(32), "d20_%d" % i) for i in range(differential_characteristic_rounds)]
        d21 = [btor.Var(btor.BitVecSort(32), "d21_%d" % i) for i in range(differential_characteristic_rounds)]
        d3 = [btor.Var(btor.BitVecSort(32), "d3_%d" % i) for i in range(differential_characteristic_rounds)]

        def t_permute_bits(x, y):
            table = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7]
            for i in range(32):
                btor.Assert(y[i] == x[table[i]])

        def operation_xor(x, x0, x1):
            for i in range(32):
                btor.Assert(x[i] == x0[i] ^ x1[i])

        def operation_branch(x, y, z):
            for i in range(32):
                btor.Assert(x[i] == y[i])
                btor.Assert(y[i] == z[i])
                btor.Assert(x[i] == z[i])

        def equal(x, y):
            for i in range(32):
                btor.Assert(y[i] == x[i])

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

                weight6 = btor.Const(0)
                for x in range(2 ** n):
                    for y in range(2 ** m):
                        if lat_weight[x, y] == 6:
                            weight6 |= (u == x) & (v == y)

                btor.Assert(weight0 | weight3 | weight4 | weight5 | weight6)
                return btor.Cond(
                    weight6,
                    btor.Const(6, state_words),
                    btor.Cond(
                        weight5,
                        btor.Const(5, state_words),
                        btor.Cond(
                            weight4,
                            btor.Const(4, state_words),
                            btor.Cond(
                                weight3,
                                btor.Const(3, state_words),
                                btor.Const(0, state_words))
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
            t_permute_bits(c30[i], c30_1[i])
            for j in range(0, 32, 8):
                u = c30_1[i][j + 7:j]
                v = c30_2[i][j + 7:j]
                a = (one_difference_characteristic[2 * i] >> j) & 0xff
                b = (one_difference_characteristic[2 * i + 1] >> j) & 0xff
                w = get_one_words_weight(a, b, u, v)
                cost += w
            operation_xor(c3[i], c30[i], c31[i])
            operation_branch(d0[i], c0[i], c30_2[i])
            operation_xor(d0[i], d00[i], d01[i])
            operation_branch(d01[i], c1[i], d1[i])
            operation_xor(d1[i], d10[i], d11[i])
            operation_branch(c2[i], d11[i], d2[i])
            operation_xor(d2[i], d20[i], d21[i])
            operation_branch(d3[i], d21[i], c31[i])

            equal(c0[i+1], d00[i])
            equal(c1[i+1], d10[i])
            equal(c2[i+1], d20[i])
            equal(c3[i+1], d3[i])

        btor.Assert(c0[0] == 0)
        btor.Assert(c1[0] == 0)
        btor.Assert(c2[0] == 0)
        btor.Assert(c3[0] == 0)
        btor.Assert(c0[differential_characteristic_rounds] == 0)
        btor.Assert(c1[differential_characteristic_rounds] == 0)
        btor.Assert(c2[differential_characteristic_rounds] == 0)
        btor.Assert(c3[differential_characteristic_rounds] == 0)

        btor.Set_opt(pyboolector.BTOR_OPT_INCREMENTAL, 1)

        def compute_sign(diffs, trail):
            correlation_sign = 0
            for round in range(differential_characteristic_rounds):
                one_round_difference_input = diffs[2 * round]
                one_round_difference_s = diffs[2 * round + 1]
                one_round_mask_input = trail[2 * round]
                one_round_mask_s = trail[2 * round + 1]

                for i in range(0, state_words):
                    a = (one_round_difference_input >> (8 * i)) & 0xff
                    b = (one_round_difference_s >> (8 * i)) & 0xff
                    u = (one_round_mask_input >> (8 * i)) & 0xff
                    v = (one_round_mask_s >> (8 * i)) & 0xff

                    if a != 0:
                        c = 1
                        if u == v == 0:
                            c = DDT[a, b]
                        else:
                            c = get_correlation_by_fixed_difference(a, b)[u, v]
                        if c < 0:
                            correlation_sign += 1
            return (-1) ** correlation_sign

        print("# differential : 0x{:08x} -> 0x{:08x}".format(one_difference_characteristic[0],
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
                for _, ccc_0, ccc_1, ccc_2, ccc_3 in previous:
                    temp = btor.Const(0)
                    for i in range(1, differential_characteristic_rounds):
                        temp |= (c0[i] != btor.Const(ccc_0[i], state_bits))
                        temp |= (c1[i] != btor.Const(ccc_1[i], state_bits))
                        temp |= (c2[i] != btor.Const(ccc_2[i], state_bits))
                        temp |= (c3[i] != btor.Const(ccc_3[i], state_bits))
                    distinct &= temp
                btor.Assume(distinct)

                r = btor.Sat()
                if r == btor.SAT:
                    print("    # Solution: [#{} of weight {}]".format(len(previous) + 1, target), file=f)
                    print("    [", file=f)
                    trail_sbox = []
                    for i in range(differential_characteristic_rounds):
                        sbox_in = int(c30_1[i].assignment, base=2)
                        sbox_out = int(c30_2[i].assignment, base=2)
                        trail_sbox.append(sbox_in)
                        trail_sbox.append(sbox_out)

                        cc_0 = int(c0[i].assignment, base=2)
                        cc_1 = int(c1[i].assignment, base=2)
                        cc_2 = int(c2[i].assignment, base=2)
                        cc_3 = int(c3[i].assignment, base=2)
                        print("     [0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}],".format(cc_0, cc_1, cc_2, cc_3), file=f)

                    s = compute_sign(one_difference_characteristic, trail_sbox)
                    if s < 0:
                        count_negative += 1
                    else:
                        count_positive += 1
                    cc_0_final = int(c0[differential_characteristic_rounds].assignment, base=2)
                    cc_1_final = int(c1[differential_characteristic_rounds].assignment, base=2)
                    cc_2_final = int(c2[differential_characteristic_rounds].assignment, base=2)
                    cc_3_final = int(c3[differential_characteristic_rounds].assignment, base=2)
                    print("     [0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}],".format(cc_0_final, cc_1_final, cc_2_final, cc_3_final), file=f)

                    print("     # Sign: {}".format(s), file=f)
                    print("    ],", file=f)
                    previous.append((s, [c0[i].assignment for i in range(0, differential_characteristic_rounds + 1)], [c1[i].assignment for i in range(0, differential_characteristic_rounds + 1)], [c2[i].assignment for i in range(0, differential_characteristic_rounds + 1)], [c3[i].assignment for i in range(0, differential_characteristic_rounds + 1)]))
                else:
                    print("     # No trails with weight equal to {}.".format(target), file=f)
                    print("     # sign = -1: {}".format(count_negative), file=f)
                    print("     # sign = +1: {}".format(count_positive), file=f)
                    break
            print("],", file=f)
            quasidifferentials.append(mask_strings)

        return quasidifferentials

    difference_characteristic_round = len(difference_route) // 2
    filename = "result/quasidifferentials_key_r_{}_differential_route_{}_w_{}_to_{}.txt".format(difference_characteristic_round, route_number, min_weight, max_weight)
    f = open(filename, "w")
    print("# --------------------------------- route {} ---------------------------------------".format(route_number),
          file=f)
    print("routes = [".format(), file=f)
    quasidifferentials = search_quasidifferential(difference_route, f)
    print("],", file=f)
    print("", file=f)
    f.close()
    return quasidifferentials


import key_differential_p_39_route_1 as differentials

routes = differentials.routes
print(len(routes))
min_weight = 0
max_weight = 50

quasidifferentials = get_quasidifferentials_by_one_difference_route(routes, 1, min_weight, max_weight)
