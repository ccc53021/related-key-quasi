
import numpy, pyboolector

AES_SBOX = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]
n = 8
m = 8
sbox = AES_SBOX
state_bits = 128
state_words = 32
sbox_bits = 8

permutation_bits_table_SR = [
    0, 1, 2, 3, 4, 5, 6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 80, 81, 82, 83, 84, 85, 86, 87, 120, 121, 122, 123, 124, 125, 126, 127,
    32, 33, 34, 35, 36, 37, 38, 39, 72, 73, 74, 75, 76, 77, 78, 79, 112, 113, 114, 115, 116, 117, 118, 119, 24, 25, 26, 27, 28, 29, 30, 31,
    64, 65, 66, 67, 68, 69, 70, 71, 104, 105, 106, 107, 108, 109, 110, 111, 16, 17, 18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63,
    96, 97, 98, 99, 100, 101, 102, 103, 8, 9, 10, 11, 12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 88, 89, 90, 91, 92, 93, 94, 95
]
mc_t_bits_matrix_2 = [
    [
        # 2
        [[0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 1, 1, 0, 0, 0]],
        # 1
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
        # 1
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
        # 3
        [[1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1],
         [1, 1, 0, 1, 1, 0, 0, 1]],
    ],
    [
        # 3
        [[1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1],
         [1, 1, 0, 1, 1, 0, 0, 1]],
        # 2
        [[0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 1, 1, 0, 0, 0]],
        # 1
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
        # 1
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],

    ],
    [
        # 1
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
        # 3
        [[1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1],
         [1, 1, 0, 1, 1, 0, 0, 1]],
        # 2
        [[0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 1, 1, 0, 0, 0]],
        # 1
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],

    ],
    [
        # 1
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
        # 1
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
        # 3
        [[1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1],
         [1, 1, 0, 1, 1, 0, 0, 1]],
        # 2
        [[0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 1, 1, 0, 0, 0]],
    ],
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
                btor.Assert(y[i] == x[permutation_bits_table_SR[i]])

        def mix_columns(x, y):
            for i in range(4):
                for j in range(32):
                    mj = []
                    for b in range(4):
                        mj = mj + mc_t_bits_matrix_2[j // 8][b][j % 8]

                    temp = btor.Const(0)
                    for k in range(len(mj)):
                        if mj[k] == 1:
                            temp ^= y[i * 32 + k]
                    btor.Assert(x[i * 32 + j] == temp)


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
            for j in range(0, state_bits, 8):
                u = mask_a[i][j + 7:j]
                v = mask_b[i][j + 7:j]
                a = (one_difference_characteristic[4 * i] >> j) & 0xff
                b = (one_difference_characteristic[4 * i + 1] >> j) & 0xff
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

        print("# differential : 0x{:032x} -> 0x{:032x}".format(one_difference_characteristic[0],
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
    print("routes = [".format(), file=f)
    quasidifferentials = search_quasidifferential(difference_route, f)
    print("],", file=f)
    print("", file=f)
    f.close()
    return quasidifferentials

import state_differential_p_48_route_1 as differentials

difference_characteristic = differentials.routes

min_weight = 0
max_weight = 50

quasidifferentials = get_quasidifferentials_by_one_difference_route(difference_characteristic, 1, min_weight, max_weight)