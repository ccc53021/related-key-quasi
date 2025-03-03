
import numpy, pyboolector
from multiprocessing import Pool

SKINNY_64_SBOX = [0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf]
SKINNY_64_INVERSE_SBOX = [0x3, 4, 6, 8, 12, 10, 1, 14, 9, 2, 5, 7, 0, 11, 13, 15]
n = 4
m = 4
sbox = SKINNY_64_SBOX
sbox_inverse = SKINNY_64_INVERSE_SBOX
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

def get_matrix_table_by_n(bit_n):
    table = []
    for i in range(4 * n):
        one_bit_table = [i, i + 8 * n, i + 12 * n]
        table.append(one_bit_table)
    for i in range(4 * n, 8 * n):
        one_bit_table = [i - 4 * n]
        table.append(one_bit_table)
    for i in range(8 * n, 12 * n):
        one_bit_table = [i - 4 * n, i]
        table.append(one_bit_table)
    for i in range(12 * n, 16 * n):
        one_bit_table = [i - 12 * n, i - 4 * n]
        table.append(one_bit_table)

    return table
mc_bits_table = get_matrix_table_by_n(sbox_bits)

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

def vector_inner_product(u, x, n):
    left = 0
    for i in range(n):
        left += ((u >> i) & 0x1) * ((x >> i) & 0x1)
    left = left % 2

    return left

def get_bct_matrices(sbox):
    def matrix_f1():
        table = numpy.zeros((2 ** (3 * n), 2 ** (3 * n)))
        for u0 in range(2 ** n):
            for u1 in range(2 ** n):
                for u2 in range(2 ** n):
                    for x0 in range(2 ** n):
                        for delta_1 in range(2 ** n):
                            for delta_2 in range(2 ** n):
                                input = (x0 << 2 * n) + (delta_1 << n) + delta_2
                                output = (u0 << 2 * n) + (u1 << n) + u2
                                if u1 != delta_1:
                                    continue
                                else:
                                    if (vector_inner_product(u0, x0, n) ^ vector_inner_product(u2, delta_2, n) == 0):
                                        table[input, output] = 1
                                    else:
                                        table[input, output] = -1

        return table

    # f2 : (y, lambda_1, lambda_2) -> (v0, v1, v2)
    def matrix_f2():
        table = numpy.zeros((2 ** (3 * n), 2 ** (3 * n)))
        for y0 in range(2 ** m):
            for lambda_1 in range(2 ** m):
                for lambda_2 in range(2 ** m):
                    for v0 in range(2 ** m):
                        for v1 in range(2 ** m):
                            for v2 in range(2 ** m):
                                input = (y0 << 2 * m) + (lambda_1 << m) + lambda_2
                                output = (v0 << 2 * m) + (v1 << m) + v2
                                if v2 != lambda_2:
                                    continue
                                else:
                                    if (vector_inner_product(v0, y0, m) ^ vector_inner_product(v1, lambda_1, n) == 0):
                                        table[input, output] = 1
                                    else:
                                        table[input, output] = -1

        return table

    def matrix_e(sbox):
        table = numpy.zeros((2 ** (3 * n), 2 ** (3 * m)))
        for x0 in range(2 ** n):
            for delta_1 in range(2 ** n):
                for delta_2 in range(2 ** n):
                    for y0 in range(2 ** m):
                        for lambda_1 in range(2 ** m):
                            for lambda_2 in range(2 ** m):
                                output = (x0 << 2 * n) + (delta_1 << n) + delta_2
                                input = (y0 << 2 * m) + (lambda_1 << m) + lambda_2
                                if (y0 == sbox[x0]) and (lambda_1 == sbox[x0] ^ sbox[x0 ^ delta_1]) and (
                                        lambda_2 == sbox[x0] ^ sbox[x0 ^ delta_2]) and (
                                        lambda_1 ^ lambda_2 == sbox[x0] ^ sbox[x0 ^ delta_1 ^ delta_2]):
                                    table[input, output] = 1


        return table

    def matrix_e1(sbox):
        table = numpy.zeros((2 ** (3 * n), 2 ** (3 * m)))
        for x in range(2 ** n):
            for u0 in range(2 ** n):
                for u1 in range(2 ** n):
                    for u2 in range(2 ** n):
                        for v0 in range(2 ** m):
                            for v1 in range(2 ** m):
                                for v2 in range(2 ** m):
                                    for delta_2 in range(2 ** n):
                                        output = (u0 << 2 * n) + (u1 << n) + u2
                                        input = (v0 << 2 * m) + (v1 << m) + v2
                                        delta_1 = u1
                                        lambda_2 = v2
                                        if (sbox[x] ^ sbox[x ^ delta_2] == lambda_2) and (
                                                sbox[x] ^ sbox[x ^ delta_1 ^ delta_2] == v2 ^ sbox[x] ^ sbox[x ^ u1]):
                                            if vector_inner_product(u0, x, n) ^ vector_inner_product(u2, delta_2,
                                                                                                     n) ^ vector_inner_product(
                                                    v0, sbox[x], m) ^ vector_inner_product(v1,
                                                                                           sbox[x] ^ sbox[x ^ delta_1],
                                                                                           m) == 0:
                                                table[input, output] += 1
                                            else:
                                                table[input, output] -= 1

        for input in range(2 ** (3 * n)):
            for output in range(2 ** (3 * m)):
                if table[input, output] != 0:
                    table[input, output] = table[input, output] / (2 ** (2 * n))

        return table

    # e1 = matrix_e1(sbox)
    f1 = matrix_f1()
    e = matrix_e(sbox)
    f2 = matrix_f2()
    f1_inverse = numpy.linalg.inv(f1)
    quasi_bct = numpy.dot(numpy.dot(f2, e), f1_inverse)

    return quasi_bct


def get_ddt_matrices(sbox):
    # f1 : (x, delta_1, delta_2) -> (u0, u1, u2)
    def matrix_f1():
        table = numpy.zeros((2 ** (3 * n), 2 ** (3 * n)))
        for u0 in range(2 ** n):
            for u1 in range(2 ** n):
                for u2 in range(2 ** n):
                    for x0 in range(2 ** n):
                        for delta_1 in range(2 ** n):
                            for delta_2 in range(2 ** n):
                                input = (x0 << 2 * n) + (delta_1 << n) + delta_2
                                output = (u0 << 2 * n) + (u1 << n) + u2
                                if u1 != delta_1:
                                    continue
                                else:
                                    if (vector_inner_product(u0, x0, n) ^ vector_inner_product(u2, delta_2, n) == 0):
                                        table[input, output] = 1
                                    else:
                                        table[input, output] = -1

        return table

    # f2 : (y, lambda_1, lambda_2) -> (v0, v1, v2)
    def matrix_f2():
        table = numpy.zeros((2 ** (3 * n), 2 ** (3 * n)))
        for y0 in range(2 ** m):
            for lambda_1 in range(2 ** m):
                for lambda_2 in range(2 ** m):
                    for v0 in range(2 ** m):
                        for v1 in range(2 ** m):
                            for v2 in range(2 ** m):
                                input = (y0 << 2 * m) + (lambda_1 << m) + lambda_2
                                output = (v0 << 2 * m) + (v1 << m) + v2
                                if v1 != lambda_1:
                                    continue
                                else:
                                    if (vector_inner_product(v0, y0, m) ^ vector_inner_product(v2, lambda_2, n) == 0):
                                        table[input, output] = 1
                                    else:
                                        table[input, output] = -1

        return table

    def matrix_e(sbox):
        table = numpy.zeros((2 ** (3 * n), 2 ** (3 * m)))
        for x0 in range(2 ** n):
            for delta_1 in range(2 ** n):
                for delta_2 in range(2 ** n):
                    for y0 in range(2 ** m):
                        for lambda_1 in range(2 ** m):
                            for lambda_2 in range(2 ** m):
                                output = (x0 << 2 * n) + (delta_1 << n) + delta_2
                                input = (y0 << 2 * m) + (lambda_1 << m) + lambda_2
                                if (y0 == sbox[x0]) and (lambda_1 == sbox[x0] ^ sbox[x0 ^ delta_1]) and (
                                        lambda_2 == sbox[x0] ^ sbox[x0 ^ delta_2]) and (
                                        lambda_1 ^ lambda_2 == sbox[x0] ^ sbox[x0 ^ delta_1 ^ delta_2]):
                                   table[input, output] = 1


        return table

    def matrix_e1(sbox):
        table = numpy.zeros((2 ** (3 * n), 2 ** (3 * m)))
        for x in range(2 ** n):
            for u0 in range(2 ** n):
                for u1 in range(2 ** n):
                    for u2 in range(2 ** n):
                        for v0 in range(2 ** m):
                            for v1 in range(2 ** m):
                                for v2 in range(2 ** m):
                                    for delta_2 in range(2 ** n):
                                        output = (u0 << 2 * n) + (u1 << n) + u2
                                        input = (v0 << 2 * m) + (v1 << m) + v2
                                        delta_1 = u1
                                        lambda_1 = v1
                                        if (sbox[x] ^ sbox[x ^ delta_1] == lambda_1) and (
                                                sbox[x] ^ sbox[x ^ delta_1 ^ delta_2] == v1 ^ sbox[x] ^ sbox[x ^ delta_2]):
                                            if vector_inner_product(u0, x, n) ^ vector_inner_product(u2, delta_2,
                                                                                                     n) ^ vector_inner_product(
                                                    v0, sbox[x], m) ^ vector_inner_product(v2,
                                                                                           sbox[x] ^ sbox[x ^ delta_2],
                                                                                           m) == 0:
                                                table[input, output] += 1
                                            else:
                                                table[input, output] -= 1

        for input in range(2 ** (3 * n)):
            for output in range(2 ** (3 * m)):
                if table[input, output] != 0:
                    table[input, output] = table[input, output] / (2 ** (2 * n))

        return table

    # e1 = matrix_e1(sbox)
    f1 = matrix_f1()
    e = matrix_e(sbox)
    f2 = matrix_f2()
    f1_inverse = numpy.linalg.inv(f1)
    quasi_ddt = numpy.dot(numpy.dot(f2, e), f1_inverse)

    return quasi_ddt

quasi_bct = get_bct_matrices(sbox)
quasi_ddt = get_ddt_matrices(sbox)

def get_small_quasi_ddt_by_u1_and_v1(quasi_ddt, u1, v1):
    table = numpy.zeros((2 ** (2 * n), 2 ** (2 * m)))

    for u0 in range(2 ** n):
        for u2 in range(2 ** n):
            for v0 in range(2 ** m):
                for v2 in range(2 ** m):
                    input = (v0 << 2 * m) + (v1 << m) + v2
                    output = (u0 << 2 * n) + (u1 << n) + u2
                    ii = (v0 << m) + v2
                    oo = (u0 << n) + u2
                    table[ii, oo] = quasi_ddt[input, output]

    return table

def get_small_quasi_bct_by_u1_and_v2(quasi_bct, u1, v2):
    table = numpy.zeros((2 ** (2 * n), 2 ** (2 * m)))

    for u0 in range(2 ** n):
        for u2 in range(2 ** n):
            for v0 in range(2 ** m):
                for v1 in range(2 ** m):
                    input = (v0 << 2 * m) + (v1 << m) + v2
                    output = (u0 << 2 * n) + (u1 << n) + u2
                    ii = (v0 << m) + v1
                    oo = (u0 << n) + u2
                    table[ii, oo] = quasi_bct[input, output]

    return table


def get_small_quasi_tables_weight_lists_by_u_and_v(table, is_ddt):
    weight_list_w_0_to_7 = []
    for w in range(0, 8):
        weight_list_w_0_to_7.append([])

    for u in range(2 ** n):
        for uu in range(2 ** n):
            for v in range(2 ** m):
                for vv in range(2 ** m):
                    input = (v << m) + vv
                    output = (u << n) + uu
                    c = table[input, output]
                    if is_ddt == 0:
                        c *= (2 ** n)
                    if c != 0:
                        w = int(abs(numpy.log2(abs(c))))
                        weight_list_w_0_to_7[w].append([u, uu, v, vv])

    print("---------------------------------")

    return weight_list_w_0_to_7

def compute_sign(route_u0, route_u1, diff_route, round_upper, round_lower):
    correlation_sign = 0

    # upper
    for r in range(round_upper):
        u0 = route_u0[4 * r]
        u1 = route_u1[4 * r]
        delta_in_1 = diff_route[4 * r]
        v0 = route_u0[4 * r + 1]
        v1 = route_u1[4 * r + 1]
        delta_out_1 = diff_route[4 * r + 1]

        for i in range(0, state_words):
            uu0 = (u0 >> (4 * i)) & (2 ** sbox_bits - 1)
            uu1 = (u1 >> (4 * i)) & (2 ** sbox_bits - 1)
            di1 = (delta_in_1 >> (4 * i)) & (2 ** sbox_bits - 1)
            vv0 = (v0 >> (4 * i)) & (2 ** sbox_bits - 1)
            vv1 = (v1 >> (4 * i)) & (2 ** sbox_bits - 1)
            do1 = (delta_out_1 >> (4 * i)) & (2 ** sbox_bits - 1)

            input = (vv0 << 2 * m) + (do1 << m) + vv1
            output = (uu0 << 2 * n) + (di1 << n) + uu1

            c = quasi_ddt[input, output]
            if c < 0:
                correlation_sign += 1

    # lower
    for r in range(round_upper + 1, round_upper + round_lower):
        u0 = route_u0[4 * r]
        u1 = route_u1[4 * r]
        delta_in_2 = diff_route[4 * r]
        v0 = route_u0[4 * r + 1]
        v1 = route_u1[4 * r + 1]
        delta_out_2 = diff_route[4 * r + 1]

        for i in range(0, state_words):
            uu0 = (u0 >> (4 * i)) & (2 ** sbox_bits - 1)
            uu1 = (u1 >> (4 * i)) & (2 ** sbox_bits - 1)
            di2 = (delta_in_2 >> (4 * i)) & (2 ** sbox_bits - 1)
            vv0 = (v0 >> (4 * i)) & (2 ** sbox_bits - 1)
            vv1 = (v1 >> (4 * i)) & (2 ** sbox_bits - 1)
            do2 = (delta_out_2 >> (4 * i)) & (2 ** sbox_bits - 1)

            input = (vv0 << 2 * m) + (vv1 << m) + do2
            output = (uu0 << 2 * n) + (uu1 << n) + di2

            c = quasi_ddt[input, output]
            if c < 0:
                correlation_sign += 1


    # em
    u0 = route_u0[4 * round_upper]
    u1 = route_u1[4 * round_upper]
    delta_in_1 = diff_route[4 * round_upper]
    v0 = route_u0[4 * round_upper + 1]
    v1 = route_u1[4 * round_upper + 1]
    delta_out_2 = diff_route[4 * round_upper + 1]
    for i in range(0, state_words):
        uu0 = (u0 >> (4 * i)) & (2 ** sbox_bits - 1)
        uu1 = (u1 >> (4 * i)) & (2 ** sbox_bits - 1)
        di1 = (delta_in_1 >> (4 * i)) & (2 ** sbox_bits - 1)
        vv0 = (v0 >> (4 * i)) & (2 ** sbox_bits - 1)
        vv1 = (v1 >> (4 * i)) & (2 ** sbox_bits - 1)
        do2 = (delta_out_2 >> (4 * i)) & (2 ** sbox_bits - 1)

        input = (vv0 << 2 * m) + (vv1 << m) + do2
        output = (uu0 << 2 * n) + (di1 << n) + uu1

        c = quasi_ddt[input, output]
        if c < 0:
            correlation_sign += 1

    return (-1) ** correlation_sign

def get_quasidifferentials_by_one_fixed_input_and_output(diff_route, diff_k, route_number, min_weight, max_weight):
    differential_characteristic_rounds_upper = 1
    differential_characteristic_rounds_lower = 1

    def search_differential(delta_in, delta_out, f):

        btor = pyboolector.Boolector()
        btor.Set_opt(pyboolector.BTOR_OPT_MODEL_GEN, 1)

        # difference
        # input
        u0_a = [btor.Var(btor.BitVecSort(state_bits), "u0_a%d" % i) for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower + 1)]
        u1_a = [btor.Var(btor.BitVecSort(state_bits), "u1_a%d" % i) for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower + 1)]
        # after sb
        u0_b = [btor.Var(btor.BitVecSort(state_bits), "u0_b%d" % i) for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower)]
        u1_b = [btor.Var(btor.BitVecSort(state_bits), "u1_b%d" % i) for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower)]
        # after_k
        u0_c = [btor.Var(btor.BitVecSort(state_bits), "u0_c%d" % i) for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower)]
        u1_c = [btor.Var(btor.BitVecSort(state_bits), "u1_c%d" % i) for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower)]
        # after sr
        u0_d = [btor.Var(btor.BitVecSort(state_bits), "u0_d%d" % i) for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower)]
        u1_d = [btor.Var(btor.BitVecSort(state_bits), "u1_d%d" % i) for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower)]

        def xor_k_diff(x, y, k):
            print(k)
            for i in range(state_bits):
                btor.Assert(y[i] == x[i] ^ ((k >> i) & 0x1))

        def xor_k_mask(x, y):
            for i in range(state_bits):
                btor.Assert(y[i] == x[i])

        def permute_bits_sr(x, y):
            for i in range(state_bits):
                btor.Assert(y[i] == x[sr_bits_table[i]])

        def mix_t_columns(x, y):
            for i in range(state_bits):
                one_bit_table = mc_t_bits_table[i]
                temp = btor.Const(0)
                for j in range(len(one_bit_table)):
                    temp ^= y[one_bit_table[j]]
                btor.Assert(x[i] == temp)


        def get_one_words_weight_by_small_quasi_table(u, uu, v, vv, small_table_weight_lists_w_0_to_7):
            weight0 = btor.Const(0)
            weight1 = btor.Const(0)
            weight2 = btor.Const(0)
            weight3 = btor.Const(0)
            weight4 = btor.Const(0)
            weight5 = btor.Const(0)
            weight6 = btor.Const(0)
            weight7 = btor.Const(0)
            for tw in range(len(small_table_weight_lists_w_0_to_7)):
                table_weight_list_w = small_table_weight_lists_w_0_to_7[tw]
                if len((table_weight_list_w)) > 0:
                    for one_group in table_weight_list_w:
                        u0 = one_group[0]
                        uu0 = one_group[1]
                        v0 = one_group[2]
                        vv0 = one_group[3]
                        if tw == 0:
                            weight0 |= (u == u0) & (uu == uu0) & (v == v0) & (vv == vv0)
                        elif tw == 1:
                            weight1 |= (u == u0) & (uu == uu0) & (v == v0) & (vv == vv0)
                        elif tw == 2:
                            weight2 |= (u == u0) & (uu == uu0) & (v == v0) & (vv == vv0)
                        elif tw == 3:
                            weight3 |= (u == u0) & (uu == uu0) & (v == v0) & (vv == vv0)
                        elif tw == 4:
                            weight4 |= (u == u0) & (uu == uu0) & (v == v0) & (vv == vv0)
                        elif tw == 5:
                            weight5 |= (u == u0) & (uu == uu0) & (v == v0) & (vv == vv0)
                        elif tw == 6:
                            weight6 |= (u == u0) & (uu == uu0) & (v == v0) & (vv == vv0)
                        elif tw == 7:
                            weight7 |= (u == u0) & (uu == uu0) & (v == v0) & (vv == vv0)


            btor.Assert(weight0 | weight1 | weight2 | weight3 | weight4 | weight5 | weight6 | weight7)
            return btor.Cond(
                weight1, btor.Const(1, state_words),
                btor.Cond(
                    weight2, btor.Const(2, state_words),
                    btor.Cond(
                        weight3, btor.Const(3, state_words),
                        btor.Cond(
                            weight4, btor.Const(4, state_words),
                            btor.Cond(
                                weight5, btor.Const(5, state_words),
                                btor.Cond(
                                    weight6, btor.Const(6, state_words),
                                    btor.Cond(
                                        weight7, btor.Const(7, state_words),
                                        btor.Const(0, state_words)
                                    )
                                )
                            )
                        )
                    )
                )
            )

        def set_fixed_difference(state, difference):
            for i in range(state_bits):
                btor.Assert(state[i] == ((difference >> i) & 0x1))

        # cost = btor.Const(0, state_words)
        cost_upper = btor.Const(0, state_words)
        cost_lower = btor.Const(0, state_words)
        cost_em = btor.Const(0, state_words)
        # the upper
        for i in range(differential_characteristic_rounds_upper):
            for j in range(0, state_bits, sbox_bits):
                uu0 = u0_a[i][j + sbox_bits - 1:j]
                vv0 = u0_b[i][j + sbox_bits - 1:j]
                uu1 = u1_a[i][j + sbox_bits - 1:j]
                vv1 = u1_b[i][j + sbox_bits - 1:j]
                delta_in_1 = (diff_route[4 * i] >> j) & (2 ** sbox_bits - 1)
                delta_out_1 = (diff_route[4 * i + 1] >> j) & (2 ** sbox_bits - 1)
                small_quasi_ddt = get_small_quasi_ddt_by_u1_and_v1(quasi_ddt, delta_in_1, delta_out_1)
                small_quasi_ddt_weight_lists_w_0_to_7 = get_small_quasi_tables_weight_lists_by_u_and_v(small_quasi_ddt, 1)
                w = get_one_words_weight_by_small_quasi_table(uu0, uu1, vv0, vv1, small_quasi_ddt_weight_lists_w_0_to_7)
                cost_upper += w
            xor_k_mask(u0_b[i], u0_c[i])
            xor_k_mask(u1_b[i], u1_c[i])
            permute_bits_sr(u0_c[i], u0_d[i])
            permute_bits_sr(u1_c[i], u1_d[i])
            mix_t_columns(u0_d[i], u0_a[i + 1])
            mix_t_columns(u1_d[i], u1_a[i + 1])

        # em
        for j in range(0, state_bits, sbox_bits):
            uu0 = u0_a[differential_characteristic_rounds_upper][j + sbox_bits - 1:j]
            uu1 = u1_a[differential_characteristic_rounds_upper][j + sbox_bits - 1:j]
            delta_in_1 = (diff_route[4 * differential_characteristic_rounds_upper] >> j) & (2 ** sbox_bits - 1)
            delta_out_2 = (diff_route[4 * differential_characteristic_rounds_upper + 1] >> j) & (2 ** sbox_bits - 1)
            vv0 = u0_b[differential_characteristic_rounds_upper][j + sbox_bits - 1:j]
            vv1 = u1_b[differential_characteristic_rounds_upper][j + sbox_bits - 1:j]
            small_quasi_bct = get_small_quasi_bct_by_u1_and_v2(quasi_bct, delta_in_1, delta_out_2)
            small_quasi_bct_weight_lists_w_0_to_7 = get_small_quasi_tables_weight_lists_by_u_and_v(small_quasi_bct, 0)
            ww = get_one_words_weight_by_small_quasi_table(uu0, uu1, vv0, vv1, small_quasi_bct_weight_lists_w_0_to_7)
            cost_em += ww

        # the upper
        for i in range(differential_characteristic_rounds_upper, differential_characteristic_rounds_upper + differential_characteristic_rounds_lower):
            if i != differential_characteristic_rounds_upper:
                for j in range(0, state_bits, sbox_bits):
                    uu0 = u0_a[i][j + sbox_bits - 1:j]
                    vv0 = u0_b[i][j + sbox_bits - 1:j]
                    uu1 = u1_a[i][j + sbox_bits - 1:j]
                    vv1 = u1_b[i][j + sbox_bits - 1:j]
                    delta_in_1 = (diff_route[4 * i] >> j) & (2 ** sbox_bits - 1)
                    delta_out_1 = (diff_route[4 * i + 1] >> j) & (2 ** sbox_bits - 1)
                    small_quasi_ddt = get_small_quasi_ddt_by_u1_and_v1(quasi_ddt, delta_in_1, delta_out_1)
                    small_quasi_ddt_weight_lists_w_0_to_7 = get_small_quasi_tables_weight_lists_by_u_and_v(small_quasi_ddt, 1)
                    w = get_one_words_weight_by_small_quasi_table(uu0, uu1, vv0, vv1, small_quasi_ddt_weight_lists_w_0_to_7)
                    cost_lower += w
            xor_k_mask(u0_b[i], u0_c[i])
            xor_k_mask(u1_b[i], u1_c[i])
            permute_bits_sr(u0_c[i], u0_d[i])
            permute_bits_sr(u1_c[i], u1_d[i])
            mix_t_columns(u0_d[i], u0_a[i + 1])
            mix_t_columns(u1_d[i], u1_a[i + 1])

        cost = cost_upper + cost_em + cost_lower

        set_fixed_difference(u0_a[0], 0x0)
        set_fixed_difference(u1_a[0], 0x0)
        set_fixed_difference(u0_a[differential_characteristic_rounds_upper + differential_characteristic_rounds_lower], 0x0)
        set_fixed_difference(u1_a[differential_characteristic_rounds_upper + differential_characteristic_rounds_lower], 0x0)
        btor.Set_opt(pyboolector.BTOR_OPT_INCREMENTAL, 1)


        print("# differential : 0x{:016x} -> 0x{:016x}".format(diff_route[0], diff_route[len(diff_route) - 1]))

        differentials = []
        for target in range(min_weight, max_weight):
            previous = []
            print("# Solution: of weight {}".format(target), file=f)
            print("[", file=f)

            count_negative = 0
            count_positive = 0

            while True:
                btor.Assume(cost == target)
                distinct = btor.Const(1)
                for _, uuu0_a, uuu1_a in previous:
                    temp = btor.Const(0)
                    for i in range(1, differential_characteristic_rounds_upper + differential_characteristic_rounds_lower):
                        temp |= (u0_a[i] != btor.Const(uuu0_a[i], state_bits))
                        temp |= (u1_a[i] != btor.Const(uuu1_a[i], state_bits))
                    distinct &= temp
                btor.Assume(distinct)

                r = btor.Sat()
                if r == btor.SAT:
                    print("    # all: {}".format(int(cost.assignment, base=2)), file=f)
                    print("    # Solution: [#{} of weight {}]".format(len(previous) + 1, target), file=f)
                    print("    [", file=f)

                    # upper
                    route_u0 = []
                    route_u1 = []
                    for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower):
                        uu0_a = int(u0_a[i].assignment, base=2)
                        uu1_a = int(u1_a[i].assignment, base=2)
                        uu0_b = int(u0_b[i].assignment, base=2)
                        uu1_b = int(u1_b[i].assignment, base=2)
                        uu0_c = int(u0_c[i].assignment, base=2)
                        uu1_c = int(u1_c[i].assignment, base=2)
                        uu0_d = int(u0_d[i].assignment, base=2)
                        uu1_d = int(u1_d[i].assignment, base=2)

                        if i < differential_characteristic_rounds_upper:
                            if i == 0:
                                print("     # upper: {}".format(int(cost_upper.assignment, base=2)), file=f)

                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_a, diff_route[4 * i], uu1_a), file=f)
                            print("     # after sb", file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_b, diff_route[4 * i + 1], uu1_b), file=f)
                            print("     # after k", file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_c, diff_route[4 * i + 2], uu1_c), file=f)
                            print("     # after sr", file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_d, diff_route[4 * i + 3], uu1_d), file=f)
                            print("     # after mc", file=f)
                        elif i == differential_characteristic_rounds_upper:
                            print("", file=f)
                            print("     # em: {}".format(int(cost_em.assignment, base=2)), file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_a, diff_route[4 * i], uu1_a), file=f)
                            print("     # after sb", file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_b, uu1_b, diff_route[4 * i + 1]), file=f)

                            print("", file=f)
                            print("     # lower: {}".format(int(cost_lower.assignment, base=2)), file=f)
                            print("     # after k", file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_c, uu1_c, diff_route[4 * i + 2]), file=f)
                            print("     # after sr", file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_d, uu1_d, diff_route[4 * i + 3]), file=f)
                            print("     # after mc", file=f)
                        else:
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_a, uu1_a, diff_route[4 * i]), file=f)
                            print("     # after sb", file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_b, uu1_b, diff_route[4 * i + 1]), file=f)
                            print("     # after k", file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_c, uu1_c, diff_route[4 * i + 2]), file=f)
                            print("     # after sr", file=f)
                            print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(uu0_d, uu1_d, diff_route[4 * i + 3]), file=f)
                            print("     # after mc", file=f)

                        route_u0.append(uu0_a)
                        route_u0.append(uu0_b)
                        route_u0.append(uu0_c)
                        route_u0.append(uu0_d)
                        route_u1.append(uu1_a)
                        route_u1.append(uu1_b)
                        route_u1.append(uu1_c)
                        route_u1.append(uu1_d)
                    u0_a_final = int(u0_a[differential_characteristic_rounds_upper + differential_characteristic_rounds_lower].assignment, base=2)
                    u1_a_final = int(u1_a[differential_characteristic_rounds_upper + differential_characteristic_rounds_lower].assignment, base=2)
                    print("", file=f)
                    print("     [0x{:016x}, 0x{:016x}, 0x{:016x}],".format(u0_a_final, u1_a_final, diff_route[4 * (differential_characteristic_rounds_upper + differential_characteristic_rounds_lower)]), file=f)
                    route_u0.append(u0_a_final)
                    route_u1.append(u1_a_final)

                    s_route = compute_sign(route_u0, route_u1, diff_route, differential_characteristic_rounds_upper, differential_characteristic_rounds_lower)
                    if s_route < 0:
                        count_negative += 1
                    else:
                        count_positive += 1

                    print("    ],", file=f)
                    previous.append((s_route, [u0_a[i].assignment for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower + 1)], [u1_a[i].assignment for i in range(differential_characteristic_rounds_upper + differential_characteristic_rounds_lower + 1)]))
                else:
                    print("     # No trails with weight equal to {}.".format(target), file=f)
                    print("     # sign = -1: {}".format(count_negative), file=f)
                    print("     # sign = +1: {}".format(count_positive), file=f)
                    break
            print("],", file=f)

        return differentials

    filename = "result/quasi_bcs_search_route_{}_upper_{}_lower_{}_w_{}_to_{}.txt".format(route_number, differential_characteristic_rounds_upper, differential_characteristic_rounds_lower, min_weight, max_weight)
    f = open(filename, "w")
    print("# --------------------------------- route {} ---------------------------------------".format(route_number),
          file=f)
    print("routes = [".format(), file=f)
    differentials = search_differential(delta_in, delta_out, f)
    print("],", file=f)
    print("", file=f)
    f.close()
    return differentials

min_weight = 0
max_weight = 30

delta_in = 0x0010001000000010
delta_out = 0x0300000003000000
diff_k = [
    0x0000000000c00000,
    0x0000000000000300
]

import diff_routes_fixed_beta_1_to_3 as routes
diff_routes = routes.routes
print(len(diff_routes))

pool_size = 10
pool = Pool(pool_size)
begin_number = 0
end_number = begin_number + pool_size
for route_number in range(begin_number, end_number):
    diff_route = diff_routes[route_number]
    pool.apply_async(get_quasidifferentials_by_one_fixed_input_and_output, (diff_route, diff_k, route_number, min_weight, max_weight))

pool.close()
pool.join()