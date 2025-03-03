
import math

import numpy
# from sage.all import *
import round_key_transition
rks = round_key_transition.key_recover_bit_from_top_to_bottom_64

GIFT_SBOX = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
n = 4
m = 4
sbox = GIFT_SBOX
state_bits = 64
state_words = 16
sbox_bits = 4
key_bits = 128

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

def get_1d_by_hex(x):
    y = [0] * state_bits
    for i in range(state_bits):
        y[i] = (x >> i) & 0x1

    return y

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

def get_master_key_by_round_key(round, round_position, key_recovery_round_top):
    true_round = round + 1 + key_recovery_round_top

    if state_bits == 128:
        if round_position % 2 == 0:
            true_position = round_position // 2
            master_key = round_key_transition.key_recover_bit_from_top_to_bottom_128[true_round][true_position]
            return master_key
        elif round_position % 2 == 1:
            true_position = (round_position - 1) // 2
            master_key = round_key_transition.key_recover_bit_from_top_to_bottom_128[true_round][true_position]
            return master_key
    elif state_bits == 64:
        if round_position % 2 == 0:
            true_position = round_position // 2
            master_key = round_key_transition.key_recover_bit_from_top_to_bottom_64[true_round][true_position]
            return master_key
        elif round_position % 2 == 1:
            true_position = ((round_position - 1) // 2) + 1
            master_key = round_key_transition.key_recover_bit_from_top_to_bottom_64[true_round][true_position]
            return master_key

def key_conditions_by_one_trail(iii, diff_and_trail, key_recovery_round_top, quasi_ddt, quasi_bct, upper_round, lower_round):
    one_trail_key_bit = key_bits
    one_key_expression = [0] * one_trail_key_bit * 3
    correlation = 1
    for round in range(upper_round + lower_round):
        # sb

        u0 = diff_and_trail[3 * round][0]
        u1 = diff_and_trail[3 * round][1]
        u2 = diff_and_trail[3 * round][2]
        v0 = diff_and_trail[3 * round + 1][0]
        v1 = diff_and_trail[3 * round + 1][1]
        v2 = diff_and_trail[3 * round + 1][2]

        for i in range(0, state_words):
            uu0 = (u0 >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
            uu1 = (u1 >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
            uu2 = (u2 >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
            vv0 = (v0 >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
            vv1 = (v1 >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
            vv2 = (v2 >> (sbox_bits * i)) & (2 ** sbox_bits - 1)

            input = (vv0 << (2 * m)) + (vv1 << m) + vv2
            output = (uu0 << (2 * n)) + (uu1 << m) + uu2

            if round == upper_round:
                c = quasi_bct[input, output]
                correlation *= c
            else:
                c = quasi_ddt[input, output]
                correlation *= c


        k0 = diff_and_trail[3 * round + 2][0]
        k1 = diff_and_trail[3 * round + 2][1]
        k2 = diff_and_trail[3 * round + 2][2]
        for i in range(state_words):
            aa_x = (k0 >> i * sbox_bits) & (2 ** sbox_bits - 1)
            for j in range(sbox_bits):
                if state_bits == 64:
                    if j != 2 and j != 3:
                        bb_x = (aa_x >> j) & 0x1
                        if bb_x == 1:
                            round_key_full = round * state_bits + (sbox_bits * i + j)
                            round_key_round = round_key_full // state_bits
                            round_key_position = round_key_full % state_bits
                            master_key = get_master_key_by_round_key(round_key_round, round_key_position, key_recovery_round_top)
                            master_key_word = master_key[0]
                            master_key_position = master_key[1]
                            master_key_full = master_key_word * 16 + master_key_position
                            one_key_expression[master_key_full] ^= 1

            if round < upper_round:
                aa_upper = (k2 >> i * sbox_bits) & (2 ** sbox_bits - 1)
                for j in range(sbox_bits):
                    if state_bits == 64:
                        if j != 2 and j != 3:
                            bb_upper = (aa_upper >> j) & 0x1
                            if bb_upper == 1:
                                round_key_full = round * state_bits + (sbox_bits * i + j)
                                round_key_round = round_key_full // state_bits
                                round_key_position = round_key_full % state_bits
                                master_key = get_master_key_by_round_key(round_key_round, round_key_position,
                                                                         key_recovery_round_top)
                                master_key_word = master_key[0]
                                master_key_position = master_key[1]
                                master_key_full = master_key_word * 16 + master_key_position
                                one_key_expression[master_key_full + one_trail_key_bit] ^= 1

            else:
               aa_lower = (k1 >> i * sbox_bits) & (2 ** sbox_bits - 1)
               for j in range(sbox_bits):
                   if state_bits == 64:
                       if j != 2 and j != 3:
                           bb_lower = (aa_lower >> j) & 0x1
                           if bb_lower == 1:
                               round_key_full = round * state_bits + (sbox_bits * i + j)
                               round_key_round = round_key_full // state_bits
                               round_key_position = round_key_full % state_bits
                               master_key = get_master_key_by_round_key(round_key_round, round_key_position,
                                                                        key_recovery_round_top)
                               master_key_word = master_key[0]
                               master_key_position = master_key[1]
                               master_key_full = master_key_word * 16 + master_key_position
                               one_key_expression[master_key_full + 2 * one_trail_key_bit] ^= 1


    key_str = ""
    for i in range(one_trail_key_bit):
        if one_key_expression[i] == 1:
            word = i // 16
            position = i % 16
            key_str += "k{}_{} +".format(word, position)
    for i in range(one_trail_key_bit, 2 * one_trail_key_bit):
        if one_key_expression[i] == 1:
            word = (i - one_trail_key_bit) // 16
            position = (i - one_trail_key_bit) % 16
            key_str += "delta_upper_k_{}_{} + ".format(word, position)
    for i in range(2 * one_trail_key_bit, 3 * one_trail_key_bit):
        if one_key_expression[i] == 1:
            word = (i - one_trail_key_bit * 2) // 16
            position = (i - one_trail_key_bit * 2) % 16
            key_str += "delta_lower_k_{}_{} + ".format(word, position)

    one_key_expression1 = one_key_expression[::]
    if correlation < 0:
        one_key_expression1.append(1)
        key_str += " = 1"
    else:
        one_key_expression1.append(0)
        key_str += " = 0"

    return one_key_expression, one_key_expression1, key_str

def count_valid_by_given_basis(key_list1, basis, basis_values):
    key_0 = [0] * (key_bits * 1 + 1)
    key_1 = [0] * (key_bits * 1 + 1)
    key_1[key_bits * 1] = 1
    count_mask_0_sign_0 = 0
    count_mask_0_sign_1 = 0
    count_mask_1_sign_0 = 0
    count_mask_1_sign_1 = 0
    trails_mask_0_sign_0 = []
    trails_mask_0_sign_1 = []
    trails_mask_1_sign_0 = []
    trails_mask_1_sign_1 = []
    for i in range(len(key_list1)):
        one_key = key_list1[i]
        if one_key == key_0:
            count_mask_0_sign_0 += 1
            trails_mask_0_sign_0.append(i + 1)
        elif one_key == key_1:
            count_mask_0_sign_1 += 1
            trails_mask_0_sign_1.append(i + 1)
        else:


            sign = 0
            one_key_list = []
            for j in range(key_bits):
                if one_key[j] == 1:
                    one_key_list.append([j // 16, j % 16])
            for j in range(len(basis)):
                one_basis = basis[j]
                one_value = basis_values[j]
                flag_y = 1
                for ob in one_basis:
                    if ob not in one_key_list:
                        flag_y = 0
                        break
                if flag_y == 1:
                    sign ^= one_value
            sign ^= one_key[key_bits]
            if sign == 0:
                count_mask_1_sign_0 += 1
                trails_mask_1_sign_0.append(i)
            else:
                count_mask_1_sign_1 += 1
                trails_mask_1_sign_1.append(i)

    print("no key , +1 : {}".format(count_mask_0_sign_0))
    print(trails_mask_0_sign_0)
    print("no key , -1 : {}".format(count_mask_0_sign_1))
    print(trails_mask_0_sign_1)
    print("xor key, +1 : {}".format(count_mask_1_sign_0))
    print(trails_mask_1_sign_0)
    print("xor key, -1 : {}".format(count_mask_1_sign_1))
    print(trails_mask_1_sign_1)
    all = count_mask_0_sign_0 + count_mask_1_sign_0 - count_mask_0_sign_1 - count_mask_1_sign_1
    print("all is {}".format(all))

    return all

import quasi_differential_routes as diffs_and_quasis
diffs_and_trails = diffs_and_quasis.routes[0]
print(len(diffs_and_trails))
print(len(diffs_and_trails[0]))

key_recovery_top = 0
upper_round = 1
lower_round = 1
key_list = []
key_list1 = []
for ii in range(25, 26):
    dts_w = diffs_and_trails[ii]
    print("route 0, w {} : {} trails".format(ii, len(dts_w)))
    if len(dts_w) > 0:
        for i in range(len(dts_w)):
            one_diff_and_trail = dts_w[i]
            one_key_expression, one_key_expression1, key_str = key_conditions_by_one_trail(i + 1,
                                                                                           one_diff_and_trail,
                                                                                           key_recovery_top,
                                                                                           quasi_ddt,
                                                                                           quasi_bct, upper_round,
                                                                                           lower_round)
            print("l {} : {}".format(i + 1, key_str))
            key_list.append(one_key_expression)
            key_list1.append(one_key_expression1)
print("all {} trails".format(len(key_list)))

import gaussian_elimination
rank, rank1, key_rref_str, key_space = gaussian_elimination.get_rank_and_base_master_key(key_list, key_list1, key_bits)

count_non_zero = []
count_all = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
condition_bits = 1
for i in range(2 ** 1):
    c0 = (i >> 0) & 0x1
    basis = [
        [[0, 9]],
    ]
    basis_values = [c0]
    print()
    print("the {}-th basis : {}".format(i, basis))
    all = count_valid_by_given_basis(key_list1, basis, basis_values)
    if all > 0:
        dex = math.floor(numpy.log2(all))
        count_all[dex] = count_all[dex] + 1
        count_non_zero.append([i, all])
print()
print("--------------------------------")
print(count_non_zero)
print(count_all)

