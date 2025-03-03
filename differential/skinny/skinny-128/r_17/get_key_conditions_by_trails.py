
import numpy
# from sage.all import *
import round_key_transition
tk1s_128 = round_key_transition.tk1s_128
tk2s_128 = round_key_transition.tk2s_128
tk3s_128 = round_key_transition.tk3s_128

tk_number = 3

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

def get_1d_by_hex(x):
    y = [0] * state_bits
    for i in range(state_bits):
        y[i] = (x >> i) & 0x1

    return y

def key_conditions_by_one_trail(diff, trail, key_recovery_round_top):
    differential_characteristic_rounds = (len(diff) - 1) // 4
    one_trail_key_bit = state_bits
    one_key_expression_tk1 = [0] * one_trail_key_bit
    one_key_expression_tk2 = [0] * one_trail_key_bit
    one_key_expression_tk3 = [0] * one_trail_key_bit

    correlation = 1
    for round in range(differential_characteristic_rounds):
        # sb
        one_round_difference_input = diff[4 * round]
        one_round_difference_s = diff[4 * round + 1]
        one_round_mask_input = trail[3 * round]
        one_round_mask_s = trail[3 * round + 1]

        for i in range(0, state_words):
            a = (one_round_difference_input >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
            b = (one_round_difference_s >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
            u = (one_round_mask_input >> (sbox_bits * i)) & (2 ** sbox_bits - 1)
            v = (one_round_mask_s >> (sbox_bits * i)) & (2 ** sbox_bits - 1)

            if a != 0:
                if u == v == 0:
                    c = DDT[a, b]
                    correlation *= c
                else:
                    c = get_correlation_by_fixed_difference(a, b)[u, v]
                    correlation *= c

        for i in range(8):
            aa = (one_round_mask_s >> i * sbox_bits) & (2 ** sbox_bits - 1)
            for j in range(sbox_bits):
                bb = (aa >> j) & 0x1
                if bb == 1:
                    if tk_number == 1:
                        pp_tk1_list = tk1s_128[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk1_list:
                            one_key_expression_tk1[pp] ^= 1
                    elif tk_number == 2:
                        pp_tk1_list = tk1s_128[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk1_list:
                            one_key_expression_tk1[pp] ^= 1
                        pp_tk2_list = tk2s_128[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk2_list:
                            one_key_expression_tk2[pp] ^= 1
                    elif tk_number == 3:
                        pp_tk1_list = tk1s_128[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk1_list:
                            one_key_expression_tk1[pp] ^= 1
                        pp_tk2_list = tk2s_128[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk2_list:
                            one_key_expression_tk2[pp] ^= 1
                        pp_tk3_list = tk3s_128[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk3_list:
                            one_key_expression_tk3[pp] ^= 1

    key_str = ""
    one_key_expression = []
    if tk_number == 1:
        for i in range(len(one_key_expression_tk1)):
            if one_key_expression_tk1[i] == 1:
                key_str += "tk1_{} + ".format(i)
        one_key_expression = one_key_expression_tk1
    elif tk_number == 2:
        for i in range(len(one_key_expression_tk1)):
            if one_key_expression_tk1[i] == 1:
                key_str += "tk1_{} + ".format(i)
        for i in range(len(one_key_expression_tk2)):
            if one_key_expression_tk2[i] == 1:
                key_str += "tk2_{} + ".format(i)
        one_key_expression = one_key_expression_tk1 + one_key_expression_tk2
    elif tk_number == 3:
        for i in range(len(one_key_expression_tk1)):
            if one_key_expression_tk1[i] == 1:
                key_str += "tk1_{} + ".format(i)
        for i in range(len(one_key_expression_tk2)):
            if one_key_expression_tk2[i] == 1:
                key_str += "tk2_{} + ".format(i)
        for i in range(len(one_key_expression_tk3)):
            if one_key_expression_tk3[i] == 1:
                key_str += "tk3_{} + ".format(i)
        one_key_expression = one_key_expression_tk1 + one_key_expression_tk2 + one_key_expression_tk3
    one_key_expression1 = one_key_expression[::]
    if correlation < 0:
        one_key_expression1.append(1)
        key_str += " = 1"
    else:
        one_key_expression1.append(0)
        key_str += " = 0"

    return one_key_expression, one_key_expression1, key_str

def count_valid_by_given_basis(key_list1, basis, basis_values):
    key_0 = [0] * (tk_number * state_bits + 1)
    key_1 = [0] * (tk_number * state_bits + 1)
    key_1[tk_number * state_bits] = 1
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
            for j in range(tk_number * state_bits):
                if one_key[j] == 1:
                    one_key_list.append(j)
            for j in range(len(basis)):
                one_basis = basis[j]
                one_value = basis_values[j]
                if tk_number == 1:
                    flag_y = 1
                    pp_tk1_list = one_basis[0]
                    for one_key_pp in pp_tk1_list:
                        if one_key_pp not in one_key_list:
                            flag_y = 0
                            break
                    if flag_y == 1:
                        sign ^= one_value
                elif tk_number == 2:
                    flag_y = 1
                    pp_tk1_list = one_basis[0]
                    pp_tk2_list = one_basis[1]
                    for one_key_pp in pp_tk1_list:
                        if one_key_pp not in one_key_list:
                            flag_y = 0
                            break
                    for one_key_pp in pp_tk2_list:
                        if (one_key_pp + state_bits) not in one_key_list:
                            flag_y = 0
                            break
                    if flag_y == 1:
                        sign ^= one_value
                elif tk_number == 3:
                    flag_y = 1
                    pp_tk1_list = one_basis[0]
                    pp_tk2_list = one_basis[1]
                    pp_tk3_list = one_basis[2]
                    for one_key_pp in pp_tk1_list:
                        if one_key_pp not in one_key_list:
                            flag_y = 0
                            break
                    for one_key_pp in pp_tk2_list:
                        if (one_key_pp + state_bits) not in one_key_list:
                            flag_y = 0
                            break
                    for one_key_pp in pp_tk3_list:
                        if (one_key_pp + 2 * state_bits) not in one_key_list:
                            flag_y = 0
                            break
                    if flag_y == 1:
                        sign ^= one_value
            sign ^= one_key[tk_number * state_bits]
            if sign == 0:
                count_mask_1_sign_0 += 1
                trails_mask_1_sign_0.append(i + 1)
            else:
                count_mask_1_sign_1 += 1
                trails_mask_1_sign_1.append(i + 1)

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


import differential_p_110 as differentials
routes = differentials.routes
print(len(routes))
diff = routes[0]
import quasidifferential_p_110 as quasidifferentials
trails = quasidifferentials.trails[0][0]
print(len(trails))

key_recovery_top = 0
key_list = []
key_list1 = []
for i in range(len(trails)):
    trail = trails[i]
    one_key_expression, one_key_expression1, key_str = key_conditions_by_one_trail(diff, trail, key_recovery_top)
    print("l {} : {}".format(i + 1, key_str))
    key_list.append(one_key_expression)
    key_list1.append(one_key_expression1)

# import gaussian_elimination
# rank, rank1, key_rref_str, key_space = gaussian_elimination.get_rank_and_base_master_key(key_list, key_list1, state_bits, tk_number)

count_non_zero = []
condition_bits = 6
for i in range(2 ** condition_bits):
    c0 = (i >> 0) & 0x1
    c1 = (i >> 1) & 0x1
    c2 = (i >> 2) & 0x1
    c3 = (i >> 3) & 0x1
    c4 = (i >> 4) & 0x1
    c5 = (i >> 4) & 0x1
    basis = [
        [[16], [16], [16]],
        [[27], [29, 31], [31]],
        [[35], [37, 39], [39]],
        [[89], [92, 94], [92]],
        [[123], [120], [126]],
        [[127], [126], [120, 126]],

        [[123, 127], [120, 126], [120]]
    ]
    basis_values = [c0, c1, c2, c3, c4, c5, c4 ^ c5]
    print()
    print("the {}-th basis : {}".format(i, basis))
    all = count_valid_by_given_basis(key_list1, basis, basis_values)
    if all != 0:
        count_non_zero.append([i, all])
print()
print("--------------------------------")
print(count_non_zero)

