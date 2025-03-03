
import numpy
# from sage.all import *
import round_key_transition
tk1s_64 = round_key_transition.tk1s_64
tk2s_64 = round_key_transition.tk2s_64
tk3s_64 = round_key_transition.tk3s_64

tk_number = 1

SKINNY_64_SBOX = [0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf]

n = 4
m = 4
sbox = SKINNY_64_SBOX
state_bits = 64
state_words = 16
sbox_bits = 4

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
                        pp_tk1_list = tk1s_64[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk1_list:
                            one_key_expression_tk1[pp] ^= 1
                    elif tk_number == 2:
                        pp_tk1_list = tk1s_64[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk1_list:
                            one_key_expression_tk1[pp] ^= 1
                        pp_tk2_list = tk2s_64[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk2_list:
                            one_key_expression_tk2[pp] ^= 1
                    elif tk_number == 3:
                        pp_tk1_list = tk1s_64[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk1_list:
                            one_key_expression_tk1[pp] ^= 1
                        pp_tk2_list = tk2s_64[round + key_recovery_round_top][i * sbox_bits + j]
                        for pp in pp_tk2_list:
                            one_key_expression_tk2[pp] ^= 1
                        pp_tk3_list = tk3s_64[round + key_recovery_round_top][i * sbox_bits + j]
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


import differential_p_46 as differentials
routes = differentials.routes
print(len(routes))
diff = routes[0]
import quasidifferential_p_46 as quasidifferentials
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
condition_bits = 2
for i in range(2 ** condition_bits):
    c0 = (i >> 0) & 0x1
    c1 = (i >> 1) & 0x1
    basis = [
        [[13],],
        [[15],],
    ]
    basis_values = [c0, c1,]
    print()
    print("the {}-th basis : {}".format(i, basis))
    all = count_valid_by_given_basis(key_list1, basis, basis_values)
    if all != 0:
        count_non_zero.append([i, all])
print()
print("--------------------------------")
print(count_non_zero)

