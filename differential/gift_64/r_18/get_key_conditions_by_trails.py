
import numpy
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

def get_master_key_by_round_key(round, round_position, key_recovery_round_top):
    true_round = round + key_recovery_round_top

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

def key_conditions_by_one_trail(diff, trail, key_recovery_round_top):
    differential_characteristic_rounds = (len(diff) - 1) // 3
    one_trail_key_bit = key_bits
    one_key_expression = [0] * one_trail_key_bit
    correlation = 1
    for round in range(differential_characteristic_rounds):
        # sb
        one_round_difference_input = diff[3 * round]
        one_round_difference_s = diff[3 * round + 1]
        one_round_mask_input = trail[2 * round]
        one_round_mask_s = trail[2 * round + 1]

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
                    if u != 0:
                        for k in range(sbox_bits):
                            if state_bits == 128:
                                if k != 0 and k != 3:
                                    key_bit = (u >> k) & 0x1
                                    if key_bit == 1:
                                        round_key_full = round * state_bits + (sbox_bits * i + k)
                                        round_key_round = round_key_full // state_bits
                                        round_key_position = round_key_full % state_bits
                                        master_key = get_master_key_by_round_key(round_key_round,
                                                                                 round_key_position,
                                                                                 key_recovery_round_top)
                                        master_key_word = master_key[0]
                                        master_key_position = master_key[1]
                                        master_key_full = master_key_word * 16 + master_key_position
                                        one_key_expression[master_key_full] = 1
                            elif state_bits == 64:
                                if k != 2 and k != 3:
                                    key_bit = (u >> k) & 0x1
                                    if key_bit == 1:
                                        round_key_full = round * state_bits + (sbox_bits * i + k)
                                        round_key_round = round_key_full // state_bits
                                        round_key_position = round_key_full % state_bits
                                        master_key = get_master_key_by_round_key(round_key_round,
                                                                                 round_key_position,
                                                                                 key_recovery_round_top)
                                        master_key_word = master_key[0]
                                        master_key_position = master_key[1]
                                        master_key_full = master_key_word * 16 + master_key_position
                                        one_key_expression[master_key_full] = 1


    key_str = ""
    for i in range(len(one_key_expression)):
        if one_key_expression[i] == 1:
            word = i // 16
            position = i % 16
            key_str += "k{}_{} +".format(word, position)
    one_key_expression1 = one_key_expression[::]
    if correlation < 0:
        one_key_expression1.append(1)
        key_str += " = 1"
    else:
        one_key_expression1.append(0)
        key_str += " = 0"

    return one_key_expression, one_key_expression1, key_str

def count_valid_by_given_basis(key_list1, basis, basis_values):
    key_0 = [0] * (key_bits + 1)
    key_1 = [0] * (key_bits + 1)
    key_1[key_bits] = 1
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
            trails_mask_0_sign_0.append(i)
        elif one_key == key_1:
            count_mask_0_sign_1 += 1
            trails_mask_0_sign_1.append(i)
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


import differential_p_58 as differentials
routes = differentials.routes
print(len(routes))
diff = routes[0]
import quasidifferential_p_58 as quasidifferentials
trails = quasidifferentials.routes[0]
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
# rank, rank1, key_rref_str, key_space = gaussian_elimination.get_rank_and_base_master_key(key_list, key_list1, key_bits)

count_non_zero = []
for i in range(2 ** 8):
    c0 = i & 0x1
    c1 = (i >> 1) & 0x1
    c2 = (i >> 2) & 0x1
    c3 = (i >> 3) & 0x1
    c4 = (i >> 4) & 0x1
    c5 = (i >> 5) & 0x1
    c6 = (i >> 6) & 0x1
    c7 = (i >> 7) & 0x1

    basis = [
        [[0, 4], [0, 6], [2, 8], [2, 15], [5, 7], [5, 9], [7, 3], [7, 11]],
        [[0, 5], [0, 6], [2, 8], [5, 7], [5, 9], [7, 11]],
        [[2, 0], [2, 8]],
        [[2, 7], [2, 15]],
        [[3, 3]],
        [[3, 11]],
        [[6, 1]],
        [[6, 6], [6, 14]],

        [[0, 4], [0, 5], [2, 15], [7, 3]],
        [[0, 4], [0, 6], [2, 0], [2, 15], [5, 7], [5, 9], [7, 3], [7, 11]],
        [[0, 5], [0, 6], [2, 0], [5, 7], [5, 9], [7, 11]],
        [[0, 4], [0, 6], [2, 7], [2, 8], [5, 7], [5, 9], [7, 3], [7, 11]],
        [[0, 4], [0, 6], [2, 0], [2, 7], [5, 7], [5, 9], [7, 3], [7, 11]],
        [[0, 4], [0, 5], [2, 7], [7, 3]],
        [[0, 4], [0, 5], [2, 0], [2, 7], [2, 8], [7, 3]],
    ]
    basis_values = [c0, c1, c2, c3, c4, c5, c6, c7, c0^c1, c0^c2, c1^c2, c0^c3, c0^c2^c3, c0^c1^c3, c0^c1^c2^c3]
    print()
    print("the {}-th basis : {}".format(i, basis))
    all = count_valid_by_given_basis(key_list1, basis, basis_values)
    if all != 0:
        count_non_zero.append(["{:08b}".format(i), all])
print()
print("--------------------------------")
print(count_non_zero)

