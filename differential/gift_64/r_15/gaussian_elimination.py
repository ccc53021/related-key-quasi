
from sage.all import *

# 高斯消元 sage版
from sympy import Matrix
def get_rank_and_base_master_key(key_list, key_list1, state_bits):
    km = MatrixSpace(GF(2), len(key_list), state_bits)
    km1 = MatrixSpace(GF(2), len(key_list1), state_bits + 1)

    key_matrix = km.matrix(Matrix(key_list))
    key_matrix1 = km1.matrix(Matrix(key_list1))

    rank = key_matrix.rank()
    rank1 = key_matrix1.rank()
    print("# rank = {}, rank1 = {}".format(rank, rank1))

    # key_matrix_rref = key_matrix1.echelon_form()
    key_matrix_rref = key_matrix1.rref()
    key_matrix_rref_list = list(key_matrix_rref)
    key_rref_str = []
    key_space = []
    one_trail_key_bit = state_bits
    for r in range(rank1):
        one_trail_rref_str = ''
        one_key_linear_contraint = []
        for i in range(one_trail_key_bit):
            if key_matrix_rref_list[r][i] != 0:
                word = i // 16
                position = i % 16
                one_trail_rref_str += "k" + str(word) + "[" + str(position) + "] + "
                one_key_linear_contraint.append([word, position])
        one_trail_rref_str += " = " + str(key_matrix_rref_list[r][one_trail_key_bit])
        one_key_linear_contraint.append(key_matrix_rref_list[r][one_trail_key_bit])

        key_rref_str.append(one_trail_rref_str)
        key_space.append(one_key_linear_contraint)
    for k in range(len(key_rref_str)):
        print("# " + key_rref_str[k])

    return rank, rank1, key_rref_str, key_space