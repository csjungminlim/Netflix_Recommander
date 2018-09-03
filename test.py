# -*- coding: utf-8 -*-

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.stats import uniform
import numpy as np
import SVD
from SGD import MF
from Find_Neighbors import Find_Neighbors

def construct_sparse_matrix():

    #/Users/limjungmin/Netflix_Recommender/netflix-prize-data/combined_data_1~4.txt
    data_file = open("/Users/limjungmin/Netflix_Recommender/netflix-prize-data/combined_data_1.txt")
    movie_row = []
    movie_column = []
    data = []
    already_used = []
    user_index = {}
    col_index = 0
    printcounter = 0
    check_example = 0

    for line in data_file:
        if check_example == 1:
            break
        if not line.strip():                            # 빈 줄 건너뛰기
            continue
        if ":" in line:                                 # movie 구분점
            printcounter += 1
            print printcounter
            length = line.__len__() - 2
            movieID = line[:length].rstrip()            # "1:" 경우 1, "234:" 경우 234
            movie_index = eval(movieID) - 1             # indexing 처리
            if eval(movieID) == 10:                     # 20번째 영화에서 정지(testing 때문)
                check_example = 1

        else:
            userID = line.partition(",")
            rating = userID[2].partition(",")[0]
            if userID[0] not in user_index.keys():      # 사용자 ID 중복이 아니라면,
                user_index[userID[0]] = col_index
                movie_row.append(movie_index)
                movie_column.append(col_index)
                data.append(eval(rating))
                col_index += 1

            else:
                x = user_index[userID[0]]               # 기존에 존재하는 사용자라면,
                movie_row.append(movie_index)
                movie_column.append(x)
                data.append(eval(rating))

if __name__ == '__main__':
    sparse_matrix = construct_sparse_matrix()
