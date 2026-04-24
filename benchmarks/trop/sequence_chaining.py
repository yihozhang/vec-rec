# C(i) = max(w_i, max{j<i} (C(j) + w_i - s(i,j)))

# C(i) = max(w_i, D(i, j))
# D(i, j) = max{j<i} (C(j) + w(i) - s(i,j))
#         = max(D(i, j-1), C(j) + w_i - s(i, j))
