# M(i,j) = max(M(i-1,j-1), I(i-1,j-1), D(i-1,j-1)) + s(i,j);
# I(i,j) = max(M(i-1,j) + g_o + g_e, I(i-1,j) + g_e);
# D(i,j) = max(M(i,j-1) + g_o + g_e, D(i,j-1) + g_e)
