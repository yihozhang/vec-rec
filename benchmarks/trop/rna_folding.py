# N(i,j) = max( N(i+1,j), N(i,j-1), N(i+1,j-1) + s(i,j), max_{m=i+1..j-1}( N(i,m) + N(m+1,j) ) )
#
# Define A(j, k) = N(j-k, j).
#
# Let i = j - k, then
#
#   N(i + 1, j) = A(j, k - 1)
#   N(i, j - 1) = A(j - 1, k - 1)
#   N(i + 1, j - 1) = A(j - 1, k - 2)
#   s(i, j) = s(j - k, j)
#
# Let m=i+t=j−k+t, where t=1..k-1, then
#   N(i, m) = N(j - k, j - k + t) = A(j - k + t, t)
#   N(m + 1, j) = N(j - k + t + 1, j) = A(j, k - t - 1)

# So A(j, k) = max { A(j, k - 1), A(j - 1, k - 1), A(j - 1, k - 2) + s(j - k, j), max_{t=1..k-1}( A(j - k + t, t) + A(j, k - t - 1) ) }
