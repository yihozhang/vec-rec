# The continued fraction thing: One big use of continued fractions is getting a very good 
# rational approximation to a real number. This can be done by just skipping the final division 
# in that second-order IIR conversion trick. Nifty. Actually I guess treating the number as a 
# rational and working through the arithmetic is precisely what gives you the transformation.

# the second-order IIR transformation is known as the wallis-euler recurrence. Lentz's algorithm is 
# more popular, which turns it into a top-down evaluation using a different similar recurrence


# Consider
#   f(x)=a(x) + b(x)/f(x-1)
# Instead of computing this, we can compute
#   h(x) = a(x)*h(x-1)+b(x)*h(x-2)
# and consider
#   h(x)/h(x-1) = { a(x)*h(x-1)+b(x)*h(x-2) } / h(x-1) = a(x) + b(x) * h(x-2) / h(x-1)
# We we take f(x) = h(x)/h(x-1), this recovers the original definition of f
