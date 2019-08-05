from src.optimizations import optimize
from src.optimizations import optimize_different_split_randoms
from src.test_basic_regression import test_basics
from src.test_basic_regression import test_regular_regression_func

# to make sure basics are not broken, and when need to debug specific regression
test_basics()
test_regular_regression_func(debug=False)
optimize_different_split_randoms() # do a few times to compare to random 365 done in the lecture
optimize()  # do similar as done in the lecture

