from src.optimizations import optimize
from src.test_basic_regression import test_basics
from src.test_basic_regression import test_regular_regression_func

# to make sure basics are not broken, and when need to debug specific regression
test_basics()
test_regular_regression_func()
optimize()
