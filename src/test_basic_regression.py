import contextlib

from src.regression import MyLinearRegression


def get_percent_str(part, total):
    return str(round(part / total * 100, 2)) + " %"


def simple_initial(debug=True):
    reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price', debug)

    if debug:
        print("Number of records: " + str(reg.get_size()) + '\n')
        print("Head:\n" + reg.head() + '\n')
        print("Describe:\n" + reg.describe() + '\n')
        print("Num null values: " + str(reg.get_num_rows_with_null_vals()) + ', ' +
              get_percent_str(reg.get_num_rows_with_null_vals(), reg.get_size()) + '\n')

    features_orig = reg.get_features()
    if debug:
        print("Features before dropping (" + str(len(features_orig)) + "):\n" + str(features_orig) + '\n')

    # reg.drop_features(features[3:5])

    features = reg.get_features()
    if debug:
        print("\n-------------------------------------------------------\nAFTER dropping features:")
        print("Features after dropping (" + str(len(features)) + "):\n" + str(features) + '\n')
        print("Head:\n" + reg.head() + '\n')
        print("Describe:\n" + reg.describe() + '\n')

    if debug:
        print("Rows before dropping null values: " + str(reg.get_size()) + ", "
              + get_percent_str(reg.get_size(), reg.get_initial_size()) + '\n')
    reg.drop_null_rows()
    if debug:
        print("Rows after dropping null values: " + str(reg.get_size()) + ", "
              + get_percent_str(reg.get_size(), reg.get_initial_size()) + '\n')

    reg.remove_outliers_low_fraction('Price', .01)
    if debug:
        print("Rows afer removing outliers low: " + str(reg.get_size()) + ", "
              + get_percent_str(reg.get_size(), reg.get_initial_size()) + '\n')
    reg.remove_outliers_high_fraction('Price', .01)
    if debug:
        print("Rows afer removing outliers high: " + str(reg.get_size()) + ", "
              + get_percent_str(reg.get_size(), reg.get_initial_size()) + '\n')

    if debug:
        print("\n-------------------------------------------------------\nAFTER REMOVING ROWS:")
        print("Number of records: " + str(reg.get_size()) + '\n')
        print("Head:\n" + reg.head() + '\n')
        print("Describe:\n" + reg.describe() + '\n')

    # reg.display_dist('Price')

    reg.do_log_on_dependent()

    if debug:
        print("\n-------------------------------------------------------\nAFTER LOG on Dependent:")
        print("Head:\n" + reg.head() + '\n')
        print("Describe:\n" + reg.describe() + '\n')

    # reg.display_dist('Price')

    if debug:
        print("VIFs: \n" + reg.get_vif(['Mileage', 'Year', 'EngineV']))

    reg.add_dummies()

    if debug:
        print("\n-------------------------------------------------------\nAFTER Adding dummies:")
        print("Head:\n" + reg.head() + '\n')
        print("Describe:\n" + reg.describe() + '\n')

    results_dic = reg.do_actual_regression_part()

    if debug:
        print("\n-------------------------------------------------------\nAFTER Adding regression:")
        print(f"Coefficients summary:\n{results_dic['Coef summary']}")
        print(f"Differences summary: \n{results_dic['Diff summary']}")
        print(f"R2: {results_dic['R2']}, Difference mean: {results_dic['Diff mean']}, "
              f"Difference STD: {results_dic['Diff STD']}")

    return results_dic


def regression_same_as_lecture():
    reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price')
    reg.drop_features(['Model'])
    reg.drop_null_rows()
    reg.remove_outliers_high_fraction('Price', .01)
    reg.remove_outliers_high_fraction('Mileage', .01)
    reg.remove_outliers_high_num('EngineV', 6.5)
    reg.remove_outliers_low_fraction('Year', 0.01)
    reg.do_log_on_dependent()
    reg.drop_features(['Year'])
    reg.add_dummies()

    return reg.do_actual_regression_part()


def test_basics():
    # Running with debug to make sure all functions are used, but turning off the actual stdout
    with contextlib.redirect_stdout(None):
        results_dic = simple_initial(debug=True)
    # R2: 0.936, pred_diff_percent_mean: 27.77, pred_diff_percent_std: 130.03
    assert (results_dic['R2'] == 0.936)
    assert (results_dic['Diff mean'] == 27.77)
    assert (results_dic['Diff STD'] == 130.03)

    results_dic = simple_initial(debug=False)
    # R2: 0.936, pred_diff_percent_mean: 27.77, pred_diff_percent_std: 130.03
    assert (results_dic['R2'] == 0.936)
    assert (results_dic['Diff mean'] == 27.77)
    assert (results_dic['Diff STD'] == 130.03)

    results_dic = regression_same_as_lecture()
    assert (results_dic['R2'] == 0.745)
    assert (results_dic['Diff mean'] == 36.26)
    assert (results_dic['Diff STD'] == 55.07)


def test_regular_regression_func():
    # same as lecture, but order is different, so results slightly different:
    # R2: 0.726, pred_diff_percent_mean: 43.04, pred_diff_percent_std: 111.73
    reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price')
    input_dic = {'Features to drop': ['Model', 'Year']}
    results_dic = reg.do_linear_regression(input_dic)

    assert (results_dic['R2'] == 0.726)
    assert (results_dic['Diff mean'] == 43.04)
    assert (results_dic['Diff STD'] == 111.73)
