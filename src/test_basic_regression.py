import contextlib

from src.regression import MyLinearRegression


def simple_initial(debug=True):
    reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price', debug)

    if debug:
        print("Number of records: " + str(reg._get_size()) + '\n')
        print("Head:\n" + reg._head() + '\n')
        print("Describe:\n" + reg._describe() + '\n')
        print(f"Num null values: {str(reg._get_num_rows_with_null_vals())}, "
              f"{round(reg._get_num_rows_with_null_vals() * 100 / reg._get_size(),2)}\n")

    if debug:
        features_after = reg._get_features()
        print("Features before dropping (" + str(len(features_after)) + "):\n" + str(features_after) + '\n')

    # reg._drop_features(features[3:5])

    features = reg._get_features()
    if debug:
        print("\n-------------------------------------------------------\nAFTER dropping features:")
        print("Features after dropping (" + str(len(features)) + "):\n" + str(features) + '\n')
        print("Head:\n" + reg._head() + '\n')
        print("Describe:\n" + reg._describe() + '\n')

    if debug:
        print(f"Rows before dropping null values: {str(reg._get_size())}, {reg._get_left_rows_percent_str()}\n")
    reg._drop_null_rows()
    if debug:
        print(f"Rows after dropping null values: {str(reg._get_size())}, {reg._get_left_rows_percent_str()}\n")

    reg._remove_outliers_low_fraction('Price', .01)
    if debug:
        print(f"Rows afer removing outliers low: {str(reg._get_size())}, {reg._get_left_rows_percent_str()}\n")

    reg._remove_outliers_high_fraction('Price', .01)
    if debug:
        print(f"Rows afer removing outliers high: {str(reg._get_size())}, {reg._get_left_rows_percent_str()}\n")

    if debug:
        print("\n-------------------------------------------------------\nAFTER REMOVING ROWS:")
        print("Number of records: " + str(reg._get_size()) + '\n')
        print("Head:\n" + reg._head() + '\n')
        print("Describe:\n" + reg._describe() + '\n')

    # reg._display_dist('Price')

    reg._do_log_on_dependent()

    if debug:
        print("\n-------------------------------------------------------\nAFTER LOG on Dependent:")
        print("Head:\n" + reg._head() + '\n')
        print("Describe:\n" + reg._describe() + '\n')

    # reg._display_dist('Price')

    if debug:
        print("VIFs: \n" + reg._get_vif(['Mileage', 'Year', 'EngineV']))

    reg._add_dummies()

    if debug:
        print("\n-------------------------------------------------------\nAFTER Adding dummies:")
        print("Head:\n" + reg._head() + '\n')
        print("Describe:\n" + reg._describe() + '\n')

    results_dic = reg._do_actual_regression_part({})

    if debug:
        print("\n-------------------------------------------------------\nAFTER Adding regression:")
        print(f"Coefficients summary:\n{results_dic['Coef summary']}")
        print(f"Differences summary: \n{results_dic['Diff summary']}")
        print(f"R2: {results_dic['R2']}, Difference mean: {results_dic['Diff mean']}, "
              f"Difference STD: {results_dic['Diff STD']}")

    return results_dic


def regression_same_as_lecture():
    reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price')
    reg._drop_features(['Model'])
    reg._drop_null_rows()
    reg._remove_outliers_high_fraction('Price', .01)
    reg._remove_outliers_high_fraction('Mileage', .01)
    reg._remove_outliers_high_num('EngineV', 6.5)
    reg._remove_outliers_low_fraction('Year', 0.01)
    reg._do_log_on_dependent()
    reg._drop_features(['Year'])
    reg._add_dummies()

    return reg._do_actual_regression_part({})


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


def test_regular_regression_func(debug=False):
    # same as lecture, but order is different, so results slightly different:
    # R2: 0.726, pred_diff_percent_mean: 43.04, pred_diff_percent_std: 111.73
    reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price', debug)
    input_dic = {'Features to drop': ['Model', 'Year'],
                 'Remove outliers': [('Price', 'high', .001),
                                     ('Price', 'low', 0),
                                     ('EngineV', 'high', 0),
                                     ('EngineV', 'low', 0),
                                     ('Mileage', 'high', 0.01),
                                     ('Mileage', 'low', 0),
                                     ('Year', 'high', 0),
                                     ('Year', 'low', 0.01)
                                     ]
                 }
    results_dic = reg.do_linear_regression(input_dic)

    #print(f"{results_dic['R2']} {results_dic['Diff mean']} {results_dic['Diff STD']}")
    assert (results_dic['R2'] == 0.742)
    assert (results_dic['Diff mean'] == 47.58)
    assert (results_dic['Diff STD'] == 119.2)
