"""
TODOS by Priority:
* Do comparison of test sample also by R2
* Check what other things can be tweaked / not done
* Make sure that VIF is not too high for the choice
* Make sure reference group is not too small
* Combine Brand and Model
* Remove one of Year / Mileage or at least see what influence it has
* Check conclusions on numerous train/test

Ask questions:
* Is this a good way to check how good the model is: min (1-R2)*(diff.mean + diff.std)?
"""

import itertools
from pprint import pprint

from src.regression import MyLinearRegression


def optimize():
    features = ['Brand', 'Body', 'Mileage', 'EngineV', 'Engine Type', 'Registration', 'Year', 'Model']
    print("All features: " + str(features))

    short_run = False
    special_run = False

    if special_run:
        outliers_dic = {
            'Price high': [0.03],
            'Price low': [0],
            'Mileage high': [.01],
            'Mileage low': [],  # was tried and didn't help
            'Year high': [],
            # decided not to try, need to give prices for newest cars, on the graph there are a lot of samples for 2015
            'Year low': [0.01]
        }

        fractions_price = [0]
        do_log_on_regression = {False}
    elif short_run:
        outliers_dic = {
            'Price high': [.01],
            'Price low': [0],
            'Mileage high': [.01],
            'Mileage low': [],  # was tried and didn't help
            'Year high': [],
            # decided not to try, need to give prices for newest cars, on the graph there are a lot of samples for 2015
            'Year low': [0.01]
        }

        fractions_price = [0]
        do_log_on_regression = {True}
    else:
        outliers_dic = {
            'Price high': [.005, .01, .02],
            'Price low': [0, 0.005],
            'Mileage high': [0.005, .01, .02],
            'Mileage low': [],  # was tried and didn't help
            'Year high': [],  # decided not to try, need to give prices for newest cars, on the graph there are a lot of samples for 2015
            'Year low': [0.005, 0.01, 0.02]
        }

        fractions_price = [0, 0.05, 0.10, 0.15]
        do_log_on_regression = {True, False}

    regs_dic = {}
    input_dic = {}

    for log_regression in do_log_on_regression:
        input_dic['Perform log on dependent'] = log_regression
        for num_features_to_remove in range(len(features)-1):
            for features_to_remove in list(itertools.combinations(features, num_features_to_remove)):
                fractions_price_list = fractions_price if ('Model' not in features_to_remove) else [0]
                for cutoff in fractions_price_list:
                    input_dic['Remove rare categorical'] = [('Model', cutoff)]

                    features_to_remove = tuple(sorted(features_to_remove))
                    input_dic['Features to drop'] = list(features_to_remove)
                    input_dic['Remove outliers'] = [('Price', 'high', .01),
                                                    ('Price', 'low', 0),
                                                    ('Mileage', 'high', 0.01),
                                                    ('Year', 'low', 0.01)
                                                    ]
                    reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price')
                    results_dic = reg.do_linear_regression(input_dic)

                    if (not regs_dic.get((tuple(set(features) - set(features_to_remove)),
                                          tuple(input_dic['Remove outliers']),
                                          cutoff,
                                          log_regression))):
                        regs_dic[(tuple(set(features) - set(features_to_remove)),
                                  tuple(input_dic['Remove outliers']),
                                  cutoff,
                                  log_regression)] = results_dic

                    if results_dic['R2'] < 0.8 or (results_dic['Diff mean'] + results_dic['Diff STD']) > 100:
                        continue

                    for price_high_outlier in outliers_dic['Price high']:
                        for price_low_outlier in outliers_dic['Price low']:
                            for mileage_high_outlier in outliers_dic['Mileage high']:
                                if 'Mileage' in features_to_remove:
                                    continue
                                for year_low_outlier in outliers_dic['Year low']:
                                    if 'Year' in features_to_remove:
                                        continue

                                    reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price')

                                    input_dic['Remove outliers'] = [('Price', 'high', price_high_outlier),
                                                                    ('Price', 'low', price_low_outlier),
                                                                    ('Mileage', 'high', mileage_high_outlier),
                                                                    ('Year', 'low', year_low_outlier)]
                                    results_dic = reg.do_linear_regression(input_dic)
                                    if results_dic['R2'] < 0.8 or (results_dic['Diff mean'] + results_dic['Diff STD']) > 100:
                                        continue
                                    regs_dic[(tuple(set(features)-set(features_to_remove)),
                                              tuple(input_dic['Remove outliers']),
                                              cutoff,
                                              log_regression)] = results_dic

    class results_opt:
        value = 0
        features = None
        outliers = None
        result = None
        cutoff = 0
        log_regression = False

    max_r2_results = results_opt()
    min_mean_plus_std_results = results_opt()
    min_mean_plus_std_results.value = float("inf")
    min_product_results = results_opt()
    min_product_results.value = float("inf")

    for (features_left, outliers, cutoff, log_regression) in regs_dic:
        result = regs_dic[(features_left, outliers, cutoff, log_regression)]
        if result['Diff mean'] + result['Diff STD'] < min_mean_plus_std_results.value:
            min_mean_plus_std_results.value = result['Diff mean'] + result['Diff STD']
            min_mean_plus_std_results.features = features_left
            min_mean_plus_std_results.outliers = outliers
            min_mean_plus_std_results.cutoff = cutoff
            min_mean_plus_std_results.log_regression = log_regression
            min_mean_plus_std_results.result = result
        if result['R2 Adj'] > max_r2_results.value:
            max_r2_results.value = result['R2 Adj']
            max_r2_results.features = features_left
            max_r2_results.outliers = outliers
            max_r2_results.cutoff = cutoff
            max_r2_results.log_regression = log_regression
            max_r2_results.result = result
        product = (1-result['R2 Adj']) * (result['Diff mean'] + result['Diff STD'])
        if product < min_product_results.value:
            min_product_results.value = product
            min_product_results.features = features_left
            min_product_results.outliers = outliers
            min_product_results.cutoff = cutoff
            min_product_results.log_regression = log_regression
            min_product_results.result = result

    print(f"\nmin_mean_plus_std: {round(min_mean_plus_std_results.value, 2)}, "
          f"cutoff: {min_mean_plus_std_results.cutoff}, "
          f"log on Price: {min_mean_plus_std_results.log_regression}, "
          f"features: {min_mean_plus_std_results.features}, "
          f"outliers: ")
    print(min_mean_plus_std_results.outliers)
    print(MyLinearRegression.get_main_results(min_mean_plus_std_results.result))

    print(f"\nmax_R2_adjusted: {max_r2_results.value}, "
          f"cutoff: {max_r2_results.cutoff}, "
          f"log on Price: {max_r2_results.log_regression}, "
          f"features: {max_r2_results.features}, "
          f"outliers:")
    print(max_r2_results.outliers)
    print(MyLinearRegression.get_main_results(max_r2_results.result))

    print(f"\nmin_product: {round(min_product_results.value,2)}, "
          f"cutoff: {min_product_results.cutoff}, "
          f"log on Price: {min_product_results.log_regression}, "
          f"features: {min_product_results.features}, "
          f"outliers:")
    print(min_product_results.outliers)
    print(MyLinearRegression.get_main_results(min_product_results.result))
