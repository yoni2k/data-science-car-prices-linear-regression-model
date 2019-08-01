"""
TODOS by Priority:
* Outliners - look into values, see results
* Make sure reference group is not too small
* Do outliners for categorical values
* Outliners - check with different ones
* Combine Brand and Model
* Add adjusted R
* Check conclusions on numerous train/test
* Last: Don't do log on price

TODOs general coding improvements:
* deal better with debugging


Ask questions:
* Is this a good way to check how good the model is: min (1-R2)*(diff.mean + diff.std)?
"""

import itertools

from src.regression import MyLinearRegression


def optimize():
    features = ['Brand', 'Body', 'Mileage', 'EngineV', 'Engine Type', 'Registration', 'Year', 'Model']
    print("All features: " + str(features))

    regs_dic = {}

    for num_features_to_remove in range(1, len(features)-1):
        for features_to_remove in list(itertools.combinations(features, num_features_to_remove)):
            reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price')
            results_dic = reg.do_linear_regression(list(features_to_remove))
            regs_dic[tuple(set(features)-set(features_to_remove))] = results_dic

    max_r2 = 0
    max_r2_features = None
    max_r2_values = None
    min_mean_plus_std = float("inf")
    min_mean_plus_std_features = None
    min_mean_plus_std_values = None
    min_product = float("inf")
    min_product_features = None
    min_product_values = None

    for features_left in regs_dic:
        result = regs_dic[features_left]
        if result['R2'] > max_r2:
            max_r2 = result['R2']
            max_r2_features = features_left
            max_r2_values = result
        if result['Diff mean'] + result['Diff STD'] < min_mean_plus_std:
            min_mean_plus_std = result['Diff mean'] + result['Diff STD']
            min_mean_plus_std_features = features_left
            min_mean_plus_std_values = result
        product = (1-result['R2']) * (result['Diff mean'] + result['Diff STD'])
        if product < min_product:
            min_product = product
            min_product_features = features_left
            min_product_values = result

    print(f"\nmax_R2: {max_r2}, features: {max_r2_features}, values:")
    print(MyLinearRegression.get_main_results(max_r2_values))
    print(f"\nmin_mean_plus_std: {min_mean_plus_std}, features: {min_mean_plus_std_features}, values:")
    print(MyLinearRegression.get_main_results(min_mean_plus_std_values))
    print(f"\nmin_product: {min_product}, features: {min_product_features}, values:")
    print(MyLinearRegression.get_main_results(min_product_values))
