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
from pprint import pprint

from src.regression import MyLinearRegression


def optimize():
    features = ['Brand', 'Body', 'Mileage', 'EngineV', 'Engine Type', 'Registration', 'Year', 'Model']
    print("All features: " + str(features))

    outliers_dic = {
        'Price high': [.005, .01, .02],
        'Price low': [],
        'Mileage high': [0.005, .01, .02],
        'Mileage low': [],
        'Year high': [],
        'Year low': [0.005, 0.01, 0.02]
    }

    regs_dic = {}
    input_dic = {}
    input_dic['Remove outliers'] = [('Price', 'high', .01),
                                    ('Mileage', 'high', 0.01),
                                    ('Year', 'low', 0.01)
                                    ]

    for num_features_to_remove in range(1, len(features)-1):
        for features_to_remove in list(itertools.combinations(features, num_features_to_remove)):
            input_dic['Features to drop'] = list(features_to_remove)
            input_dic['Remove outliers'] = [('Price', 'high', .01),
                                            ('Mileage', 'high', 0.01),
                                            ('Year', 'low', 0.01)
                                            ]
            reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price')
            results_dic = reg.do_linear_regression(input_dic)

            if results_dic['R2'] < 0.8 and (results_dic['Diff mean'] + results_dic['Diff STD']) > 100:
                continue

            regs_dic[(tuple(set(features) - set(features_to_remove)), tuple(input_dic['Remove outliers']))] = results_dic
            continue

            for price_high_outlier in outliers_dic['Price high']:
                for mileage_high_outlier in outliers_dic['Mileage high']:
                    if 'Mileage' in features_to_remove:
                        continue
                    for year_low_outlier in outliers_dic['Year low']:
                        if 'Year' in features_to_remove:
                            continue
                        reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price')

                        input_dic['Remove outliers'] = [('Price', 'high', price_high_outlier),
                                                            ('Mileage', 'high', mileage_high_outlier),
                                                            ('Year', 'low', year_low_outlier)]
                        results_dic = reg.do_linear_regression(input_dic)
                        regs_dic[(tuple(set(features)-set(features_to_remove)), tuple(input_dic['Remove outliers']))] = results_dic

    max_r2 = 0
    max_r2_features = None
    max_r2_outliers = None
    max_r2_values = None
    min_mean_plus_std = float("inf")
    min_mean_plus_std_features = None
    max_mean_plus_std_outliers = None
    min_mean_plus_std_values = None
    min_product = float("inf")
    min_product_features = None
    min_product_outliers = None
    min_product_values = None

    for (features_left, outliers) in regs_dic:
        result = regs_dic[(features_left, outliers)]
        if result['R2'] > max_r2:
            max_r2 = result['R2']
            max_r2_features = features_left
            max_r2_outliers = outliers
            max_r2_values = result
        if result['Diff mean'] + result['Diff STD'] < min_mean_plus_std:
            min_mean_plus_std = result['Diff mean'] + result['Diff STD']
            min_mean_plus_std_features = features_left
            min_mean_plus_std_outliers = outliers
            min_mean_plus_std_values = result
        product = (1-result['R2']) * (result['Diff mean'] + result['Diff STD'])
        if product < min_product:
            min_product = product
            min_product_features = features_left
            min_product_outliers = outliers
            min_product_values = result

    print(f"\nmax_R2: {max_r2}, features: {max_r2_features}, outliers:")
    print(max_r2_outliers)
    print(MyLinearRegression.get_main_results(max_r2_values))

    print(f"\nmin_mean_plus_std: {min_mean_plus_std}, features: {min_mean_plus_std_features}, outliers: ")
    print(min_mean_plus_std_outliers)
    print(MyLinearRegression.get_main_results(min_mean_plus_std_values))

    print(f"\nmin_product: {min_product}, features: {min_product_features}, outliers:")
    print(min_product_outliers)
    print(MyLinearRegression.get_main_results(min_product_values))
