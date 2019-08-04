"""
TODOS by Priority:
* Check best answers don't have very low coefficients
* Make sure reference group is not too small
* Check conclusions on numerous train/test
* Check if being without Year can do (can't have Mileage and Year because of high correlation,
    and Year is worse since it has also bigger corellation to Model


Ask questions:
* Is this a good way to check how good the model is: min (1-R2)*(diff.mean + diff.std) or perhaps (1-R2 train)*(1-R2 test)
* There are many different correlations between different variables.
    - How do we check correlation for categorical variables?
    - My intuition / playing with the data:
        - Brand and Model has extremely high correlation.  Can't leave together, perhaps combine
        - Year has high correlation to: Mileage, but also to Model
        - Mileage has high correction to Model also probably
"""

import itertools
from pprint import pprint

from src.regression import MyLinearRegression


def optimize():
    features = ['Brand', 'Body', 'Mileage', 'EngineV', 'Engine Type', 'Registration', 'Year', 'Model']
    print("All features: " + str(features))

    short_run = False
    special_run = False

    # TODO - return removing year always? Year probably in addition to high correlation to Mileage, has
    #  also high correlation to Model
    features_to_always_remove = ['Year']

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
        do_log_on_regression = {True}
        combine_brand_model = {True}
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
        combine_brand_model = {True}
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
        # TODO - because of high correlation, and the fact that it doesn't hurt predictions too much,
        #  removed ability to have both at the same time not combined
        #  combine_brand_model = {True, False}
        combine_brand_model = {True}

    regs_dic = {}
    input_dic = {}

    for log_regression in do_log_on_regression:
        input_dic['Perform log on dependent'] = log_regression
        for num_features_to_remove in range(len(features)-1):
            for features_to_remove in list(itertools.combinations(features, num_features_to_remove)):
                contains_features_must_remove = False
                for to_remove in features_to_always_remove:
                    if to_remove not in features_to_remove:
                        contains_features_must_remove = True
                        break
                if contains_features_must_remove:
                    continue
                if ('Year' not in features_to_remove) and ('Mileage' not in features_to_remove):
                    continue  # do not allow together due to high correlation
                for combine_models in combine_brand_model:
                    if combine_models and (('Brand' in features_to_remove) or ('Model' in features_to_remove)):
                        continue
                    if combine_models:
                        input_dic['Combine features'] = [('Brand', 'Model')]
                    else:
                        input_dic['Combine features'] = None

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
                                              log_regression,
                                              combine_models))):
                            regs_dic[(tuple(set(features) - set(features_to_remove)),
                                      tuple(input_dic['Remove outliers']),
                                      cutoff,
                                      log_regression,
                                      combine_models)] = results_dic

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
                                                  log_regression,
                                                  combine_models)] = results_dic

    class results_opt:
        value = 0
        features = None
        outliers = None
        result = None
        cutoff = 0
        log_regression = False
        combine_models = False

        def copy(self):
            new_res = results_opt()
            new_res.value = self.value
            new_res.features = self.features
            new_res.outliers = self.outliers
            new_res.result = self.result
            new_res.cutoff = self.cutoff
            new_res.log_regression = self.log_regression
            new_res.combine_models = self.combine_models
            return new_res

    max_r2_results = results_opt()
    min_mean_plus_std_results = results_opt()
    min_mean_plus_std_results.value = float("inf")
    min_product_results = results_opt()
    min_product_results.value = float("inf")
    max_product_adj_r2 = results_opt()
    max_product_adj_r2.value = float("-inf")

    results = results_opt()

    for (results.features, results.outliers, results.cutoff, results.log_regression, results.combine_models) in regs_dic:
        results.result = regs_dic[(results.features, results.outliers, results.cutoff, results.log_regression, results.combine_models)]
        if results.result['Diff mean'] + results.result['Diff STD'] < min_mean_plus_std_results.value:
            results.value = results.result['Diff mean'] + results.result['Diff STD']
            min_mean_plus_std_results = results.copy()
        if results.result['R2 Adj'] > max_r2_results.value:
            results.value = results.result['R2 Adj']
            max_r2_results = results.copy()
        product = (1-results.result['R2 Adj']) * (results.result['Diff mean'] + results.result['Diff STD'])
        if product < min_product_results.value:
            results.value = product
            min_product_results = results.copy()
        product_adj_r2 = results.result['R2 Adj'] * results.result['R2 Adj Test']
        if product_adj_r2 > max_product_adj_r2.value:
            results.value = product_adj_r2
            max_product_adj_r2 = results.copy()

    print(f"\nmin_mean_plus_std: {round(min_mean_plus_std_results.value, 2)}, "
          f"cutoff: {min_mean_plus_std_results.cutoff}, "
          f"log on Price: {min_mean_plus_std_results.log_regression}, "
          f"combine_models: {min_mean_plus_std_results.combine_models}, "
          f"features: {min_mean_plus_std_results.features}, "
          f"outliers: ")
    print(min_mean_plus_std_results.outliers)
    print(MyLinearRegression.get_main_results(min_mean_plus_std_results.result))

    print(f"\nmin_product: {round(min_product_results.value,2)}, "
          f"cutoff: {min_product_results.cutoff}, "
          f"log on Price: {min_product_results.log_regression}, "
          f"combine_models: {min_product_results.combine_models}, "
          f"features: {min_product_results.features}, "
          f"outliers:")
    print(min_product_results.outliers)
    print(MyLinearRegression.get_main_results(min_product_results.result))
    #print(min_product_results.result['Coef summary'].to_string(max_rows=1500))

    print(f"\nmax_product_adj_r2: {round(max_product_adj_r2.value,2)}, "
          f"cutoff: {max_product_adj_r2.cutoff}, "
          f"log on Price: {max_product_adj_r2.log_regression}, "
          f"combine_models: {max_product_adj_r2.combine_models}, "
          f"features: {max_product_adj_r2.features}, "
          f"outliers:")
    print(max_product_adj_r2.outliers)
    print(MyLinearRegression.get_main_results(max_product_adj_r2.result))
    #print(max_product_adj_r2.result['Coef summary'].to_string(max_rows=1500))

    print(f"\nmax_R2_adjusted: {max_r2_results.value}, "
          f"cutoff: {max_r2_results.cutoff}, "
          f"log on Price: {max_r2_results.log_regression}, "
          f"combine_models: {max_r2_results.combine_models}, "
          f"features: {max_r2_results.features}, "
          f"outliers:")
    print(max_r2_results.outliers)
    print(MyLinearRegression.get_main_results(max_r2_results.result))
    #print(max_r2_results.result['Coef summary'].to_string(max_rows=1500))
