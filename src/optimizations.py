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

# TODO go over imports
import itertools

from src.regression import MyLinearRegression


def optimize():
    features = ['Brand', 'Body', 'Mileage', 'EngineV', 'Engine Type', 'Registration', 'Year', 'Model']
    print("All features: " + str(features))

    my_dic = {}

    for i in range(1, len(features)-1):
        remove_features = list(itertools.combinations(features, i))
        for j in remove_features:
            reg = MyLinearRegression('../resources/1.04. Real-life example.csv', 'Price')
            # TODO rename method
            results_dic = reg.do_linear_regression(list(j))
            my_dic[tuple(set(features)-set(j))] = results_dic

    max_r2 = 0
    max_r2_features = None
    max_r2_values = None
    min_mean_plus_std = float("inf")
    min_mean_plus_std_features = None
    min_mean_plus_std_values = None
    min_product = float("inf")
    min_product_features = None
    min_product_values = None

    # TODO rename i, j vars
    for i in my_dic:
        # TODO can't print everything
        # print(f"{i}:" + str(my_dic[i]))
        if my_dic[i]['R2'] > max_r2:
            max_r2 = my_dic[i]['R2']
            max_r2_features = i
            max_r2_values = my_dic[i]
        if my_dic[i]['Diff mean'] + my_dic[i]['Diff STD'] < min_mean_plus_std:
            min_mean_plus_std = my_dic[i]['Diff mean'] + my_dic[i]['Diff STD']
            min_mean_plus_std_features = i
            min_mean_plus_std_values = my_dic[i]
        product = (1-my_dic[i]['R2']) * (my_dic[i]['Diff mean'] + my_dic[i]['Diff STD'])
        if product < min_product:
            min_product = product
            min_product_features = i
            min_product_values = my_dic[i]

    print(f"\nmax_R2: {max_r2}, features: {max_r2_features}, values:")
    print(MyLinearRegression.get_main_results(max_r2_values))
    print(f"\nmin_mean_plus_std: {min_mean_plus_std}, features: {min_mean_plus_std_features}, values:")
    print(MyLinearRegression.get_main_results(min_mean_plus_std_values))
    print(f"\nmin_product: {min_product}, features: {min_product_features}, values:")
    print(MyLinearRegression.get_main_results(min_product_values))
