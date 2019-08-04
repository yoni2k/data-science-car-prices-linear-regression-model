import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set()


class MyLinearRegression:
    """ TODO Fill in - also in other places"""
    initial_data = pd.DataFrame()
    current_data = pd.DataFrame()
    name_dependent = ""
    debug = False

    """ Methods that affect the data """

    def __init__(self, file_path, name_dependent, debug=False):
        self.initial_data = pd.read_csv(file_path)
        self.current_data = self.initial_data.copy()
        self.name_dependent = name_dependent
        self.debug = debug

    """ Methods that don't affect the data """

    def _get_size(self):
        return len(self.current_data)

    def _get_initial_size(self):
        return len(self.initial_data)

    def _get_left_rows_percent_str(self):
        return str(round(len(self.current_data) * 100 / len(self.initial_data), 2)) + " %"

    def _get_features(self):
        return self.current_data.columns.values

    def _describe(self):
        return self.current_data.describe(include='all').to_string()

    def _head(self, num_rows=5):
        return self.current_data.head(num_rows).to_string()

    def _get_num_rows_with_null_vals(self):
        return self.current_data.isnull().sum().sum()

    def _display_dist(self, feature_name):
        sns.distplot(self.current_data[feature_name])
        plt.title(f'Distribution of {feature_name}')
        plt.show()

    def _get_frequency(self, feature_name):
        return self.current_data[feature_name].value_counts()

    def _get_vif(self, features_list):
        variables = self.current_data[features_list]
        vif = pd.DataFrame()
        vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
        vif["features"] = variables.columns
        return vif.to_string()


    """ Methods that affect the data """

    def _replace_categorical_fraction_from_max(self, feature_name, frac_from_max, name_to_replace):
        if frac_from_max == 0:
            return
        counts = self.current_data[feature_name].value_counts()
        max_count = counts.max()
        index = -1
        axes = counts.axes
        for count in counts:
            index += 1
            if count < (frac_from_max * max_count):
                if self.debug:
                    print(f"Replacing {axes[0][index]}, count: {count}, frac {frac_from_max}, max_count: {max_count}, frac_from_max * max_count: {frac_from_max * max_count}")
                self.current_data[feature_name].replace(axes[0][index], name_to_replace, inplace=True)

    def _drop_features(self, feature_list):
        self.current_data.drop(feature_list, axis=1, inplace=True)

    def _drop_null_rows(self):
        self.current_data.dropna(axis=0, inplace=True)
        self.current_data.reset_index(drop=True, inplace=True)

    def _remove_outliers_low_fraction(self, feature_name, fraction_to_remove):
        percentile_value = self.current_data[feature_name].quantile(fraction_to_remove)
        self.current_data = self.current_data[self.current_data[feature_name] > percentile_value]
        self.current_data.reset_index(drop=True, inplace=True)

    def _remove_outliers_high_fraction(self, feature_name, fraction_to_remove):
        percentile_value = self.current_data[feature_name].quantile(1 - fraction_to_remove)
        self.current_data = self.current_data[self.current_data[feature_name] < percentile_value]
        self.current_data.reset_index(drop=True, inplace=True)

    def _remove_outliers_low_num(self, feature_name, num_to_remove_up_to):
        self.current_data = self.current_data[self.current_data[feature_name] > num_to_remove_up_to]
        self.current_data.reset_index(drop=True, inplace=True)

    def _remove_outliers_high_num(self, feature_name, num_to_remove_from):
        self.current_data = self.current_data[self.current_data[feature_name] < num_to_remove_from]
        self.current_data.reset_index(drop=True, inplace=True)

    def _combine_features(self, feature1, feature2):
        self.current_data[feature1 + '_' + feature2] = self.current_data[feature1] + '_' + self.current_data[feature2]
        self.current_data.drop([feature1, feature2], axis=1, inplace=True)

    def _do_log_on_dependent(self):
        self.current_data[self.name_dependent] = np.log(self.current_data[self.name_dependent])

    def _add_dummies(self):
        self.current_data = pd.get_dummies(self.current_data, drop_first=True)

    """ Main methods """

    def _fix_prediction_inf(self, predicted_y, y_train_max):
        inf_replaced = False
        inf_replaced_with = None

        #  print("20 largest before:\n" + str(predicted_y['Prediction'].nlargest(20)) + "\n")
        if predicted_y['Prediction'].max() == np.inf:
            if self.debug:
                print(f"Replacing inf with {y_train_max}")
            inf_replaced = True
            inf_replaced_with = y_train_max
            predicted_y['Prediction'].replace(np.inf, y_train_max, inplace=True)

        return predicted_y, inf_replaced, inf_replaced_with

    def _fix_prediction_zero(self, predicted_y, y_train_min):
        zero_replaced = False
        zero_replaced_with = None

        if predicted_y['Prediction'].min() == 0:
            if self.debug:
                print(f"Replacing 0 with {y_train_min}")
            zero_replaced = True
            zero_replaced_with = y_train_min
            predicted_y['Prediction'].replace(0, y_train_min, inplace=True)

        return predicted_y, zero_replaced, zero_replaced_with

    def do_linear_regression(self, input_dic):

        output_dic = {}

        if self.debug:
            features_orig = self._get_features()
            print(f"Features before dropping ({str(len(features_orig))}): {str(features_orig)}\n")
        self._drop_features(input_dic['Features to drop'])
        if self.debug:
            features_after = self._get_features()
            print("\n-------------------------------------------------------\nAFTER dropping features:")
            print(f"Features after dropping ({str(len(features_after))}): {str(features_after)}\n")

        if self.debug:
            print(f"Rows before dropping null values: {str(self._get_size())} {self._get_left_rows_percent_str()}")
        self._drop_null_rows()
        if self.debug:
            print(f"Rows after dropping null values: {str(self._get_size())} {self._get_left_rows_percent_str()}")

        # Always remove EngineV's top
        if 'EngineV' not in input_dic['Features to drop']:
            self._remove_outliers_high_num('EngineV', 6.5)

        for outlier in input_dic['Remove outliers']:
            if outlier[0] not in input_dic['Features to drop']:
                if outlier[2] != 0:
                    if outlier[1] == 'high':
                        if self.debug:
                            print(f"Rows after removing outliers - high: {outlier[0]} {outlier[2]}: {str(self._get_size())} {self._get_left_rows_percent_str()}")
                        self._remove_outliers_high_fraction(outlier[0], outlier[2])
                    elif outlier[1] == 'low':
                        self._remove_outliers_low_fraction(outlier[0], outlier[2])
                        if self.debug:
                            print(f"Rows after removing outliers - low: {outlier[0]} {outlier[2]}: {str(self._get_size())} {self._get_left_rows_percent_str()}")
                    else:
                        assert False, "invalid parameter"

        if input_dic['Remove rare categorical']:
            for remove_rare in input_dic['Remove rare categorical']:
                if remove_rare[0] not in input_dic['Features to drop']:
                    self._replace_categorical_fraction_from_max(remove_rare[0], remove_rare[1], "Other")

        if input_dic.get('Combine features'):
            for combine_features in input_dic.get('Combine features'):
                self._combine_features(combine_features[0], combine_features[1])

        if input_dic.get('Perform log on dependent') is None or input_dic.get('Perform log on dependent'):
            self._do_log_on_dependent()

        self._add_dummies()

        results_dic = self._do_actual_regression_part(input_dic)
        output_dic.update(results_dic)
        # TODO print better - copy from other places
        if self.debug:
            print(f"Coefficients summary:\n{output_dic['Coef summary']}")
            print(f"Differences summary: \n{output_dic['Diff summary']}")
            print(f"R2: {output_dic['R2']}, pred_diff_percent_mean: {output_dic['Diff mean']}, "
                  f"pred_diff_percent_std: {output_dic['Diff STD']}")
        return output_dic

    def _do_actual_regression_part(self, input_dic):
        results = {}

        targets = self.current_data[self.name_dependent]
        inputs = self.current_data.drop(self.name_dependent, axis=1)

        # Scale
        scaler = StandardScaler()
        scaler.fit(inputs)
        inputs_scaled = scaler.transform(inputs)

        # Split training and test
        x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

        # Removed, way way too slow, leaving in case every needed in the future
        # max_vif = max([variance_inflation_factor(x_train, i) for i in range(x_train.shape[1])])

        # Perform regression
        reg = LinearRegression()
        reg.fit(x_train, y_train)

        # Predict
        y_hat = reg.predict(x_train)

        # Check results
        r2 = reg.score(x_train, y_train)
        n = x_train.shape[0]
        p = x_train.shape[1]
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Prepare summary
        coef_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
        coef_summary['Weights'] = reg.coef_

        # Test
        # Perform common preparations regardless if log function is done or not
        n_test = x_test.shape[0]
        p_test = x_test.shape[1]
        r2_test = reg.score(x_test, y_test)
        r2_adj_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p_test - 1)
        # print(f"n_test: {n_test}, p_test: {p_test}, r2_test: {r2_test}, r2_adj_test: {r2_adj_test}")

        y_hat_test = reg.predict(x_test)
        y_test = y_test.reset_index(drop=True)
        y_train_max = y_train.max()
        y_train_min = y_train.min()
        if input_dic.get('Perform log on dependent') is None or input_dic.get('Perform log on dependent'):
            with np.errstate(over='ignore'):
                y_hat_test_exp = np.exp(y_hat_test)
            df_pf = pd.DataFrame(y_hat_test_exp, columns=['Prediction'])
            df_pf['Target'] = np.exp(y_test)
            y_train_max = np.exp(y_train_max)
            y_train_min = np.exp(y_train_min)
        else:
            df_pf = pd.DataFrame(y_hat_test, columns=['Prediction'])
            df_pf['Target'] = y_test

        # TODO should leave this handling? Switch to something else? Enlarge the range?
        df_pf, inf_replaced, inf_replaced_with = self._fix_prediction_inf(df_pf, y_train_max)
        df_pf, zero_replaced, zero_replaced_with = self._fix_prediction_zero(df_pf, y_train_min)

        df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
        df_pf['Difference%'] = np.absolute(df_pf['Residual'] / df_pf['Target'] * 100)

        # TODO make sure these are printed properly when needed, and used in the correct place
        results['Num records original'] = len(self.initial_data)
        results['Num records regression'] = len(self.current_data)
        results['Percent dropped'] = round((1 - len(self.current_data) / len(self.initial_data)) * 100, 2)
        results['Model cutoff'] = input_dic['Remove rare categorical'][0][1] if input_dic['Remove rare categorical'] else 0
        results['R2'] = round(r2, 3)
        results['R2 Adj'] = round(r2_adj, 3)
        results['R2 Adj Test'] = round(r2_adj_test, 3)
        results['n'] = n
        results['p'] = p
        results['Diff mean'] = round(df_pf.describe()['Difference%']['mean'], 2)
        results['Diff STD'] = round(df_pf.describe()['Difference%']['std'], 2)
        results['Coef summary'] = coef_summary
        results['Diff summary'] = df_pf.describe().to_string()
        results['inf_replaced'] = inf_replaced
        results['zero_replaced'] = zero_replaced
        results['inf_replaced_with'] = inf_replaced_with
        results['zero_replaced_with'] = zero_replaced_with

        return results

    """ Static methods"""

    @staticmethod
    def get_main_results(full_dic):
        return {
            'R2 Adj': full_dic['R2 Adj'],
            'R2 Adj Test': full_dic['R2 Adj Test'],
            'Diff mean + STD': full_dic['Diff mean'] + full_dic['Diff STD'],
            '% dropped': full_dic['Percent dropped'],
            'Model cutoff': full_dic['Model cutoff'],
            'Diff mean': full_dic['Diff mean'],
            'Diff STD': full_dic['Diff STD'],
            'n': full_dic['n'],
            'p': full_dic['p'],
            'R2': full_dic['R2'],
            'inf_replaced': full_dic['inf_replaced'],
            'zero_replaced': full_dic['inf_replaced']
            }

