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

    def get_size(self):
        return len(self.current_data)

    def get_initial_size(self):
        return len(self.initial_data)

    def get_features(self):
        return self.current_data.columns.values

    def describe(self):
        return self.current_data.describe(include='all').to_string()

    def head(self, num_rows=5):
        return self.current_data.head(num_rows).to_string()

    def get_num_rows_with_null_vals(self):
        return self.current_data.isnull().sum().sum()

    def display_dist(self, feature_name):
        sns.distplot(self.current_data[feature_name])
        plt.title(f'Distribution of {feature_name}')
        plt.show()

    def get_vif(self, features_list):
        variables = self.current_data[features_list]
        vif = pd.DataFrame()
        vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
        vif["features"] = variables.columns
        return vif.to_string()

    """ Methods that affect the data """

    def drop_features(self, feature_list):
        self.current_data.drop(feature_list, axis=1, inplace=True)

    def drop_null_rows(self):
        self.current_data.dropna(axis=0, inplace=True)
        self.current_data.reset_index(drop=True, inplace=True)

    def remove_outliers_low_fraction(self, feature_name, fraction_to_remove):
        percentile_value = self.current_data[feature_name].quantile(fraction_to_remove)
        self.current_data = self.current_data[self.current_data[feature_name] > percentile_value]
        self.current_data.reset_index(drop=True, inplace=True)

    def remove_outliers_high_fraction(self, feature_name, fraction_to_remove):
        percentile_value = self.current_data[feature_name].quantile(1 - fraction_to_remove)
        self.current_data = self.current_data[self.current_data[feature_name] < percentile_value]
        self.current_data.reset_index(drop=True, inplace=True)

    def remove_outliers_low_num(self, feature_name, num_to_remove_up_to):
        self.current_data = self.current_data[self.current_data[feature_name] > num_to_remove_up_to]
        self.current_data.reset_index(drop=True, inplace=True)

    def remove_outliers_high_num(self, feature_name, num_to_remove_from):
        self.current_data = self.current_data[self.current_data[feature_name] < num_to_remove_from]
        self.current_data.reset_index(drop=True, inplace=True)

    def do_log_on_dependent(self):
        self.current_data[self.name_dependent] = np.log(self.current_data[self.name_dependent])

    def add_dummies(self):
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

        # TODO copy here different debug features from test functions

        self.drop_features(input_dic['Features to drop'])

        features = self.get_features()
        if self.debug:
            print("Features after dropping (" + str(len(features)) + "):\n" + str(features) + '\n')

        self.drop_null_rows()

        self.remove_outliers_high_fraction('Price', .01)

        if 'Mileage' not in input_dic['Features to drop']:
            self.remove_outliers_high_fraction('Mileage', .01)

        if 'EngineV' not in input_dic['Features to drop']:
            self.remove_outliers_high_num('EngineV', 6.5)

        if 'Year' not in input_dic['Features to drop']:
            self.remove_outliers_low_fraction('Year', 0.01)

        self.do_log_on_dependent()

        self.add_dummies()

        results_dic = self.do_actual_regression_part()
        # TODO print better - copy from other places
        if self.debug:
            print(f"Coefficients summary:\n{results_dic['Coef summary']}")
            print(f"Differences summary: \n{results_dic['Diff summary']}")
            print(f"R2: {results_dic['R2']}, pred_diff_percent_mean: {results_dic['Diff mean']}, "
                  f"pred_diff_percent_std: {results_dic['Diff STD']}")
        return results_dic

    def do_actual_regression_part(self):
        results = {}

        targets = self.current_data[self.name_dependent]
        inputs = self.current_data.drop(self.name_dependent, axis=1)

        # Scale
        scaler = StandardScaler()
        scaler.fit(inputs)
        inputs_scaled = scaler.transform(inputs)

        # Split training and test
        x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

        # Perform regression
        reg = LinearRegression()
        reg.fit(x_train, y_train)

        # Predict
        y_hat = reg.predict(x_train)

        # Check results
        r2 = reg.score(x_train, y_train)

        # Prepare summary
        coef_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
        coef_summary['Weights'] = reg.coef_

        # Test
        y_hat_test = reg.predict(x_test)
        # TODO should leave this handling? Switch to something else? Enlarge the range?
        with np.errstate(over='ignore'):
            y_hat_test_exp = np.exp(y_hat_test)
        df_pf = pd.DataFrame(y_hat_test_exp, columns=['Prediction'])
        y_test = y_test.reset_index(drop=True)
        df_pf['Target'] = np.exp(y_test)
        df_pf, inf_replaced, inf_replaced_with = self._fix_prediction_inf(df_pf, np.exp(y_train.max()))
        df_pf, zero_replaced, zero_replaced_with = self._fix_prediction_zero(df_pf, np.exp(y_train.min()))

        df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
        df_pf['Difference%'] = np.absolute(df_pf['Residual'] / df_pf['Target'] * 100)

        # TODO make sure these are printed properly when needed, and used in the correct place
        results['Num records original'] = len(self.initial_data)
        results['Num records regression'] = len(self.current_data)
        results['Percent dropped'] = round(len(self.current_data) / len(self.initial_data) * 100, 2)
        results['R2'] = round(r2, 3)
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
            'Percent dropped': full_dic['Percent dropped'],
            'R2': full_dic['R2'],
            'Diff mean': full_dic['Diff mean'],
            'Diff STD': full_dic['Diff STD'],
            'inf_replaced': full_dic['inf_replaced'],
            'zero_replaced': full_dic['inf_replaced']
            }

