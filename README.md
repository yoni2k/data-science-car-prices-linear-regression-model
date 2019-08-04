# data-science-car-prices-linear-regression-model
Model of Linear Regression for predicting car prices

Based on exercise given as part of the Data Science bootcamp https://www.udemy.com/the-data-science-course-complete-data-science-bootcamp

Before running, do `pip install -r requirements.txt`

## Main conclusions:
1. Getting Adjusted R2 above .91-.92 is overfitting: 
    - the coefficients are huge
    - testing fails with bad results
2. Not doing **Log function** on `Price` was not shown to be a good idea
3. `Year` is extremely important in the predictions of the model.  Not including it lowers R2 significantly
4. `Model` is also extremely important in the predictions of the model.  Not including it lowers R2 significantly
5. Playing with outliers didn't help much, most outliers were left after optimization to be ~0.01 percent in the direction as presented in the lecture
6. `Brand` is an important features, there are models that are very rare (and might not be known from training), and Brands differ greatly in price.
7. However, there are a lot collinearity between results (some was shown in the course, and some is intuition that needs to be proven):
    - `Year` and `Mileage` - high
    - `Brand` and `Model` - very high
    - `Year` and `Model` - unknown, presumed high
    - `Mileage` and `Model` - unknown, presumed average
8.  Solutions for collinearity:
    - Not including `Year` and `Mileage` together, only one of them. Empirically `Year` wins
    - Not including `Brand` and `Model` - however, to make results better, they were made into one feature with their values concatenated
    - Following these 2 changes, the coefficients come out decent (and not too high / low which shows overfitting, and including 2 correlated variables)
    - It is possible that because of `Year` and `Model` are still very correlated, one of them will need to be removed to improve coefficients, and VIFs.  This will make the predictions drastically worse.
9.  In addition, because `Model` has a lot of rare values, it was attempted to group into `Other` all rare fields
10. It is very possible that with the given data, and using only Linear Regression, it is not possible to create both low collinearity / good coefficients, and good predictions
11. When using Data, and with very high/low coefficients, added a feature of replacing 0's with lowers value in training set, and *inf* with highest value of training set.  Later, it became clear that it's not really needed since it's a side effect of overfitting & collinearity
12. Adjusted R2 was used for optimizations.  However, since the set is large compared to number of features, the difference between R2 and Adjusted R2 is not significant.
13. Most of the outliers / removing null values remove too big of percentage of the dataset (~10%), needs to be better resolved by putting values instead of deleting rows 

## Suggested inputs for regression:
- Features: at least `Year`, `Brand`, `Model`, `Registration`
- Sometimes part or all of the below are included: `Engine Type`, `EngineV`, `Body`
- Combine fields of `Brand` and `Model`
- Perform Log on `Price`
- Outliers:
    - Price - high 1%
    - Year - low 1%
- Cut off percent for grouping `Model` values into `Other` (percent out of most frequent group of `Models`): 3-5%.  If appears less than that, group into `Other`

## Outputs of regression: 
- Adjusted R2 for training > .92
- Adjusted R2 for test > .91
- Mean of differences for test group is <20% and STB is <20%
- Example of feature set: 'Engine Type', 'EngineV', 'Body', 'Year', 'Brand', 'Model', 'Registration'
- Example of weights:
                                         Features   Weights
    0                                     EngineV  0.142481
    1                                        Year  0.599226
    2                                  Body_hatch -0.057315
    3                                  Body_other -0.035382
    4                                  Body_sedan -0.091220
    5                                  Body_vagon -0.075688
    6                                    Body_van -0.066470
    7                             Engine Type_Gas -0.035117
    8                           Engine Type_Other -0.007689
    9                          Engine Type_Petrol -0.039312
    10                           Registration_yes  0.260559
    11                        Brand_Model_Audi_80 -0.003793
    12                        Brand_Model_Audi_A3 -0.015053
    13                        Brand_Model_Audi_A4 -0.030319
    14                        Brand_Model_Audi_A5  0.003108
    15                        Brand_Model_Audi_A6 -0.041867
    16                Brand_Model_Audi_A6 Allroad -0.012489
    17                        Brand_Model_Audi_A8 -0.005898
    18                     Brand_Model_Audi_Other  0.006040
 