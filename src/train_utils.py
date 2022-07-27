
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from  seaborn import heatmap
import pickle

def columnsToDrop(df):
    var_thr = VarianceThreshold() #Removing both constant and quasi-constant
    #  keep other columns except ENGINE_NUMBER and Will_Fail (i.e y)
    feature_check_df = df.loc[:, ~df.columns.isin(['ENGINE_NUMBER', 'willFail' ])]
    var_thr.fit(feature_check_df.values)

    constantColumns = [column for column in feature_check_df.columns 
            if column not in feature_check_df.columns[var_thr.get_support()]]
    
    # Dropping Engine Number and Time in Cycles per the Nasa Turbo fan study recommendations
    dropColumns =   ['ENGINE_NUMBER', 'TIME_IN_CYCLES'] + constantColumns
    return dropColumns

def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))


def processTestData(df):

  X_test = df

  # create temporary dataframe that contains engine_number and maximum number of cycles
  temp = X_test.groupby('ENGINE_NUMBER')['TIME_IN_CYCLES'].max().reset_index()
  temp.columns = ['ENGINE_NUMBER','MAX']

  # append temporary dataframe to end of X_test
  X_test = X_test.merge(temp, on=['ENGINE_NUMBER'], how='left')

  # append 'delete' column which contains the difference between cycles and maximum cycles
  X_test['DELETE'] = X_test['TIME_IN_CYCLES'] - X_test['MAX']

  # removes all rows that difference between cycles is non-zero
  # this gives us the data for the last point for each engine
  X_test = X_test[X_test['DELETE'] == 0]

  # final cleanup to remove unnecessary columns
  colsDrop = columnsToDrop(X_test)
  X_test.drop(columns = colsDrop, inplace=True)
  X_test.drop(columns = ["MAX"], inplace=True)
  return X_test.values

def plot_graph(var_name , train, ax):
    for itr in train['ENGINE_NUMBER'].unique():
        if (itr % 8 == 0):  # only 8th unit of Engine else the graph becomes too noisy
            ax.plot('RUL', var_name, 
                     data=train[train['ENGINE_NUMBER']==itr])
    ax.set_xlim(250,0) # reversing the axis allows us to see reducing RUL and impact of the variable
    ax.set_xticks(np.arange(0, 275, 25))
    ax.set_ylabel(var_name ,fontsize=12)
    ax.set_xlabel('RUL(Time Cycles)', fontsize=12)


def cross_validate(model, X_values, y_values):
    # Cross Validation to get a more reliable measure of our model
    kfold = KFold( n_splits=10, random_state=7, shuffle=True )
    results = cross_val_score(model, X_values, y_values,scoring='neg_mean_absolute_error',  cv=kfold)
    print("Cross Validation : %0.2f accuracy with a standard deviation of %0.2f" % (np.absolute(results.mean()), results.std())) # Train set needs to be refined further

def save_model( scaler, model, path, filename):

    serializer  = [scaler, model]
    with open(path+filename, 'wb') as files:
        pickle.dump(serializer, files)