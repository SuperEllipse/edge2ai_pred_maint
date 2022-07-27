import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from  seaborn import heatmap
from train_utils import columnsToDrop, evaluate, plot_graph, cross_validate, cross_val_score, processTestData, save_model

#DEFAULT VALUES
file = 'FD001'
Cycle_Alert_Threshold = 40

file_relative_path = "./data/"  ## To be changed to an environment variable

train_filename    = "train_" + file + ".csv"
test_filename     = "test_" + file + ".csv"
test_RUL_filename = "RUL_" + file + ".txt"

"""
STEP 0: Starting point : Fetch DATA 
"""

# Load our Datasets
train_df = pd.read_csv(file_relative_path + train_filename)
test_df = pd.read_csv(file_relative_path + test_filename)
# true_RUL: The 'true' RUL test values are provided, these are used for testing the predictions of our Test Data set 
true_RUL = pd.read_csv(file_relative_path + test_RUL_filename, names=["RUL"])

## Before this step the Data exploration and feature engineering is completed in 
## Time to get to modelling.
"""
STEP 1: TRAINING DATA - Filter out unnecessary data
"""


dropColumns = columnsToDrop(train_df)
print('Following Columns are dropped from Train dataset {0}'.format(dropColumns))

X_train = train_df.drop(columns = dropColumns, axis=1, inplace=False)
X_train.drop(columns = ["RUL"], axis=1, inplace=True)                # Only present in training DS
X_train = X_train.values
y_train = train_df.RUL.values
X_test =  processTestData(test_df)
y_test = true_RUL.values

### Scaling the Input features to minimize any outlier impacts.
print(X_train.shape)
print(X_test.shape) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
STEP 2A: Creating a baseline Regression Model  
"""


print("EVALUATING RUL for Base Model : ")
lmodel = LinearRegression()
lmodel.fit(X_train, y_train)

# Cross Validation to get a more reliable measure of our model
cross_validate(lmodel, X_train, y_train)

# prediction
y_hat_test_lr = lmodel.predict(X_test)
# evaluate Prediction performance
evaluate(y_test, y_hat_test_lr, 'test')


"""
STEP 2B: Decision Tree Regressor
"""

print("EVALUATING RUL for Decision Tree Model : ")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)


# Cross Validation to get a more reliable measure of our model
cross_validate(tree_reg, X_train, y_train)

#prediction
y_hat_test_dtree = tree_reg.predict(X_test)

#evaluate model performance
evaluate(y_test, y_hat_test_dtree, 'test')

"""
STEP 2C: Random Forest Regressor 
"""

print("EVALUATING RUL for Random Forest Regressor : ")

fr_reg = RandomForestRegressor()
fr_reg.fit(X_train, y_train)

# Cross Validation to get a more reliable measure of our model
cross_validate(fr_reg, X_train, y_train)

# predict 
y_hat_test_fr = fr_reg.predict(X_test)

#evaluate model performance
evaluate(y_test, y_hat_test_fr, 'test')



"""
STEP 2D: Trying out XGB Regressor with some known hyperparameters from internal / external papers/ articles etc.  
"""
print("EVALUATING RUL for XGBR Model : ")
xgbr_model = xgb.XGBRegressor( n_estimators=85, learning_rate=0.018, gamma=0, subsample=0.5,
                           colsample_bytree=0.5, max_depth=3)
xgbr_model.fit(X_train,y_train)


# Cross Validation to get a more reliable measure of our model
cross_validate(xgbr_model, X_train, y_train)

# predict and evaluate
y_hat_test_xgbr = xgbr_model.predict(X_test)

evaluate(y_test, y_hat_test_xgbr, 'test')


"""
STEP 2E:# training a Regression MLP using Sequential API   
"""


from sklearn.model_selection import train_test_split
from tensorflow  import keras 

X_train_sub1, X_valid, y_train_sub1, y_valid = train_test_split(X_train, y_train)
scaler_mlp = StandardScaler()
X_train_sub1_scaled = scaler_mlp.fit_transform(X_train_sub1)
X_valid_scaled = scaler_mlp.transform(X_valid)
X_test_scaled = scaler_mlp.transform(X_test)

model = keras.models.Sequential([
 keras.layers.Dense(100, activation="relu", input_shape=X_train_sub1_scaled.shape[1:]),
 keras.layers.Dense(1)
 ])
model.compile(loss="mse", optimizer="adam",   metrics=[keras.metrics.RootMeanSquaredError()])
model.fit(X_train_sub1_scaled, y_train_sub1, epochs=20,
                    validation_data=(X_valid, y_valid))
y_hat_test_MLP = model.predict(X_test_scaled)
evaluate(y_test, y_hat_test_MLP)

"""
STEP 4: Training an  XGBoost Classifiers for Classifying Fail/ Not Fail based on predefined Threshold
# This classifier model has been created but not serialized ( for now)
"""
# Training a classifier ( Directly training XGBoost here but the same approach as for the Regressor would be adopted here.)
train_df['willFail'] = np.where(train_df['RUL'] <= Cycle_Alert_Threshold, 1, 0 )
train_y = train_df['willFail']
xgbCl_model = xgb.XGBRFClassifier()
xgbCl_model.fit(X_train, train_y)
true_RUL_class = pd.DataFrame()
true_RUL_class['willFail'] = np.where(true_RUL['RUL'] <= Cycle_Alert_Threshold, 1, 0)
true_RUL_class = true_RUL_class.values
pred_fail = xgbCl_model.predict(X_test)
accuracy = "{:.2f}%".format(accuracy_score(true_RUL_class, pred_fail) * 100)

#+-------------+-------------+
y_real = np.transpose(np.squeeze(true_RUL_class))
y_pred = np.array(pred_fail)
AUC = "{:.2f}".format(roc_auc_score(y_real, y_pred))

tn,fp,fn,tp = confusion_matrix(y_real,y_pred).ravel() 

print('tp={0} tn={1} \nfn={2} fp={3}'.format( tp, tn, fn, fp))
print(f"Accuracy: {accuracy}")
print(f"AUC: {AUC}")

# save the model, the path needs to be in Environment variables after refactoring
model_name = "pred_RUL.pkl"
path = "./model/"
save_model(scaler, xgbr_model, path, model_name )