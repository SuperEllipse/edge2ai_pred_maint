
## 
## Purpose : The purpose of this code is to run the inference model for predictive maintenance 
## Function Call : 
##          python3 run_inference.py <path to model.pkl> <sample record as a list>
##          example call here from the base project directory
##          python3 ./src/run_inference_test.py "./model/pred_RUL.pkl"  "[1,4,0.0042,0.0000,100.0,518.67,642.44,1584.12,1406.42,14.62,21.61,554.07,2388.03,9045.29,1.30,47.28,521.38,2388.05,8132.90,8.3917,0.03,391,2388,100.00,39.00,23.3737]"
##          Author: Vish Rajagopalan

import numpy as np
import sys
import ast
import pickle
import importlib as imp 
from serv_utils import run_model_inference_reg, run_model_inference_clf

#imp.reload(serv_utils)

def main():
    if  len(sys.argv) != 3 :
        print('Incorrect format : needs 3 arguments: eg. python3 run_inference.py pred_RUL.pkl <record-input of 26 values>') 
        sys.exit()
    else:
        filename = sys.argv[1]

        test_record  = ast.literal_eval( sys.argv[2] )
#        print('record :', type(test_record)) # DEBUG
        with open(filename, 'rb') as f:
            serialized_list = pickle.load(f)
        scaler = serialized_list[0]
        model_reg = serialized_list[1]
        model_clf = serialized_list[2]

        ## Check that the model prediction works ok here
        try :
            sample_record = [1,1,0.0023,0.0003,100.0,518.67,643.02,1585.29,1398.21,14.62,21.61,553.90,2388.04,9050.17,1.30,47.20,521.72,2388.03,8125.55,8.4052,0.03,392,2388,100.00,38.86,23.3735]
            # Get the inference from the Remaining Useful life Regressor i.e Remaining useful life prediction in X Timecycles
            y_check = run_model_inference_reg(model_reg, scaler, sample_record)
            if round(y_check[0]) > 135  or round(y_check[0]) < 120 :
                raise ValueError 
        except ValueError:
            print('Invalid Value from Predict Model, expecting result in the range generally between 120 and 135, getting {0}'.format(round(y_check[0])))
            raise    

        # Get the inference from the Remaining Useful life Regressor i.e Remaining useful life prediction in X Timecycles
        y_test = run_model_inference_reg(model_reg, scaler, test_record)
        print('Test value of prediction : {0}'.format(y_test))

        # Get the inference from the Remaining Useful life clasifier around Failure Probabilities 
        fail_pred, prob = run_model_inference_clf(model_clf, scaler, test_record)
        print(fail_pred, prob) # Expected values [0] [[0.87898725 0.12101275]]

if __name__ == "__main__":  
    main()
