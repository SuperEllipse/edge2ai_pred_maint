import pandas as pd
import numpy as np

COLUMNNAMES = [                                                                
                "ENGINE_NUMBER"     
              , "TIME_IN_CYCLES"  
              , "SETTING_1"       
              , "SETTING_2"       
              , "TRA"             
              , "T2"              
              , "T24"             
              , "T30"             
              , "T50"             
              , "P2"              
              , "P15"             
              , "P30"             
              , "NF"              
              , "NC"              
              , "EPR"             
              , "PS30"            
              , "PHI"             
              , "NRF"             
              , "NRC"             
              , "BPR"             
              , "FARB"            
              , "HTBLEED"         
              , "NF_DMD"          
              , "PCNFR_DMD"       
              , "W31"             
              , "W32"             
#              , "RUL"                                 # Column not present in Test Record
  ]


COLUMNSTOREMOVE = [                                        \
                "ENGINE_NUMBER"     # NO INFLUENCE TO MODEL  \
              , "TIME_IN_CYCLES"  # NO INFLUENCE TO MODEL  \
              , "TRA"             # NO INFLUENCE TO MODEL  \
              , "T2"              # NO INFLUENCE TO MODEL  \
              , "P2"              # NO INFLUENCE TO MODEL  \
              , "EPR"             # NO INFLUENCE TO MODEL  \
              , "FARB"            # NO INFLUENCE TO MODEL  \
              , "NF_DMD"          # NO INFLUENCE TO MODEL  \
              , "PCNFR_DMD"       # NO INFLUENCE TO MODEL  \
              ]



 ## Vish : Function takes a record and extracts relevant features from the Dataset 
def prepare_test_data(test_record) : 
# check for mismatch of columns 
    try:
        if len(test_record) != len(COLUMNNAMES):
            raise ValueError
    except ValueError:
        print('Invalid Test Data Format:: \n  Expected # of Columns {0:d} \n Actual  # of Columns {1:d}' \
              .format( len(COLUMNNAMES), len(test_record)))
        raise

    array_record = np.asarray(test_record).reshape((-1), len(COLUMNNAMES))
    #print(array_record)
    X_test = pd.DataFrame(array_record, columns= COLUMNNAMES)
    #print(X_test)  
              
    # drop columns that have no impact on Models
    X_test.drop(labels=COLUMNSTOREMOVE, axis=1, inplace=True)
    #print(X_test)
    return X_test.to_numpy()

def run_model_inference_reg( model, scaler, test_record) :    
  X_test = prepare_test_data(test_record)
  X_test_scl = scaler.transform(X_test)
  return model.predict(X_test_scl)   


def run_model_inference_clf( model, scaler, test_record) :    
  X_test = prepare_test_data(test_record)
  X_test_scl = scaler.transform(X_test)
  return model.predict(X_test_scl), model.predict_proba(X_test_scl)