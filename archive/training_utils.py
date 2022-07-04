import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def trainDataPlot(dataFrame):
  colNames = [                                                                \
#               "ENGINE_NUMBER"     # NO INFLUENCE TO MODEL                     \
#             , "TIME_IN_CYCLES"  # NO INFLUENCE TO MODEL                     \
                "SETTING_1"       # r = -0.0032    | r-squared =  0.00001024  \
              , "SETTING_2"       # r = -0.0019    | r-squared =  0.00000361  \
#             , "TRA"             # NO INFLUENCE TO MODEL                     \
#             , "T2"              # NO INFLUENCE TO MODEL                     \
              , "T24"             # r = -0.61      | r-squared =  0.3721      \
              , "T30"             # r = -0.58      | r-squared =  0.3364      \
              , "T50"             # r = -0.68      | r-squared =  0.4624      \
#             , "P2"              # NO INFLUENCE TO MODEL                     \
              , "P15"             # r = -0.13      | r-squared =  0.0169      \
              , "P30"             # r =  0.66      | r-squared =  0.4356      \
              , "NF"              # r = -0.56      | r-squared =  0.3136      \
              , "NC"              # r = -0.39      | r-squared =  0.1521      \
#             , "EPR"             # NO INFLUENCE TO MODEL                     \
              , "PS30"            # r = -0.7       | r-squared =  0.49        \
              , "PHI"             # r =  0.67      | r-squared =  0.4489      \
              , "NRF"             # r = -0.56      | r-squared =  0.3136      \
              , "NRC"             # r = -0.31      | r-squared =  0.0961      \
              , "BPR"             # r = -0.64      | r-squared =  0.4096      \
#             , "FARB"            # NO INFLUENCE TO MODEL                     \
              , "HTBLEED"         # r = -0.61      | r-squared =  0.3721      \
#             , "NF_DMD"          # NO INFLUENCE TO MODEL                     \
#             , "PCNFR_DMD"       # NO INFLUENCE TO MODEL                     \
              , "W31"             # r =  0.63      | r-squared =  0.3969      \
              , "W32"             # r =  0.64      | r-squared =   0.4096     \
              , "RUL"             # COMPARE ALL TO THIS                       \
  ]

  for col in colNames:
    if col != "RUL":
      plt.title(col)
      plt.ylabel('RUL')
      plt.xlabel(col)
      plt.plot(dataFrame[col], dataFrame['RUL'], 'ro')
      plt.show()



def trainDataCorrelation(dataFrame):
  df_filtered = dataFrame.drop(columns = [                                    \
                "ENGINE_NUMBER"     # NO INFLUENCE TO MODEL                     \
              , "TIME_IN_CYCLES"  # NO INFLUENCE TO MODEL                     \
#             , "SETTING_1"       # r = -0.0032    | r-squared =  0.00001024  \
#             , "SETTING_2"       # r = -0.0019    | r-squared =  0.00000361  \
              , "TRA"             # NO INFLUENCE TO MODEL                     \
              , "T2"              # NO INFLUENCE TO MODEL                     \
#             , "T24"             # r = -0.61      | r-squared =  0.3721      \
#             , "T30"             # r = -0.58      | r-squared =  0.3364      \
#             , "T50"             # r = -0.68      | r-squared =  0.4624      \
              , "P2"              # NO INFLUENCE TO MODEL                     \
#             , "P15"             # r = -0.13      | r-squared =  0.0169      \
#             , "P30"             # r =  0.66      | r-squared =  0.4356      \
#             , "NF"              # r = -0.56      | r-squared =  0.3136      \
#             , "NC"              # r = -0.39      | r-squared =  0.1521      \
              , "EPR"             # NO INFLUENCE TO MODEL                     \
#             , "PS30"            # r = -0.7       | r-squared =  0.49        \
#             , "PHI"             # r =  0.67      | r-squared =  0.4489      \
#             , "NRF"             # r = -0.56      | r-squared =  0.3136      \
#             , "NRC"             # r = -0.31      | r-squared =  0.0961      \
#             , "BPR"             # r = -0.64      | r-squared =  0.4096      \
              , "FARB"            # NO INFLUENCE TO MODEL                     \
#             , "HTBLEED"         # r = -0.61      | r-squared =  0.3721      \
              , "NF_DMD"          # NO INFLUENCE TO MODEL                     \
              , "PCNFR_DMD"       # NO INFLUENCE TO MODEL                     \
#             , "W31"             # r =  0.63      | r-squared =  0.3969      \
#             , "W32"             # r =  0.64      | r-squared =   0.4096     \
#             , "RUL"             # COMPARE ALL TO THIS                       \
    ], axis=1, inplace=False)

  df_filtered.describe()
  corrMatrix = df_filtered.corr()
  sns.heatmap(corrMatrix, annot=True)
  plt.show()




def makeTestData(filename, model):
  data = pd.read_csv(filename)

  #---------------------------------------------
  # X_test:
  #     Data for the last point for each engine
  #---------------------------------------------
  X_test = pd.DataFrame(data)

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
  X_test.drop(columns = COLUMNSTOREMOVE, inplace=True)
  X_test.drop(columns = ["MAX","DELETE"], inplace=True)
  X_test = X_test.values



  #---------------------------------------------
  # pred_RUL:
  #     model predicted values for engine
  #---------------------------------------------
  test = pd.DataFrame(data)
  test.drop(columns = COLUMNSTOREMOVE, inplace=True)

  final_df = pd.DataFrame()
  final_df["ENGINE_NUMBER"] = data.ENGINE_NUMBER.values
  final_df["TIME_IN_CYCLES"] = data.TIME_IN_CYCLES.values
  # Vish : final_df["PREDICTED_RUL"] = model.predict(data = test.values) throws an error , need to remove data  paramenter assigment
  final_df["PREDICTED_RUL"] = model.predict(test.values)

  y_pred = final_df.groupby('ENGINE_NUMBER')['PREDICTED_RUL'].nth(-1)
  pred_RUL = pd.DataFrame(y_pred)


  return pred_RUL, X_test