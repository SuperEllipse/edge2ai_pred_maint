# edge2ai_pred_maint 
### Note : The readme will be edited in the future
#### Goal / Purpose:
Demonstrate the ability to demonstrate the ability to deploy Machine Learning applications at the Edge using key components of Cloudera Platform. 
To Be expanded further. 

#### Use Case: 
https://blog.cloudera.com/using-cml-to-build-a-predictive-maintenance-model-for-jet-engines/ 


#### Folder : 
1. __archive:__ earlier work done in Cloudera 
2. __data:__ Train Dataset, Test Dataset , Ground Truth for the test dataset
3. __src:__ 
>> a. Predictive Maintenance.ipynb - This jupyter notebook runs with the end to end data analysis. ( Note: Will be updated to into a python file (.py) in CML that could then be run as job or experiments) <br>
>> b. run_inference_test.py - Will be run at edge everytime a new model is deployed to ensure model sanity checks are performed <br>
>> c. utils.py - contains utility functions e.g. Feature extractions <br>

4.__models:__ The serialized models are saved here. Need to now create functionality to call the the API End point provided to transfer these models to the EFP Server ( Nifi node)  
