# **Predictive Maintenance at the Edge**
## **Motivations:**
Predictive maintenance stems from the goal of predicting the future trend of equipment condition. This condition based monitoring drives the goal of performing asset maintenance at a scheduled point in a cost effective manner, with an objective of reducing unplanned downtime avoiding loss of revenue, significant  cost reductions from improved asset lifespan and fines from loss of delivery. 

Traditional computing architecture based predictive analytics engine relies on the model training and serving computations hosted on an on prem, cloud or hybrid infrastructure. The engine is invoked by a remote asset over a network with data on which the engine provides the inferences. This to and fro information exchange between the data generating asset and the predictive engine suffers from latency, potential dependence on network bandwidth and stability.
 
 As connected devices proliferate in Industrial assets, there will be use cases of computation locally on these devices driven by limited or intermittent network connectivity, need for real-time decision making and XX. Additionally, Edge data processing and computations are made on devices with constrained power, compute and storage capabilities. 
Cloudera Edge Management (CEM) manages, controls, and monitors data collection and processing at the edge with a low code authorship experience addressing data management challenges with streaming and IoT use cases.

It provides two categories of capabilities:<br>
* **Edge Data Collection:** MiNiFi is a lightweight edge agent that implements the core features of Apache NiFi, focusing on data collection and processing at the edge. The MiNiFi agents come in two flavors: MiNiFi Java agents for full capabilities of Apache NiFi and MiNiFi C++ for very low footprint agents
* **Edge Flow Management:** Edge Flow Manager is an agent management hub that provides a low-code experience for designing, deploying, and monitoring edge flow applications on thousands of MiNiFi agents. It also acts as the single management and monitoring layer for all the MiNiFi agents deployed at the edge. EFM supports the entire edge flow lifecycle including authorship, deployment, and monitoring  

## **High-level Architecture**

![Architectural View](./others/Architecture.png?raw=true "Optional Title")

## **Structure**
````
.
|-- data  # The dataset in use: data1 for e.g has 27 variables including 2 settings and 15 sensor readings
|-- jobs  # This folder will have jobs(tbd) to transfer the model to a remote server for edge deployment 
|-- model # The serialized model .pkl file will be be persisted here
|-- src   # All the source code for the project including setup files
    |-- Predictive Maintenance.ipynb  #Notebook that describes the workflow from dataload to model persist
    |-- run_inference_test.py # Tests that model can be loaded from disc and serves inference
|-- images # image file for the architecture etc. 
````

## **Use Case Description:** 


https://blog.cloudera.com/using-cml-to-build-a-predictive-maintenance-model-for-jet-engines/ 


#### Folder : 
1. __archive:__ earlier work done in Cloudera 
2. __data:__ Train Dataset, Test Dataset , Ground Truth for the test dataset
3. __src:__ 
>> a. Predictive Maintenance.ipynb - This jupyter notebook runs with the end to end data analysis. ( Note: Will be updated to into a python file (.py) in CML that could then be run as job or experiments) <br>
>> b. run_inference_test.py - Will be run at edge everytime a new model is deployed to ensure model sanity checks are performed <br>
>> c. utils.py - contains utility functions e.g. Feature extractions <br>

4.__models:__ The serialized models are saved here. Need to now create functionality to call the the API End point provided to transfer these models to the EFP Server ( Nifi node)  