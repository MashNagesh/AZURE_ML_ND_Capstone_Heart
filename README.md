#  MNIST CLASIFICATION -CAPSTONE PROJECT - AZURE ML ENGINEER 

This Capstone project is part of the Azure ML Engineer NanoDegree.The Key components that were covered as part of the project areas follows:

1.Traning and identifying the Best model Run using AUTOML

2.Traning and identifying the Best model Run using Hyperdrive

3.Deploying the best model from the above 2 steps  and testing the functioning of the API


## Dataset

### Overview
This dataset contains 12 features that can be used to predict mortality by heart failure (indicated by the variable DEATH_EVENT).
Below are the feaures

1. age

2. anaemia

3.creatinine_phosphokinase

4.diabetes

5.ejection_fraction

6.high_blood_pressure

7.platelets

8.serum_creatinine

9.serum_sodium

10.sex

11.smoking

12.time


Source of the data: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv


### Task
To predict the DEATH_EVENT  column 

The training data set, (train.csv), has 13 columns. The first column, called 'DEATH_EVENT'which is the dependent variable. The Other columsn provide medical features of a person.The objective is to predict the DEATH_EVENT column using the other columns which provide the medical condition.

### Access
The data is accessed in the AzureML notebook using Kaggle API.

#### Steps:

1.Install Kaggle

2.Setup the Directory structure

3.Generate API token from Kaggle Account Page

4.Upload the Kaggle.Json containing UserName and Key

5.Once uploaded use chmod to change access permissions

6.Download the CompetitionZip file into the data directory

Reference:https://inclusive-ai.medium.com/how-to-use-kaggle-api-with-azure-machine-learning-service-da056708fc5a

## Automated ML

#### Choice of AutoML Settings:

1. n_cross_validation
Indicates how many cross validations to perform and in our case splitting it into 5 portions will ensure that we have ~8000 records for training and ~2000 for validation.

2. Primary Metric
Primary metric chosen here is accuracy to understand how much of the sample has been correctly classified.We could also use AUC as metric where we can see multiple one versus all Precision recall curves for each of the MNIST digits

3. enable early stopping
Early stopping is enabled to prevent overfitting

4. Experiment Stop time
To handle costs and time

5. Remote compute
Going for Remote Compute to avoid dependency issues which came up while using Local compute

### Results
There are about 11 models which have run as a part of this experiment.

Voting Ensemble is the top performing model w.r.t the Primary metric.

The Ensemble models perform better as opposed to the individual models since they combine bagging,bosting and stacking to provide the results.
They also combine the results and minimise the variance component of the error.


#### Results from AUTOML run using RunDetails
The Voting Ensemble models performs the best interms of the Primary Metric - Accuracy

![image](https://user-images.githubusercontent.com/26400438/127836178-e9a80615-a885-4a09-b7a2-f259cd44ac93.png)

#### Best Model trained Parameters
Parameters of the Best Model - Voting Ensemble.The best model is a combination of the below submodels with appropriate weights mentioned in the brackets

1.SparseNormalizer, XG Boost(0.4) -

{
        "booster": "gbtree",
        "colsample_bytree": 0.9,
        "eta": 0.3,
        "gamma": 0,
        "max_depth": 10,
        "max_leaves": 15,
        "n_estimators": 25,
        "objective": "reg:logistic",
        "reg_alpha": 0,
        "reg_lambda": 0.5208333333333334,
        "subsample": 0.6,
        "tree_method": "auto"
    }

2.StandardScalerWrapper, Random Forest(0.1) -
    {
        "bootstrap": false,
        "class_weight": null,
        "criterion": "entropy",
        "max_features": 0.2,
        "min_samples_leaf": 0.01,
        "min_samples_split": 0.10368421052631578,
        "n_estimators": 25,
        "oob_score": false
    }

3.MaxAbsScaler, Gradient Boosting(0.1) -

    {
        "criterion": "friedman_mse",
        "learning_rate": 0.046415888336127774,
        "max_depth": 4,
        "max_features": 0.4,
        "min_samples_leaf": 0.08736842105263157,
        "min_samples_split": 0.5252631578947369,
        "n_estimators": 100,
        "subsample": 1
    }


4.MinMaxScaler, SVM (0.1) -
    
    {
        "C": 159.98587196060572,
        "class_weight": null,
        "kernel": "rbf"
    }

5.MinMaxScaler, RandomForest (0.1)-
    
    {
        "bootstrap": true,
        "class_weight": null,
        "criterion": "gini",
        "max_features": "sqrt",
        "min_samples_leaf": 0.035789473684210524,
        "min_samples_split": 0.01,
        "n_estimators": 10,
        "oob_score": true
    }

6.MinMaxScaler, RandomForest (0.1) - 

    {
        "bootstrap": false,
        "class_weight": null,
        "criterion": "gini",
        "max_features": "sqrt",
        "min_samples_leaf": 0.01,
        "min_samples_split": 0.10368421052631578,
        "n_estimators": 10,
        "oob_score": false
    }

6.MaxAbsScaler, Light GBM (0.1) - :"min_data_in_leaf" : 20}



#### Scope of Improvement:

1.*Limitations of the Data set* :The dataset has close to 300 records which is a very small sample for an ML model and also we ahve only 12 columns provided which is not an exhaustive medical record for a person.There could be other variables which could act as highly predictive features.

2. *Primary Metric* : The Primary metric used is Accuracy.The dataset provided is an imbalanced dataset hence accuracy is not an appropriate metric.Also since the implications of missing out on predicting a death_event is huge it is important to chose our metric which places emplasis on **higher recall** rate since it would be costly to miss a death event.High Recall would be a better metric

3. *Time* : experiment_timeout_minutes si 20 minutes and can be extended to check performances of other models

#### Best Model trained Metrics:
The Primary metric used for model evaluation is Accuracy in this case.However we are able to see good values across multiple evlauation metrics for the best model

![image](https://user-images.githubusercontent.com/26400438/127838665-8d3620fe-968e-4543-bd87-82b072e06bcc.png)

## Hyperparameter Tuning

#### Choice of Model
The model being used is a simple Logistic regression. The focus of this excercise has been to understand the features of hyperdrive and to try out the same.

#### Early termination Policy
MedianStopping is a Conservative policy that provides savings without terminating promising jobs.It computes running averages across all runs and cancels runs whose best performance is worse than the median of the running averages.

#### Sampling Policy
The sampling Policy used is a Random Sampling Policy since the grid search suffers from limitations pertaining to higher dimensionality issues and Random Sampling though it functions very similar to grid search has been able to provide equal or better results in many scenarios. The chances of finding the optimal parameter are comparatively higher in random search because of the random search pattern where the model might end up being trained on the optimised parameters.

Range and values of Parameters as stated below 
'C': uniform(0.1, 10),'max_iter': choice(50,100,200,300)

#### Hyperparamters
Below hyperparameters are tuned in this model

C - Inverse of Regularisation strength

Max_iter - Maximum number of iterations to converge


### Results

Below are the results from the HyperDrive model

#### Run Details Widget - Models
![image](https://user-images.githubusercontent.com/26400438/127840221-4a3acf72-9593-4b07-bd15-bf7e609a4bb8.png)

#### Capturing the Logs
![image](https://user-images.githubusercontent.com/26400438/127840346-fca56978-506f-4dcb-96e9-ff57a215e00b.png)


#### Results from the Best Model 

The best model has an accuracy of 0.75

![image](https://user-images.githubusercontent.com/26400438/127840535-a2f53df0-317c-4e4e-817f-c3931fe7ec29.png)

#### Parameters of the Best Model

1.C = 8.94

2.max_iter = 100

![image](https://user-images.githubusercontent.com/26400438/127840631-125f70dd-5a6f-488f-b61f-978ad3b03ba6.png)

#### Scope for Improvement

1. Choice of Model - Logistic Regression is the simplest choice among the classification models.The next experiment can be explored with some bagging or boosting algorithms.

2. For the experiment only 2 parameters (C and max_iter) have been sampled.We could try playing with other parameters like the penalty,Multi_class options.

3. Using Bayesian Sampling - This would be a cost performance tradeoff since Bayesian sampling since this method picks samples based on how previous samples did, so that new samples improve the primary metric.

## Model Deployment

Best runs from both the models are registered

![model_list](https://user-images.githubusercontent.com/26400438/127418941-5187b1e9-2f67-4cee-9419-8d250b6fed3d.PNG)

Comparing the performances of the Best models from the AutoML Run and the Hyperdrive Run

| Method      | Primary Metric |
| ----------- |--------------- |
| AutoML      |0.87            |
| HyperDrive  |0.75            |

The model chosen for deployment is the Best Childrun from the AUTOML which is the Voting Ensemble model and is deployed as ACI Webservice.When we deploy a model as an ACIWebService in Azure Machine Learning Service, we do not need to specify any deployment_target but we need to provide the Environment details (file: env.yaml)

#### Steps:

##### Inference Configuration

1.Define the Environment(env.yaml)

2.Use Scoring Script(Score1.py) and the Saved Environment details  to define the configuration for Inference

##### Deployment Configuration

3.ACIWebservice spins an instance based on the Specification

Snapshot of Deployed model with the rest API
![image](https://user-images.githubusercontent.com/26400438/127841034-0f7ea44b-fb30-4fa5-9a2f-e53721cee61c.png)


![image](https://user-images.githubusercontent.com/26400438/127841162-343407d2-4844-4731-9bf1-e74d55663dab.png)



##### Querying the deployed rest API:

The Consume section in the API provides details on querying the API.Below is the screen shot of steps used in scoring few records in one go
Using the run function returns the response.

    response = aci_service.run(test_data)

![image](https://user-images.githubusercontent.com/26400438/127841262-06c099c6-7e3e-4098-b354-f965318afb00.png)

#### Log of Web Service

![image](https://user-images.githubusercontent.com/26400438/127841365-529eda83-2dd1-4310-9326-ad2bab01f641.png)

## Screen Recording
Link to the screen recording




