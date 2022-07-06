pset2
==============================
This project depicts e2e implementation of mlops for a machine learning model using the below set of tools. The ML mpdel trained in this project predicts the chances of person having diabetes based on certain parameters like BloodSugarLevel, Glucose, skinthickness etc. The model is trained using AdaBoostClassifier and data source is kaggle. <br/>
1. CookieCutter - to provide the proper folder structure or scaffolding, so that modularity of the code is maintained.<br/>
Project Organization <br/>
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
2.The visual studio code is used as IDE during development on local machine. 
3. Git hub is used as code versioning tool and DVC as data versioning tool.
4. The entire training process is divided into set of stages and mentioned in the dvc.yaml file 
   The source data is placed under data/external in the form of csv. The stages mentioned in the dvc.yaml executes in sequential order. The example of 2 different stages are added for reference. <br/>
               stages: <br/>
              raw_dataset_creation: <br/>
                cmd: python src/data/load_data.py --config=params.yaml <br/>
                deps: <br/>
                - src/data/load_data.py <br/>
                - data/external/train.csv <br/>
                outs: <br/>
                - data/raw/train.csv <br/>
              split_data: <br/>
                cmd: python src/data/split_data.py --config=params.yaml <br/>
                deps: <br/>
                - src/data/split_data.py <br/>
                - data/raw/train.csv <br/>
                outs: <br/>
                - data/processed/diabetes_train.csv <br/>
                - data/processed/diabetes_test.csv  <br/>
    
   In the current project, the stages are divided as 
   raw_dataset_creation  - load the data from csv and convert it to pandas dataframe
   split_data - split the data into train and test split 
   model_train - train the data using AdaboostClassifier from sklearn with 2 parameters n_estimators and learning_rate.  <br/>
5. All these configuration details related stages has been added in params.yaml <br/>
6. whenver the code change is done , used git commands to add(git add filename) , commit(git commit -m 'message') and push(git push origin brnach) it to remote repo
7. As DVC is used, the data files will not be stored on git, instead only a refernce to the data file will be added in git ex. train.csv.dvc <br/>
8. new data files are added to dvc using dvc add command <br/>
9. All the dependencies required for the project is mentioned in requirements.txt
10. CML is integrated , through git hub workflow, which monitors metric evaluation, comparing ML experiments across your project history <br/>
11. whenever a new code is pushed remote repo, the cml.yaml workflow will initiate,executes the job mentioned and output the metrics to metrics.txt and confusio matrix as an image file. <br/>
12.Further new branches are created and some parameters of the models are tuned and pushed to the repo, cml.yaml initiated the workflow and shows the differnece between metrics. Later we can merge the new branch with better results(trial1) to the main branch. <br/>
13. After all this setup, the code is moved to aws container, using cloud9 service of aws console. <br/>

  


--------

