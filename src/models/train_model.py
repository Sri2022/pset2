import json
import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report
from sklearn.metrics import ConfusionMatrixDisplay

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def accuracymeasures(y_test,predictions,avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = ['0','1']
    print("Classification report ")
    print("---------------------","\n")
    print(classification_report(y_test, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy,precision,recall,f1score

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y    

def train_and_evaluate(config_path):
    print('Train model - in function   ')
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    n_estimators=config["ada_boost"]["n_estimators"]
    learning_rate=config["ada_boost"]["learning_rate"]


    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

    #model
    model = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    accuracy,precision,recall,f1score = accuracymeasures(test_y,y_pred,'weighted')
    with open("metrics.txt", "w") as outfile:
        outfile.write("Param - n_estimators : " + str(n_estimators) + "\n")
        outfile.write("Param - learning_rate : " + str(learning_rate) + "\n")
        outfile.write("Accuracy: " + str(accuracy) + "\n")
        outfile.write("Precision: " + str(precision) + "\n")
        outfile.write("Recall: " + str(recall) + "\n")
        outfile.write("f1_score: " + str(f1score) + "\n")

    disp = ConfusionMatrixDisplay.from_estimator(model, test_x, test_y, normalize="true", cmap=plt.cm.Blues)
    plt.savefig("plot.png")
    print('accuracy,precision,recall,f1score ',accuracy,precision,recall,f1score )
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
