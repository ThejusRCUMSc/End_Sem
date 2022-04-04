import os, sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=62)

from lightgbm import LGBMClassifier

def write_to_csv():
    a = '"0","1","2","3","4","5","6","7","8","9","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff","??","size","Class"'
    
    k=0
    files = os.listdir("data/test")
    files.sort()
    train_labels=pd.read_csv("data/trainLabels.csv")
    train_np = np.zeros((len(files),len(a.split(","))))
    
    for file_ in tqdm(files):
        statinfo = os.stat("data/test/"+file_)
        with open("data/test/"+file_,"r") as fp:  
            for lines in fp.readlines():
                line=lines.rstrip().split(" ")[1:]
                for hex_code in line:
                    if hex_code=='??':
                        train_np[k][256]+=1
                    else:
                        train_np[k][int(hex_code,16)]+=1
            train_np[k][257]=statinfo.st_size/(1024*1024)
            train_np[k][258]=train_labels[train_labels["Id"]==file_.split('.')[0]]["Class"].tolist()[0]
        fp.close()
        k +=1
    train = pd.DataFrame(train_np, columns=a[1:-1].split('","'))
    train.to_csv("data/train_data.csv", index=False)
    train.head()

def LGBMClass(X,y,stack_df):
    for i,j in enumerate(fold.split(X,y)):
        stack_train_X = X.iloc[j[0]]
        stack_train_y = y.iloc[j[0]]
        stack_test_X = X.iloc[j[1]]
        stack_test_y = y.iloc[j[1]]
        model = LGBMClassifier(learning_rate= 0.025, n_estimators = 850, min_child_weight = 1, boosting_type = "gbdt", min_child_samples=68,random_state = 62,objective = "multi-class",metric = "multi_logloss")
        model.fit(stack_train_X,stack_train_y)
        preds = model.predict_proba(stack_test_X)
        preds = pd.DataFrame(preds)
        stack_df["0"].iloc[j[1]] = preds[0]
        stack_df["1"].iloc[j[1]] = preds[1]
        stack_df["2"].iloc[j[1]] = preds[2]
        stack_df["3"].iloc[j[1]] = preds[3]
        stack_df["4"].iloc[j[1]] = preds[4]
        stack_df["5"].iloc[j[1]] = preds[5]
        stack_df["6"].iloc[j[1]] = preds[6]
        stack_df["7"].iloc[j[1]] = preds[7]
        stack_df["8"].iloc[j[1]] = preds[8]
    lgbm_stack = stack_df[["0","1","2","3","4","5","6","7","8"]]
    lgbm_stack.columns = ["lgbm_Prediction1","lgbm_Prediction2","lgbm_Prediction3","lgbm_Prediction4","lgbm_Prediction5","lgbm_Prediction6","lgbm_Prediction7","lgbm_Prediction8","lgbm_Prediction9"]
    print(lgbm_stack)

def modelling():
    train = pd.read_csv("data/train_data.csv")
    X = train.drop(["Class"], axis=1)
    y = train["Class"]
    stack_df = pd.read_csv("data/train_data.csv")
    
    LGBMClass(X,y,stack_df) 


def main():
    #write_to_csv()
    modelling()


if __name__=="__main__":
    main()
