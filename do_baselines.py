import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from baseline_models import *
from bc_subtypes import *


data_name = 'NEO'
outcome_name = 'resp.pCR'
treatment_plans = ['TP1','TP2','TP3','TP4']
remove_attributes = ['Trial.ID','resp.Chemosensitive','resp.Chemoresistant','RCB.score','RCB.category',
                  'Chemo.NumCycles','Chemo.first.Taxane','Chemo.first.Anthracycline', 'Chemo.second.Taxane',
                  'Chemo.second.Anthracycline', 'Chemo.any.Anthracycline','Chemo.any.antiHER2'] #remove 12 features

base_folder = os.getcwd()
inputPath = os.path.join(base_folder, 'input', data_name)
outputPath = os.path.join(base_folder, 'output')

threshold = -999 ## A threshold -999 means using all predited treatment effect values to compare.
random_seed = 42
fold = 5
test_size = 0.30


def split_data(base_folder, fold, test_size, data_name):
    inputPath = os.path.join(base_folder, 'input', data_name)
    input_data = f"{inputPath}/{data_name}.csv"

    dataset = pd.read_csv(input_data, encoding="ISO-8859-1", engine='python')
    fileCount = 1
    for icount in range(40, 40 + fold):
        Mtrain, Mtest = train_test_split(dataset, test_size=test_size, random_state=icount, shuffle=True)
        Mtrain.reset_index(drop=True, inplace=True)
        Mtest.reset_index(drop=True, inplace=True)
        MtrainPath = os.path.join(inputPath, data_name + '_' + 'train' + '_' + str(fileCount) + '.csv')
        Mtrain.to_csv(MtrainPath, index=False)
        MtestPath = os.path.join(inputPath, data_name + '_' + 'test' + '_' + str(fileCount) + '.csv')
        Mtest.to_csv(MtestPath, index=False)
        fileCount = fileCount + 1
# split_data(base_folder, fold, test_size, data_name)


model_validation_fold_allTP(data_name, treatment_plans, outcome_name, remove_attributes, threshold, base_folder, random_seed)  #Do LorgRa, SvmRa, RfRa methods
model_validation_fold_eachTP(data_name, treatment_plans, outcome_name, remove_attributes, threshold, base_folder, random_seed)  #Do LorgRe, SvmRe, RfRe methods
current_protocol_validation_fold(data_name, treatment_plans, outcome_name, remove_attributes, threshold, base_folder, random_seed)  #Calculate Recovery Rate for The Current Protocol

