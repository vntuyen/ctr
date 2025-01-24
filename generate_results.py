import os
import sys
import statistics 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl, matplotlib.pyplot as plt
import os.path
from os import path
import random
import matplotlib.pyplot as plt


from bc_subtypes import *

data_name = 'NEO'
outcome_name = 'resp.pCR'

base_folder = os.getcwd()
inputPath = os.path.join(base_folder, 'input', data_name)
outputPath = os.path.join(base_folder, 'output')

threshold = -999 ## A threshold -999 means using all predited treatment effect values to compare.


def estimateQiniCurve(estimatedImprovements, outcomeName, modelName):
    ranked = pd.DataFrame({})
    ranked['uplift_score'] = estimatedImprovements['Improvement']
    ranked['NUPLIFT'] = estimatedImprovements['UPLIFT']
    ranked['FollowRec'] = estimatedImprovements['FollowRec']
    ranked[outcomeName] = estimatedImprovements[outcomeName]
    ranked['countnbr'] = 1
    ranked['n'] = ranked['countnbr'].cumsum() / ranked.shape[0]
    uplift_model, random_model = ranked.copy(), ranked.copy()
    C, T = sum(ranked['FollowRec'] == 0), sum(ranked['FollowRec'] == 1)
    ranked['CR'] = 0
    ranked['TR'] = 0
    ranked.loc[(ranked['FollowRec'] == 0)
               & (ranked[outcomeName] == 1), 'CR'] = ranked[outcomeName]
    ranked.loc[(ranked['FollowRec'] == 1)
               & (ranked[outcomeName] == 1), 'TR'] = ranked[outcomeName]
    ranked['NotFollowRec'] = 1
    ranked['NotFollowRec'] = ranked['NotFollowRec'] - ranked['FollowRec']
    ranked['NotFollowRecCum'] = ranked['NotFollowRec'].cumsum()
    ranked['FollowRecCum'] = ranked['FollowRec'].cumsum()
    ranked['CR/C'] = ranked['CR'].cumsum() / ranked['NotFollowRec'].cumsum()
    ranked['TR/T'] = ranked['TR'].cumsum() / ranked['FollowRec'].cumsum()
    # Calculate and put the uplift into dataframe
    uplift_model['uplift'] = round((ranked['TR/T'] - ranked['CR/C']) * ranked['n'], 5)
    uplift_model['uplift'] = round((ranked['NUPLIFT']) * ranked['n'], 5)
    uplift_model['grUplift'] = ranked['NUPLIFT']
    random_model['uplift'] = round(ranked['n'] * uplift_model['uplift'].iloc[-1], 5)
    ranked['uplift'] = ranked['TR/T'] - ranked['CR/C']
    # Add q0
    q0 = pd.DataFrame({'n': 0, 'uplift': 0}, index=[0])
    uplift_model = pd.concat([q0, uplift_model]).reset_index(drop=True)
    random_model = pd.concat([q0, random_model]).reset_index(drop=True)
    # Add model name & concat
    uplift_model['model'] = modelName
    random_model['model'] = 'Random model'
    return uplift_model


def areaUnderCurve(models, modelNames):
    modelAreas = []
    for modelName in modelNames:
        area = 0
        tempModel = models[models['model'] == modelName].copy()
        tempModel.reset_index(drop=True, inplace=True)
        for i in range(1, len(tempModel)):  # df['A'].iloc[2]
            delta = tempModel['n'].iloc[i] - tempModel['n'].iloc[i - 1]
            y = (tempModel['uplift'].iloc[i] + tempModel['uplift'].iloc[i - 1]) / 2
            area += y * delta
        modelAreas.append(area)
    return modelAreas


def getAUUCTopGroup(FolderLocation, fold, prefileName, postfileName, outcomeName, plotFig=True):
    improvementMtreeModels = []
    for fileCount in range(1, fold + 1):
        improveFilePath = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '.csv')
        if (not (path.exists(improveFilePath))):
            continue
        results = pd.read_csv(improveFilePath, encoding="ISO-8859-1", engine='python')
        if (not ('Improvement' in results.columns)):
            results['Improvement'] = results['LIFT_SCORE']
        if (not ('FollowRec' in results.columns)):
            results['FollowRec'] = results['FOLLOW_REC']

        newImprovement = estimateQiniCurve(results, outcomeName, 'Tree')
        improvementMtreeModels.append(newImprovement)
    improvementMtreeCurves = pd.DataFrame({})
    improvementMtreeCurves['n'] = improvementMtreeModels[0]['n']
    improvementMtreeCurves['model'] = improvementMtreeModels[0]['model']
    icount = 1
    modelNames = []
    groupModelNames = []
    for eachM in improvementMtreeModels:
        improvementMtreeCurves['uplift' + str(icount)] = eachM['uplift']
        modelNames.append('uplift' + str(icount))
        improvementMtreeCurves['grUplift' + str(icount)] = eachM['grUplift']
        groupModelNames.append('grUplift' + str(icount))
        icount = icount + 1
    improvementMtreeCurves['uplift'] = improvementMtreeCurves[modelNames].mean(axis=1)
    improvementMtreeCurves['grUplift'] = improvementMtreeCurves[groupModelNames].mean(axis=1)
    improvementModels = pd.DataFrame({})
    improvementModels = pd.concat([improvementModels, improvementMtreeCurves], ignore_index=True)

    ## convert to percent
    improvementModels['uplift'] = improvementModels['uplift'] * 100
    improvementModels['grUplift'] = improvementModels['grUplift'] * 100
    # if (plotFig):
    #     plotQini(improvementModels)
    curveNames = ['Tree']
    improvementModels['uplift'] = improvementModels['uplift'].fillna(0)
    estimateAres = areaUnderCurve(improvementModels, curveNames)
    return estimateAres[0]


def estimate_AUUC_scores(baseInFolder, inputName, testName, outcomeName, threshold):
    fold_nbr = 5
    methods = []
    auucScores = []


    method = 'LogrRe'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    # fullResultFolder = baseInFolder + '/'+ method +'/' + inputName
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    opiAuc = getAUUCTopGroup(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    opiAuc = abs(opiAuc)
    auucScores.append(opiAuc)
    print(method + ": " + str(opiAuc))


    method = 'LogrRa'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    # fullResultFolder = baseInFolder + '/'+ method +'/' + inputName
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    opiAuc = getAUUCTopGroup(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)

    opiAuc = abs(opiAuc)
    auucScores.append(opiAuc)
    print(method + ": " + str(opiAuc))

    method = 'SvmRe'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    # fullResultFolder = baseInFolder + '/'+ method +'/' + inputName
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    opiAuc = getAUUCTopGroup(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    opiAuc = abs(opiAuc)
    auucScores.append(opiAuc)
    print(method + ": " + str(opiAuc))

    method = 'SvmRa'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    # fullResultFolder = baseInFolder + '/'+ method +'/' + inputName
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    opiAuc = getAUUCTopGroup(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    opiAuc = abs(opiAuc)
    auucScores.append(opiAuc)
    print(method + ": " + str(opiAuc))


    method = 'RfRe'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    # fullResultFolder = baseInFolder + '/'+ method +'/' + inputName
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    opiAuc = getAUUCTopGroup(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    opiAuc = abs(opiAuc)
    auucScores.append(opiAuc)
    print(method + ": " + str(opiAuc))

    method = 'RfRa'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    # fullResultFolder = baseInFolder + '/'+ method +'/' + inputName
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    opiAuc = getAUUCTopGroup(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    opiAuc = abs(opiAuc)
    auucScores.append(opiAuc)
    print(method + ": " + str(opiAuc))


    method = 'CTR'
    methods.append(method)
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    # fullResultFolder = baseInFolder + '/'+ method +'/' + inputName
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    opiAuc = getAUUCTopGroup(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    opiAuc = abs(opiAuc)
    auucScores.append(opiAuc)
    print(method + ": " + str(opiAuc))

    AUUC_scores = pd.DataFrame({'AUUC': auucScores, 'Method': methods})

    return AUUC_scores


def plot_AUUC_bar(auuc_scores, data_test, threshold):
    """
    Plots a bar chart of AUUC scores for different methods with a custom color for each category using matplotlib.

    Parameters:
    - auuc_scores: DataFrame with 'Method' and 'AUUC' columns.
    - data_test: A string indicating the data test type (e.g., "Train" or "Test").
    - threshold: Threshold value to be shown in the title (if needed).

    Returns:
    - plt: The plot object.
    """

    # Customize the size of the graph
    plt.figure(figsize=(10, 6))

    # Define custom colors and order of methods for the plot
    colors = {
        # 'CF': '#1f77b4',  # Blue
        'LogrRe': '#27ae60',  # Olive
        'LogrRa': '#27ae60',  # Olive
        'SvmRe': '#c8e526',  # Lightsteelblue
        'SvmRa': '#c8e526',  # Lightsteelblue
        'RfRe': '#f4d03f',  # Yellow
        'RfRa': '#f4d03f',  # Yellow
        'CTR': '#ff7f0e'  # Orange
    }

    # Specify the order of methods to display
    # order = ['CF', 'LOGRe', 'LOGRa', 'SVMe', 'SVMa', 'RFe', 'RFa', 'CTR']
    order = ['LogrRe', 'LogrRa', 'SvmRe', 'SvmRa', 'RfRe', 'RfRa', 'CTR']

    # Filter and sort the DataFrame based on the order
    auuc_scores = auuc_scores.set_index('Method').reindex(order).reset_index()

    # Extract data for plotting
    methods = auuc_scores['Method']
    auuc_values = auuc_scores['AUUC']
    bar_colors = [colors.get(method, '#333333') for method in methods]

    # Plot bars
    x_positions = np.arange(len(methods))
    plt.bar(x_positions, auuc_values, color=bar_colors, width=0.7)

    # Add labels, title, and legend
    plt.xticks(x_positions, methods, rotation=0, ha='center', fontsize=16)
    plt.xlabel('Method', fontsize=18)
    plt.ylabel('AUUC', fontsize=18)
    # plt.title(f'{data_test} Dataset with Threshold = {threshold}', fontsize=16)
    plt.tight_layout()

    return plt
#

def calculate_1fold_recovery_rate(df):
    """
    Calculate the Recovery Rate based on the following conditions:
    - Set Recovery variable = 1 if both 'resp.pCR' = 1 and 'FOLLOW_REC' = 1.
    - Calculate Recovery Rate as the total count of Recovery = 1 divided by the total count of FOLLOW_REC = 1.

    Parameters:
    - df: DataFrame containing columns 'resp.pCR' and 'FOLLOW_REC'.

    Returns:
    - recovery dataframe: The set of calculated Recovery Rate and total follow recommendation for 1 data fold
    """
    recovery = []
    # Set Recovery = 1 where both "resp.pCR" = 1 and "FOLLOW_REC" = 1
    df['Recovery'] = ((df['resp.pCR'] == 1) & (df['FOLLOW_REC'] == 1)).astype(int)

    # Calculate the total count of Recovery = 1
    total_recovery = df['Recovery'].sum()

    # Calculate the total count of FOLLOW_REC = 1
    total_follow_rec = df['FOLLOW_REC'].sum()

    recovery = pd.DataFrame({'recovery': [total_recovery],
                             'follow_rec': [total_follow_rec]
                             })

    # Calculate Recovery Rate
    # if total_follow_rec > 0:
    #     recovery_rate = (total_recovery / total_follow_rec) * 100
    # else:
    #     recovery_rate = 0  # To handle division by zero if no FOLLOW_REC = 1

    # return recovery_rate
    return recovery


def calculate_average_recovery_rate(FolderLocation, fold, prefileName, postfileName, outcomeName, plotFig=True):
    """
        Calculate the average Recovery Rate from a list of DataFrames.

        Parameters:
        - dataframes: List of DataFrames containing columns 'resp.pCR' and 'FOLLOW_REC'.

        Returns:
        - average_recovery_rate: The average Recovery Rate across all DataFrames.
        """
    total_recovery_rate = 0
    count = 0
    total_recovery = 0
    total_follow_rec =0

    # improvementMtreeModels = []
    for fileCount in range(1, fold + 1):
        improveFilePath = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '.csv')
        if (not (path.exists(improveFilePath))):
            continue
        results = pd.read_csv(improveFilePath, encoding="ISO-8859-1", engine='python')


        # Calculate recovery rate for each DataFrame
        recovery_rate = calculate_1fold_recovery_rate(results)
        # total_recovery_rate += recovery_rate
        # count += 1
        # Use .iloc[0] to retrieve the scalar values from the DataFrame
        total_recovery += recovery_rate['recovery'].iloc[0]
        total_follow_rec += recovery_rate['follow_rec'].iloc[0]

    # Calculate the average recovery rate
    if total_follow_rec > 0:
        average_recovery_rate = (total_recovery / total_follow_rec) * 100
    else:
        average_recovery_rate = 0  # Handle case where total_follow_rec is zero

    return average_recovery_rate


def calculate_recovery_rate(baseInFolder, inputName, testName, outcomeName, threshold):
    fold_nbr = 5
    methods = []
    recovery_rates = []

    # method = 'CF'
    # methods.append(method)
    # # prefileName = inputName + '_' + testName + '_' +  method + '_'
    # prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    # postfileName = ''
    # # fullResultFolder = baseInFolder + '/'+ method +'/' + inputName
    # fullResultFolder = os.path.join(baseInFolder, method, inputName)
    #
    # recovery_rate = calculate_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    # recovery_rates.append(recovery_rate)
    # print(method + ": " + str(recovery_rate))

    #


    method = 'CurrentProtocol'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = calculate_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))

    #
    method = 'LogrRe'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = calculate_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))


    method = 'LogrRa'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = calculate_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))


    method = 'SvmRe'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = calculate_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))


    method = 'SvmRa'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = calculate_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))

    method = 'RfRe'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = calculate_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))

    method = 'RfRa'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = calculate_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))

    method = 'CTR'
    methods.append(method)
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = calculate_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))

    # method = 'CurrentProtocol'
    # methods.append(method)
    # recovery_rate = 100 * 57/222
    # recovery_rates.append(recovery_rate)
    # print(method + ": " + str(recovery_rate))


    RecoveryRates = pd.DataFrame({'RecoveryRate': recovery_rates, 'Method': methods})

    return RecoveryRates



def plot_RecoveryRate_bar(RecoveryRates, data_test, threshold):
    """
    Plots a bar chart of RecoveryRate for different Methods with a custom color for each category using matplotlib.

    Parameters:
    - RecoveryRates: DataFrame with 'Method' and 'RecoveryRate' columns.
    - data_test: A string indicating the data test type (e.g., "Train" or "Test").
    - threshold: Threshold value to be shown in the title (if needed).

    Returns:
    - plt: The plot object.
    """
    # Customize the size of the graph
    plt.figure(figsize=(10, 6))

    # Define custom colors and order of methods for the plot
    colors = {
        'CurrentProtocol': '#1f77b4',  # Blue
        # 'CF': '#1f77b4',  # Blue
        'LogrRe': '#27ae60',  # Olive
        'LogrRa': '#27ae60',  # Olive
        'SvmRe': '#c8e526',  # Lightsteelblue
        'SvmRa': '#c8e526',  # Lightsteelblue
        'RfRe': '#f4d03f',  # Yellow
        'RfRa': '#f4d03f',  # Yellow
        'CTR': '#ff7f0e'  # Orange
    }


    # Order of methods to display
    order = ['CurrentProtocol', 'LogrRe', 'LogrRa', 'SvmRe', 'SvmRa', 'RfRe', 'RfRa', 'CTR']

    # Filter and sort RecoveryRates DataFrame based on the defined order
    RecoveryRates = RecoveryRates.set_index('Method').reindex(order).reset_index()

    # Extract data for plotting
    methods = RecoveryRates['Method']
    recovery_rates = RecoveryRates['RecoveryRate']
    bar_colors = [colors.get(method, '#333333') for method in methods]

    # Plot bars
    x_positions = np.arange(len(methods))
    plt.bar(x_positions, recovery_rates, color=bar_colors, width=0.8)

    # Add labels, title, and legend
    plt.xticks(x_positions, methods, rotation=0, ha='center', fontsize=13)
    plt.xlabel('Method', fontsize=18)
    plt.ylabel('Recovery Rate (%)', fontsize=18)
    # plt.title(f'{data_test} Dataset with Threshold = {threshold}', fontsize=16)
    plt.tight_layout()

    return plt


def generate_AUUC_results(baseInFolder, datasetName, outcomeName, threshold):
    # Create PerformanceEval directories
    PerformanceEval_folder = os.path.join(baseInFolder, 'PerformanceEval')
    if not os.path.exists(PerformanceEval_folder):
        os.makedirs(PerformanceEval_folder)

    appResults = os.path.join(baseInFolder, PerformanceEval_folder)

    data_test = 'test'
    auuc_scores =  estimate_AUUC_scores(baseInFolder, datasetName, data_test, outcomeName, threshold)
    plt = plot_AUUC_bar(auuc_scores, data_test, threshold)

    AUUCfileName = datasetName + data_test + '_' + str(threshold) + '_AUUC.csv'
    AUUCfile = os.path.join(appResults, AUUCfileName)
    auuc_scores.to_csv(AUUCfile, index=False)

    # Save the plot to a file
    plt.savefig(appResults + '/'  + datasetName + data_test + '_' + str(threshold) + '_AUUC.png', dpi=400, facecolor = 'w')


def recovery_rate_results(baseInFolder, datasetName, outcomeName, threshold):
    # Create PerformanceEval directories
    PerformanceEval_folder = os.path.join(baseInFolder, 'PerformanceEval')
    if not os.path.exists(PerformanceEval_folder):
        os.makedirs(PerformanceEval_folder)

    appResults = os.path.join(baseInFolder, PerformanceEval_folder)
    data_test = 'test'
    RecoveryRates = calculate_recovery_rate(baseInFolder, datasetName, data_test, outcomeName, threshold)
    plt = plot_RecoveryRate_bar(RecoveryRates, data_test, threshold)

    RecoveryRatefileName = datasetName + data_test + '_' + str(threshold) + '_RecoveryRate.csv'
    RecoveryRatefile = os.path.join(appResults, RecoveryRatefileName)
    RecoveryRates.to_csv(RecoveryRatefile, index=False)

    # Save the plot to a file
    plt.savefig(appResults + '/'  + datasetName + data_test + '_' + str(threshold) + 'RecoveryRate.png', dpi=400, facecolor = 'w')



recovery_rate_results(outputPath, data_name, outcome_name, threshold)  #Section 3.5.1. Recovery Rate Evaluation
subtypes_recovery_comparison(outputPath, data_name, outcome_name, threshold)    #Section 3.5.2. Recovery Rate in each Breast Cancer Subtype
generate_AUUC_results(outputPath, data_name, outcome_name, threshold) #Section Evaluating the Performance of Recommendation Models
