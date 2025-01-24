import os
import pandas as pd
import seaborn as sns
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os.path
from os import path
import random
# import seaborn as sns
import matplotlib.pyplot as plt

# from generate_results import *


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

def classify_breast_cancer(ER_status, HER2_status, ER_Allred, Grade_pre_chemotherapy):
    """
    Classify breast cancer subtypes based on input features.

    Parameters:
        ER_status (str): 'Positive' or 'Negative'
        HER2_status (str): 'Positive' or 'Negative'
        ER_Allred (int): ER Allred score (0-8)
        Grade_pre_chemotherapy (int): Tumour grade (1, 2, or 3)

    Returns:
        str: Breast cancer subtype - 'Luminal A', 'Luminal B', 'HER2 Positive', or 'Triple Negative'
    """
    # Check for ER-positive cases
    if ER_status == 1:
        if HER2_status == -1:
            if ER_Allred >= 7 and Grade_pre_chemotherapy <= 2:
                return 'Luminal A'
            else:
                return 'Luminal B'
        elif HER2_status == 1:
            return 'Luminal B'

    # Check for ER-negative cases
    elif ER_status == -1:
        if HER2_status == 1:
            return 'HER2 Positive'
        elif HER2_status == -1:
            return 'Triple Negative'

    # Default case for invalid inputs
    return 'Unknown Subtype'


def subtypes_average_recovery_rate(FolderLocation, fold, prefileName, postfileName, outcomeName, plotFig=True):
    """
    Calculate the average Recovery Rate for each breast cancer subtype.

    Parameters:
        FolderLocation (str): Folder location of the CSV files.
        fold (int): Number of folds to process.
        prefileName (str): Prefix of the file name.
        postfileName (str): Suffix of the file name.
        outcomeName (str): Outcome name column (not used explicitly).
        plotFig (bool): Flag to plot figures (not implemented).

    Returns:
        List of tuples: Each tuple contains (Subtype, Average Recovery Rate).
    """
    subtypes_recovery = {
        'Luminal A': {'recovery': 0, 'follow_rec': 0},
        'Luminal B': {'recovery': 0, 'follow_rec': 0},
        'HER2 Positive': {'recovery': 0, 'follow_rec': 0},
        'Triple Negative': {'recovery': 0, 'follow_rec': 0},
        # 'Unknown Subtype': {'recovery': 0, 'follow_rec': 0}
    }

    # Iterate through each fold
    for fileCount in range(1, fold + 1):
        improveFilePath = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '.csv')
        if not path.exists(improveFilePath):
            continue
        df = pd.read_csv(improveFilePath, encoding="ISO-8859-1", engine='python')

        # Apply classification function row-wise
        df['BC_Subtype'] = df.apply(lambda row: classify_breast_cancer(
            ER_status=row['ER.status'],
            HER2_status=row['HER2.status'],
            ER_Allred=row['ER.Allred'],
            Grade_pre_chemotherapy=row['Grade.pre.chemotherapy']), axis=1)

        # # Save updated DataFrame for reference
        # output_file = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '_sub.csv')
        # df.to_csv(output_file, index=False)
        # print(f"Classification completed. Results saved to {output_file}")

        # Calculate recovery rate for each subtype
        for subtype in subtypes_recovery.keys():
            # Explicitly make a copy to avoid SettingWithCopyWarning
            subtype_df = df[df['BC_Subtype'] == subtype].copy()

            # Calculate recovery rate for the subtype
            recovery_rate = calculate_1fold_recovery_rate(subtype_df)
            subtypes_recovery[subtype]['recovery'] += recovery_rate['recovery'].iloc[0]
            subtypes_recovery[subtype]['follow_rec'] += recovery_rate['follow_rec'].iloc[0]

    # Calculate average recovery rate for each subtype
    average_recovery_rates = []
    for subtype, values in subtypes_recovery.items():
        if float(values['follow_rec']) > 0:
            avg_rate = (float(values['recovery']) / float(values['follow_rec'])) * 100
        else:
            avg_rate = 0.0
        #
        # if values['follow_rec'] > 0:
        #     avg_rate = (values['recovery'] / values['follow_rec']) * 100
        # else:
        #     avg_rate = 0  # Handle division by zero
        average_recovery_rates.append((subtype, avg_rate))

    return average_recovery_rates



def subtypes_average_recovery_rate_1(FolderLocation, fold, prefileName, postfileName, outcomeName, plotFig=True):
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
        df = pd.read_csv(improveFilePath, encoding="ISO-8859-1", engine='python')

        # Apply classification function row-wise
        df['BC_Subtype'] = df.apply(lambda row: classify_breast_cancer(
            ER_status=row['ER.status'],
            HER2_status=row['HER2.status'],
            ER_Allred=row['ER.Allred'],
            Grade_pre_chemotherapy=row['Grade.pre.chemotherapy']), axis=1)



        output_file = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '_sub.csv')
        # Write the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Classification completed. Results saved to {output_file}")

        # Separate data frames for each subtype
        luminal_a_df = df[df['BC_Subtype'] == 'Luminal A']
        luminal_b_df = df[df['BC_Subtype'] == 'Luminal B']
        her2_positive_df = df[df['BC_Subtype'] == 'HER2 Positive']
        triple_negative_df = df[df['BC_Subtype'] == 'Triple Negative']


        bc_subtype = luminal_a_df
        # Calculate recovery rate for each DataFrame
        recovery_rate = calculate_1fold_recovery_rate(bc_subtype)
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



def subtypes_clasification_saveFile(FolderLocation, fold, prefileName, postfileName, outcomeName, plotFig=True):
    """
           Read a CSV file with breast cancer features, classify subtypes, and write to a new CSV file.
    Additionally, save each subtype into separate data frames.
        CaLL from bc_subtypes_methods() function for each method .csv file
    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file with classifications.
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
        df = pd.read_csv(improveFilePath, encoding="ISO-8859-1", engine='python')

        # Apply classification function row-wise
        df['BC_Subtype'] = df.apply(lambda row: classify_breast_cancer(
            ER_status=row['ER.status'],
            HER2_status=row['HER2.status'],
            ER_Allred=row['ER.Allred'],
            Grade_pre_chemotherapy=row['Grade.pre.chemotherapy']), axis=1)



        output_file = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '_sub.csv')
        # Write the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Classification completed. Results saved to {output_file}")


def bc_subtypes_clasification(baseInFolder, inputName, testName, outcomeName, threshold):
    fold_nbr = 5
    methods = []
    recovery_rates = []

    method = 'CurrentProtocol'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)
    subtypes_clasification_saveFile(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False) #Generate a .csv file of Subtypes Classification, No return


    method = 'CTR'
    methods.append(method)
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)
    subtypes_clasification_saveFile(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False) #Generate a .csv file of Subtypes Classification, No return



def subtypes_recovery_rate1(baseInFolder, inputName, testName, outcomeName, threshold):
    fold_nbr = 5
    methods = []
    recovery_rates = []


    method = 'CurrentProtocol'
    methods.append(method)
    # prefileName = inputName + '_' + testName + '_' +  method + '_'
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = subtypes_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))



    method = 'CTR'
    methods.append(method)
    prefileName = inputName + '_' + testName + '_' + method + '_' + str(threshold) + '_'
    postfileName = ''
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    recovery_rate = subtypes_average_recovery_rate(fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False)
    recovery_rates.append(recovery_rate)
    print(method + ": " + str(recovery_rate))


    RecoveryRates = pd.DataFrame({'RecoveryRate': recovery_rates, 'Method': methods})

    return RecoveryRates

def subtypes_recovery_rate(baseInFolder, inputName, testName, outcomeName, threshold):
    """
    Calculate the recovery rate for each breast cancer subtype across two methods and multiple folds.

    Parameters:
        baseInFolder (str): Base input folder.
        inputName (str): Input file name prefix.
        testName (str): Test name.
        outcomeName (str): Outcome column name.
        threshold (float): Threshold parameter for file naming.

    Returns:
        List of tuples: Each tuple contains (Method, Subtype, Recovery Rate).
    """
    fold_nbr = 5
    methods = ['CurrentProtocol', 'RfRa', 'CTR']
    results = []


    for method in methods:
        prefileName = f"{inputName}_{testName}_{method}_{threshold}_"
        postfileName = ''
        fullResultFolder = os.path.join(baseInFolder, method, inputName)

        # Get average recovery rates for this method
        average_recovery_rates = subtypes_average_recovery_rate(
            fullResultFolder, fold_nbr, prefileName, postfileName, outcomeName, False
        )

        # Append results with method name
        for subtype, rate in average_recovery_rates:
            results.append((method, subtype, rate))

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results, columns=['Method', 'Subtype', 'Recovery Rate'])

    # # Save the DataFrame to a CSV file
    # output_file = os.path.join(baseInFolder, "subtypes_recovery_rates.csv")
    # results_df.to_csv(output_file, index=False)
    # print(f"Results saved to {output_file}")

    return results_df



def plot_recovery_rate_comparison(results_df, output_path=None):
    """
    Plot a comparison of recovery rates for each subtype between two methods in a single graph.

    Parameters:
        results_df (DataFrame): DataFrame containing Method, Subtype, and Recovery Rate.
        output_path (str): Path to save the graph image (optional).
    """
    subtypes = results_df['Subtype'].unique()
    methods = results_df['Method'].unique()

    # Custom colours for methods
    method_colours = {
        'CurrentProtocol': '#1f77b4',  # Blue
        'RfRa': '#f4d03f',
        'CTR': '#ff7f0e'  # Orange

    }

    # Plot setup
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = range(len(subtypes))

    for i, method in enumerate(methods):
        recovery_rates = [
            results_df[(results_df['Method'] == method) & (results_df['Subtype'] == subtype)]['Recovery Rate'].iloc[0]
            for subtype in subtypes
        ]
        plt.bar(
            [pos + i * bar_width for pos in index],
            recovery_rates,
            bar_width,
            label=method,
            color=method_colours.get(method, '#333333')  # Default to grey if method not found
        )

    plt.xlabel('Breast Cancer Subtypes', fontsize=18)
    plt.ylabel('Recovery Rate (%)', fontsize=18)
    # plt.title('Recovery Rate Comparison by Subtype and Method')
    plt.xticks([pos + bar_width / 2 for pos in index], subtypes, fontsize=16)
    plt.legend(title='Methods')
    plt.tight_layout()

    return plt


    # # Save the plot
    # if output_path:
    #     plt.savefig(output_path)
    #     print(f"Saved comparison plot to {output_path}")
    # plt.show()


def plot_recovery_rate_comparison1(results_df, output_path=None):
    """
    Plot a comparison of recovery rates for each subtype between two methods and save the graph.

    Parameters:
        results_df (DataFrame): DataFrame containing Method, Subtype, and Recovery Rate.
        output_path (str): Path to save the graph image (optional).
    """
    subtypes = results_df['Subtype'].unique()
    methods = results_df['Method'].unique()

    for subtype in subtypes:
        plt.figure(figsize=(6, 4))
        subset = results_df[results_df['Subtype'] == subtype]
        plt.bar(subset['Method'], subset['Recovery Rate'], color=['skyblue', 'lightgreen'])
        plt.xlabel('Method')
        plt.ylabel('Recovery Rate (%)')
        plt.title(f'Recovery Rate Comparison for {subtype}')
        plt.tight_layout()

        if output_path:
            save_path = os.path.join(output_path, f"{subtype}_recovery_rate_comparison.png")
            plt.savefig(save_path)
            print(f"Saved plot for {subtype} to {save_path}")
        plt.close()

def generate_subtypes_classification(baseInFolder, datasetName, outcomeName, threshold):
    # Create PerformanceEval directories
    PerformanceEval_folder = os.path.join(baseInFolder, 'PerformanceEval')
    if not os.path.exists(PerformanceEval_folder):
        os.makedirs(PerformanceEval_folder)

    appResults = os.path.join(baseInFolder, PerformanceEval_folder)
    data_test = 'test'
    bc_subtypes_clasification(baseInFolder, datasetName, data_test, outcomeName, threshold)  #Generate .csv file to classify BC Subtypes

    data_test = 'train'
    bc_subtypes_clasification(baseInFolder, datasetName, data_test, outcomeName, threshold)     #Generate .csv file to classify BC Subtypes


def subtypes_recovery_comparison(baseInFolder, datasetName, outcomeName, threshold):
    # Create PerformanceEval directories
    PerformanceEval_folder = os.path.join(baseInFolder, 'PerformanceEval')
    if not os.path.exists(PerformanceEval_folder):
        os.makedirs(PerformanceEval_folder)

    appResults = os.path.join(baseInFolder, PerformanceEval_folder)
    data_test = 'test'

    # bc_subtypes_clasification(baseInFolder, datasetName, data_test, outcomeName, threshold)  #Generate .csv file to classify BC Subtypes
    RecoveryRates = subtypes_recovery_rate(baseInFolder, datasetName, data_test, outcomeName, threshold)

    # plt = plot_RecoveryRate_bar(RecoveryRates, data_test, threshold)

    RecoveryRatefileName = datasetName + data_test + '_' + str(threshold) + '_RecoveryRate-sub.csv'
    RecoveryRatefile = os.path.join(appResults, RecoveryRatefileName)
    RecoveryRates.to_csv(RecoveryRatefile, index=False)

    plt = plot_recovery_rate_comparison(RecoveryRates,appResults)
    # Save the plot to a file
    # plt.savefig(output_path)
    plt.savefig(appResults + '/'  + datasetName + data_test + '_' + str(threshold) + 'RecoveryRate_sub.png', dpi=400, facecolor = 'w')


    # data_test = 'train'
    # # bc_subtypes_clasification(baseInFolder, datasetName, data_test, outcomeName, threshold)     #Generate .csv file to classify BC Subtypes
    # RecoveryRates = subtypes_recovery_rate(baseInFolder, datasetName, data_test, outcomeName, threshold)
    #
    # # plt = plot_RecoveryRate_bar(RecoveryRates, data_test, threshold)
    # #
    # RecoveryRatefileName = datasetName + data_test + '_' + str(threshold) + '_RecoveryRate-subtype1.csv'
    # RecoveryRatefile = os.path.join(appResults, RecoveryRatefileName)
    # RecoveryRates.to_csv(RecoveryRatefile, index=False)
    # #
    # plt =   plot_recovery_rate_comparison(RecoveryRates, appResults)
    #
    # # # Save the plot to a file
    # plt.savefig(appResults + '/'  + datasetName + data_test + '_' + str(threshold) + 'RecoveryRate_sub1.png', dpi=400, facecolor = 'w')
