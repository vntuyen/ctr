import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold




def train_model_allTP(method, X_train, y_train, random_seed):
    # if method == 'RfRa':
    #     RFa_model = RandomForestClassifier(n_estimators=1000, random_state=random_seed)
    #     trained_model = RFa_model.fit(X_train, y_train)
    #
    #
    # if method == 'SvmRa':
    #      SVMa_model= SVC(probability=True, random_state=random_seed)
    #         # Train the model
    #      trained_model = SVMa_model.fit(X_train, y_train)
    #
    #
    # if method == 'LogrRa':
    #     # Initialize the Logistic Regression model model = LogisticRegression(max_iter=500, solver='liblinear')  # Increased max_iter and changed solver
    #     LOGRa_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=random_seed)  # Increased max_iter and changed solver
    #     # Train the model
    #     trained_model = LOGRa_model.fit(X_train, y_train)

    pipeline = get_model_pipeline(method, random_seed)
    trained_model = pipeline.fit(X_train, y_train)

    # Save the model
    # # Create output directories
    # output_dir = os.path.join(base_folder, 'output')
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # method_folder = os.path.join(output_dir, method)
    # if not os.path.exists(method_folder):
    #     os.makedirs(method_folder)
    # trained_model_file = os.path.join(method_folder, 'LOGRa_model.joblib')
    # dump(model, trained_model_file)


    return trained_model



def predict_prob_allTP(trained_model, data,  treatment_plans):
    X_test = data.copy()
    # X_test = data.drop(columns=[outcome_name], axis=1)

    # data_test = data.copy()  # Use a copy to avoid modifying X_test directly
    for factor in treatment_plans:
        for index, row in data.iterrows():
            if row[factor] == 1:
                data.at[index, 'CURRENT_TP'] = factor
        X_test[factor] = 1
        for col in treatment_plans:
            if col != factor:
                X_test[col] = 0


        # Make predictions
        # # Probability estimates
        probabilities = trained_model.predict_proba(X_test)
        # # For binary classification, probabilities for the positive class (class 1)
        positive_class_probabilities = probabilities[:, 1]
        # Assign each Treatment Plan to the prediction outcome values
        data[factor] = positive_class_probabilities

    return data


def estimateUplift(estimatedImprovements, outcomeName, sortbyabs=False):
    if (sortbyabs):
        estimatedImprovements['ABS_Improvement'] = estimatedImprovements['LIFT_SCORE'].abs()
        estimatedImprovements.sort_values(by=['ABS_Improvement'], ascending=[False], inplace=True, axis=0)
    else:
        estimatedImprovements.sort_values(by=['LIFT_SCORE'], ascending=[False], inplace=True, axis=0)
    estimatedImprovements = estimatedImprovements.reset_index(drop=True)

    Sum_Y_Follow_Rec = np.array([])
    Sum_Nbr_Follow_Rec = np.array([])
    Sum_Y_Not_Follow_Rec = np.array([])
    Sum_Nbr_Not_Follow_Rec = np.array([])
    Improvement = np.array([])
    total_Y_Follow_Rec = 0
    total_Nbr_Follow_Rec = 0
    total_Y_Not_Follow_Rec = 0
    total_Nbr_Not_Follow_Rec = 0
    for index, individual in estimatedImprovements.iterrows():
        improvementTemp = 0
        if (individual['FOLLOW_REC'] == 1):
            total_Nbr_Follow_Rec = total_Nbr_Follow_Rec + 1
            total_Y_Follow_Rec = total_Y_Follow_Rec + individual[outcomeName]
        else:
            total_Nbr_Not_Follow_Rec = total_Nbr_Not_Follow_Rec + 1
            total_Y_Not_Follow_Rec = total_Y_Not_Follow_Rec + individual[outcomeName]
        Sum_Nbr_Follow_Rec = np.append(Sum_Nbr_Follow_Rec, total_Nbr_Follow_Rec)
        Sum_Y_Follow_Rec = np.append(Sum_Y_Follow_Rec, total_Y_Follow_Rec)
        Sum_Nbr_Not_Follow_Rec = np.append(Sum_Nbr_Not_Follow_Rec, total_Nbr_Not_Follow_Rec)
        Sum_Y_Not_Follow_Rec = np.append(Sum_Y_Not_Follow_Rec, total_Y_Not_Follow_Rec)
        if (total_Nbr_Follow_Rec == 0 or total_Nbr_Not_Follow_Rec == 0):
            if (total_Nbr_Follow_Rec > 0):
                improvementTemp = (total_Y_Follow_Rec / total_Nbr_Follow_Rec)
            else:
                improvementTemp = 0
        else:
            improvementTemp = (total_Y_Follow_Rec / total_Nbr_Follow_Rec) - (
                        total_Y_Not_Follow_Rec / total_Nbr_Not_Follow_Rec)
        Improvement = np.append(Improvement, improvementTemp)
    ser = pd.Series(Sum_Nbr_Follow_Rec)
    estimatedImprovements['N_TREATED'] = ser
    ser = pd.Series(Sum_Y_Follow_Rec)
    estimatedImprovements['Y_TREATED'] = ser
    ser = pd.Series(Sum_Nbr_Not_Follow_Rec)
    estimatedImprovements['N_UNTREATED'] = ser
    ser = pd.Series(Sum_Y_Not_Follow_Rec)
    estimatedImprovements['Y_UNTREATED'] = ser
    ser = pd.Series(Improvement)
    estimatedImprovements['UPLIFT'] = ser

    return estimatedImprovements


def make_recommendation(method, data_name, trained_model, X_test, y_test, base_folder, treatment_plans, threshold):
    data = X_test.copy()  # Use a copy to avoid modifying X_test directly
    data_test = X_test.copy()  # Use a copy to avoid modifying X_test directly
    data['HIGHEST_PROBABILITY'] = 0.0
    data['OPTIMAL_TP'] = ''
    data['CURRENT_TP'] = ''
    data['FOLLOW_REC'] = 0

    for factor in treatment_plans:
        for index, row in data.iterrows():
            if row[factor] == 1:
                data.at[index, 'CURRENT_TP'] = factor
        data_test[factor] = 1
        for col in treatment_plans:
            if col != factor:
                data_test[col] = 0

        # Make predictions
        # # Probability estimates
        probabilities = trained_model.predict_proba(data_test)
        # # For binary classification, probabilities for the positive class (class 1)
        positive_class_probabilities = probabilities[:, 1]
        # Assign each Treatment Plan to the prediction outcome values
        data[factor] = positive_class_probabilities

    for index2, row2 in data.iterrows():
        prev_y = -9999
        optimal_treatments_list = []
        for factor2 in treatment_plans:
            if prev_y < row2[factor2]:
                prev_y = row2[factor2]
                highest_treatment_name = factor2
            if row2[factor2] > threshold:
                optimal_treatment_name = factor2
                optimal_treatments_list.append(optimal_treatment_name)

        # val = list(val)
        # # data.at[index, 'PREDICT_OUTCOME'] = float(val[0])
        # # Extract single elements explicitly before assignment
        # predict_outcome_value = float(val[0].item())  # Extract single element
        data.at[index2, 'HIGHEST_PROBABILITY'] = prev_y
        data.at[index2, 'OPTIMAL_TP'] = optimal_treatments_list
        if (data.at[index2, 'CURRENT_TP'] in optimal_treatments_list):
            data.at[index2, 'FOLLOW_REC'] = 1

    data_out = data.copy()
    # data_out['RealOUTCOME'] = data[outcome_name]
    data_out['RealOUTCOME'] = y_test

    # Create output directories
    output_dir = os.path.join(base_folder, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    method_folder = os.path.join(output_dir, method)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    output_data_name = os.path.join(method_folder, data_name)
    if not os.path.exists(output_data_name):
        os.makedirs(output_data_name)

    sum_REC = data_out['FOLLOW_REC'].sum()
    new_file_name = f'{data_name}_{method}_{sum_REC}_REC.csv'
    # new_file_name = f'{data_name}_{method}_TPs_REC.csv'
    full_path = os.path.join(output_data_name, new_file_name)
    data_out.to_csv(full_path, index=False)


def cross_validation_uplift_score_fold_allTP(method, data_name, trained_model, testFile, outcome_name, base_folder, treatment_plans, remove_attributes, threshold):
    highest_treatment_name = ''
    test_data = pd.read_csv(testFile)

    # Flip 0 → 1 and 1 → 0
    if data_name == "METABRIC":
        test_data[outcome_name] = 1 - test_data[outcome_name]

    test = test_data.copy()
    test = test.drop(columns=remove_attributes, axis=1)

    data = test.drop(columns=[outcome_name], axis=1)
    # y_test = test_Data[outcome_name]


    ## Predict the Probability of receving Positive outcome (1) for testing data.
    data = predict_prob_allTP(trained_model, data,  treatment_plans)


    data['OPTIMAL_TP'] = ''
    data['LIFT_SCORE'] = 0.0
    # data['RealOUTCOME'] = ''
    data['FOLLOW_REC'] = 0
    data['Y_TREATED'] = 0
    data['N_TREATED'] = 0
    data['Y_UNTREATED'] = 0
    data['N_UNTREATED'] = 0
    data['UPLIFT'] = 0


    for index2, row2 in data.iterrows():
        prev_y = -9999
        highest_treatment_name = ''
        # optimal_treatments_list = []
        for factor2 in treatment_plans:
            if prev_y < row2[factor2]:
                prev_y = row2[factor2]
                # if prev_y > threshold:
                highest_treatment_name = factor2

        data.at[index2, 'LIFT_SCORE'] = prev_y


        data.at[index2, 'OPTIMAL_TP'] = highest_treatment_name
        if (data.at[index2,'CURRENT_TP'] == highest_treatment_name):
            data.at[index2, 'FOLLOW_REC'] = 1

    data[outcome_name] =  test_data[outcome_name]

    data_out = estimateUplift(data, outcome_name)

    # Create output directories
    output_dir = os.path.join(base_folder, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    method_folder = os.path.join(output_dir, method)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    output_data_name = os.path.join(method_folder, data_name)
    if not os.path.exists(output_data_name):
        os.makedirs(output_data_name)


    fileName = os.path.basename(testFile)

    # Split the base name by '.'
    fileNameParts = fileName.split('.')

    # Further split the first part by '_'
    secondFileNameParts = fileNameParts[0].split('_')

    # Create the new file name using the specified method and parts
    new_file_name = f"{secondFileNameParts[0]}_{secondFileNameParts[1]}_{method}_{threshold}_{secondFileNameParts[2]}.{fileNameParts[1]}"

    full_path = os.path.join(output_data_name, new_file_name)
    # data_out.to_csv(full_path, index=False)

    # ✅ Create the new CSV file with test_data + FOLLOW_REC
    new_file_name_follow = f"{secondFileNameParts[0]}_{secondFileNameParts[1]}_{method}_{threshold}_{secondFileNameParts[2]}_follow.{fileNameParts[1]}"
    full_path_follow = os.path.join(output_data_name, new_file_name_follow)

    # Merge FOLLOW_REC into original test_data
    test_data_with_rec = test_data.copy()
    test_data_with_rec["FOLLOW_REC"] = data["FOLLOW_REC"].values

    # Save new file
    # test_data_with_rec.to_csv(full_path_follow, index=False)

def get_model_pipeline(method: str, random_seed: int):
    imputer = SimpleImputer(strategy="median")
    no_constants = VarianceThreshold(0.0)
    scaler = StandardScaler()

    if method in ["LogrRa", "LogrRe"]:
        # Deterministic + stable: single thread, scaled, more iters
        model = LogisticRegression(
            solver="saga", penalty="l2", C=1.0,
            max_iter=5000, tol=1e-3, n_jobs=1,
            random_state=random_seed
        )
        steps = [("imputer", imputer), ("varthresh", no_constants), ("scaler", scaler), ("clf", model)]

    elif method in ["SvmRa", "SvmRe"]:
        # SVC: keep single-threaded; give random_state to stabilise prob. estimates
        model = SVC(
            kernel="rbf", probability=True, C=1.0, gamma="scale",
            class_weight=None, random_state=random_seed
        )
        steps = [("imputer", imputer), ("varthresh", no_constants), ("scaler", scaler), ("clf", model)]

    elif method in ["RfRa", "RfRe"]:
        # RF: set n_jobs=1 for full determinism; keep bootstrap default True
        model = RandomForestClassifier(
            n_estimators=300, random_state=random_seed, n_jobs=1,
            class_weight="balanced_subsample"
        )
        steps = [("imputer", imputer), ("varthresh", no_constants), ("clf", model)]

    else:
        raise ValueError(f"Unknown method: {method}")

    return Pipeline(steps)

# def get_model_pipeline(method: str, random_seed: int):
#     """
#     Returns a sklearn Pipeline with imputer + (feature filter) + (scaler if needed) + classifier.
#     Supported: LogrRa, LogrRe, SvmRa, SvmRe, RfRa, RfRe
#     """
#     method = str(method)
#
#     # Shared steps
#     imputer = SimpleImputer(strategy="median")  # median is robust to outliers
#     # Remove constant features (helps LR/SVM convergence a lot)
#     no_constants = VarianceThreshold(threshold=0.0)
#     scaler = StandardScaler()
#
#     if method in ["LogrRa", "LogrRe"]:
#         # More iterations, looser tol, robust solver; scale features
#         model = LogisticRegression(
#             solver="saga",            # robust for large/high-dim; supports l1/l2/elasticnet
#             penalty="l2",
#             C=1.0,
#             max_iter=5000,
#             tol=1e-3,                 # slightly looser tolerance speeds convergence
#             n_jobs=-1,
#             class_weight=None,        # or "balanced" if your classes are skewed
#             random_state=random_seed
#         )
#         pipeline = Pipeline([
#             ("imputer", imputer),
#             ("varthresh", no_constants),
#             ("scaler", scaler),
#             ("clf", model),
#         ])
#
#     elif method in ["SvmRa", "SvmRe"]:
#         # SVM *must* be scaled
#         model = SVC(
#             kernel="rbf",
#             probability=True,
#             C=1.0,
#             gamma="scale",
#             class_weight=None,        # or "balanced" if needed
#             random_state=random_seed
#         )
#         pipeline = Pipeline([
#             ("imputer", imputer),
#             ("varthresh", no_constants),
#             ("scaler", scaler),
#             ("clf", model),
#         ])
#
#     elif method in ["RfRa", "RfRe"]:
#         # RF does not need scaling; keep as-is
#         model = RandomForestClassifier(
#             n_estimators=300,
#             max_depth=None,
#             min_samples_split=2,
#             min_samples_leaf=1,
#             random_state=random_seed,
#             n_jobs=-1,
#             class_weight="balanced_subsample"  # keeps trees stable if imbalance exists
#         )
#         pipeline = Pipeline([
#             ("imputer", imputer),
#             ("varthresh", no_constants),
#             ("clf", model),
#         ])
#
#     else:
#         raise ValueError(f"Unknown method: {method}")
#
#     return pipeline

# def get_model_pipeline(method: str, random_seed: int):
#     """
#     Returns a sklearn Pipeline with imputer + classifier for a given method.
#     Supported methods: LogrRa, LogrRe, SvmRa, SvmRe, RfRa, RfRe
#     """
#     if method in ["LogrRa", "LogrRe"]:
#         model = LogisticRegression(random_state=random_seed, max_iter=1000)
#     elif method in ["SvmRa", "SvmRe"]:
#         model = SVC(probability=True, random_state=random_seed)
#     elif method in ["RfRa", "RfRe"]:
#         model = RandomForestClassifier(
#             n_estimators=200,
#             random_state=random_seed,
#             class_weight="balanced"
#         )
#     else:
#         raise ValueError(f"Unknown method: {method}")
#
#     pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="mean")),
#         ("clf", model)
#     ])
#     return pipeline

def model_validation_fold_allTP(data_name, treatment_plans, outcome_name, remove_attributes, threshold, base_folder, random_seed):
    input_data_folder = os.path.join(base_folder, 'input', data_name)
    #Using 5 folds Cross validation for dataset 222 samples
    for timecn in range(1, 6):
        trainingFile = f"{input_data_folder}/{data_name}_train_{timecn}.csv"
        train_Data = pd.read_csv(trainingFile)

        # Flip 0 → 1 and 1 → 0
        if data_name == "METABRIC":
            train_Data[outcome_name] = 1 - train_Data[outcome_name]

        X_train = train_Data.drop(columns=remove_attributes + [outcome_name], axis=1)
        y_train = train_Data[outcome_name]



        #Training Models:
        method = 'LogrRa'
        # pipeline = get_model_pipeline(method, random_seed)
        # LOGRa_model = pipeline.fit(X_train, y_train)

        LOGRa_model = train_model_allTP(method, X_train, y_train, random_seed)
        method = 'RfRa'
        # pipeline = get_model_pipeline(method, random_seed)
        # RFa_model = pipeline.fit(X_train, y_train)
        RFa_model = train_model_allTP(method, X_train, y_train, random_seed)
        method = 'SvmRa'
        # pipeline = get_model_pipeline(method, random_seed)
        # SVMa_model = pipeline.fit(X_train, y_train)
        SVMa_model = train_model_allTP(method, X_train, y_train, random_seed)


        #Testing phase
        testFile = f"{input_data_folder}/{data_name}_test_{timecn}.csv"

        method = 'LogrRa'
        trained_model = LOGRa_model
        cross_validation_uplift_score_fold_allTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
                                      treatment_plans, remove_attributes, threshold)

        method = 'RfRa'
        trained_model = RFa_model
        cross_validation_uplift_score_fold_allTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
                                      treatment_plans, remove_attributes, threshold)
        method = 'SvmRa'
        trained_model = SVMa_model
        cross_validation_uplift_score_fold_allTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
                                      treatment_plans, remove_attributes, threshold)


        # testFile = f"{input_data_folder}/{data_name}_train_{timecn}.csv"
        # method = 'LogrRa'
        # trained_model = LOGRa_model
        # cross_validation_uplift_score_fold_allTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
        #                               treatment_plans, remove_attributes, threshold)
        #
        # method = 'RfRa'
        # trained_model = RFa_model
        # cross_validation_uplift_score_fold_allTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
        #                               treatment_plans, remove_attributes, threshold)
        # method = 'SvmRa'
        # trained_model = SVMa_model
        # cross_validation_uplift_score_fold_allTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
        #                               treatment_plans, remove_attributes, threshold)


def model_validation_fold_eachTP(data_name, treatment_plans, outcome_name, remove_attributes, threshold, base_folder, random_seed):
    input_data_folder = os.path.join(base_folder, 'input', data_name)
    #Using 5 folds Cross validation for dataset 222 samples
    for timecn in range(1, 6):
        trainingFile = f"{input_data_folder}/{data_name}_train_{timecn}.csv"
        train_Data = pd.read_csv(trainingFile)


        X_train = train_Data.drop(columns=remove_attributes + [outcome_name], axis=1)
        y_train = train_Data[outcome_name]


        # Training Models:

        method = 'LogrRe'
        LOGRe_model = train_model_eachTP(method, X_train, y_train, treatment_plans, random_seed)
        method = 'RfRe'
        RFe_model = train_model_eachTP(method, X_train, y_train, treatment_plans, random_seed)
        method = 'SvmRe'
        SVMe_model = train_model_eachTP(method, X_train, y_train, treatment_plans, random_seed)

        # Testing phase
        testFile = f"{input_data_folder}/{data_name}_test_{timecn}.csv"

        method = 'LogrRe'
        trained_model = LOGRe_model
        validation_uplift_score_fold_eachTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
                                           treatment_plans, remove_attributes, threshold)

        method = 'RfRe'
        trained_model = RFe_model
        validation_uplift_score_fold_eachTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
                                           treatment_plans, remove_attributes, threshold)
        method = 'SvmRe'
        trained_model = SVMe_model
        validation_uplift_score_fold_eachTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
                                           treatment_plans, remove_attributes, threshold)


        # testFile = f"{input_data_folder}/{data_name}_train_{timecn}.csv"
        # method = 'LogrRe'
        # trained_model = LOGRe_model
        # validation_uplift_score_fold_eachTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
        #                                    treatment_plans, remove_attributes, threshold)
        #
        # method = 'RfRe'
        # trained_model = RFe_model
        # validation_uplift_score_fold_eachTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
        #                                    treatment_plans, remove_attributes, threshold)
        # method = 'SvmRe'
        # trained_model = SVMe_model
        # validation_uplift_score_fold_eachTP(method, data_name, trained_model, testFile, outcome_name, base_folder,
        #                                    treatment_plans, remove_attributes, threshold)
#
#

#
def train_model_eachTP(method, trainingData, y_train, treatment_plans, random_seed):
    """
    Trains a specified model for each treatment plan.

    Parameters:
    - method: The type of model to train ('RFe', 'SVMe', or 'LOGRe').
    - trainingData: DataFrame containing training data with features and treatments.
    - y_train: Target variable.
    - treatment_plans: List of treatment plan columns.
    - random_seed: Random seed for reproducibility.

    Returns:
    - results: List of dictionaries, each containing a trained model and the corresponding factor.
    """
    results = []
    for factor in treatment_plans:
        # Define covariates excluding other treatment plans and adding the current factor
        covariates = list((set(trainingData.columns) - set(treatment_plans)).union({factor}))
        X_train = trainingData[covariates].to_numpy()

        # Train based on specified method
        if method == 'RfRe':
            pipeline = get_model_pipeline(method, random_seed)
            RFetrained = pipeline.fit(X_train, y_train)

            # RFe_model = RandomForestClassifier(n_estimators=1000, random_state=random_seed)
            # RFetrained = RFe_model.fit(X_train, y_train)
            trained_models = {'model': RFetrained, 'factor': factor}

        elif method == 'SvmRe':
            pipeline = get_model_pipeline(method, random_seed)
            SVMetrained = pipeline.fit(X_train, y_train)

            # SVMe_model = SVC(probability=True, random_state=random_seed)
            # SVMetrained = SVMe_model.fit(X_train, y_train)
            trained_models = {'model': SVMetrained, 'factor': factor}

        elif method == 'LogrRe':
            # LOGRe_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=random_seed)
            pipeline = get_model_pipeline(method, random_seed)
            LOGRetrained = pipeline.fit(X_train, y_train)

            # LOGRe_model = LogisticRegression(max_iter=100, solver='liblinear', random_state=random_seed)
            # LOGRetrained = LOGRe_model.fit(X_train, y_train)

            trained_models = {'model': LOGRetrained, 'factor': factor}

        else:
            raise ValueError("Unsupported method. Choose from 'RFe', 'SVMe', or 'LOGRe'.")

        # Append each trained model and factor to results
        results.append(trained_models)

    return results



def validation_uplift_score_fold_eachTP(method, data_name, trained_model, testFile, outcome_name, base_folder, treatment_plans, remove_attributes, threshold):
    test_data = pd.read_csv(testFile)
    test = test_data.copy()

    test = test.drop(columns=remove_attributes, axis=1)

    data = test.drop(columns=[outcome_name], axis=1)


    ## Predict the Probability of receving Positive outcome (1) for testing data.
    data = predict_prob_eachTP(trained_model, data,  treatment_plans)

    # data['CURRENT_TP'] = ''
    data['OPTIMAL_TP'] = ''
    data['LIFT_SCORE'] = 0.0
    # data['RealOUTCOME'] = ''
    data['FOLLOW_REC'] = 0
    data['Y_TREATED'] = 0
    data['N_TREATED'] = 0
    data['Y_UNTREATED'] = 0
    data['N_UNTREATED'] = 0
    data['UPLIFT'] = 0


    for index2, row2 in data.iterrows():
        prev_y = -9999
        highest_treatment_name = ''
        # optimal_treatments_list = []
        for factor2 in treatment_plans:
            if prev_y < row2[factor2]:
                prev_y = row2[factor2]
                # if prev_y > threshold:
                highest_treatment_name = factor2

        data.at[index2, 'LIFT_SCORE'] = prev_y


        data.at[index2, 'OPTIMAL_TP'] = highest_treatment_name
        if (data.at[index2,'CURRENT_TP'] == highest_treatment_name):
            data.at[index2, 'FOLLOW_REC'] = 1

    data[outcome_name] =  test_data[outcome_name]

    data_out = estimateUplift(data, outcome_name)


    #
    # Create output directories
    output_dir = os.path.join(base_folder, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    method_folder = os.path.join(output_dir, method)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    output_data_name = os.path.join(method_folder, data_name)
    if not os.path.exists(output_data_name):
        os.makedirs(output_data_name)

    # sum_REC = data_out['FOLLOW_REC'].sum()
    # new_file_name = f'{data_name}_test_{method}_thres{threshold}_{sum_REC}REC_validation4.csv'
    # new_file_name = f'{data_name}_{method}_TPs_REC.csv'

    fileName = os.path.basename(testFile)

    # Split the base name by '.'
    fileNameParts = fileName.split('.')

    # Further split the first part by '_'
    secondFileNameParts = fileNameParts[0].split('_')

    # Create the new file name using the specified method and parts
    new_file_name = f"{secondFileNameParts[0]}_{secondFileNameParts[1]}_{method}_{threshold}_{secondFileNameParts[2]}.{fileNameParts[1]}"

    full_path = os.path.join(output_data_name, new_file_name)
    # data_out.to_csv(full_path, index=False)

    # ✅ Create the new CSV file with test_data + FOLLOW_REC
    new_file_name_follow = f"{secondFileNameParts[0]}_{secondFileNameParts[1]}_{method}_{threshold}_{secondFileNameParts[2]}_follow.{fileNameParts[1]}"
    full_path_follow = os.path.join(output_data_name, new_file_name_follow)

    # Merge FOLLOW_REC into original test_data
    test_data_with_rec = test_data.copy()
    test_data_with_rec["FOLLOW_REC"] = data["FOLLOW_REC"].values

    # Save new file
    # test_data_with_rec.to_csv(full_path_follow, index=False)


def predict_prob_eachTP(trained_models, test_data, treatment_plans):
    """
    Predicts the probability of the positive class for each treatment plan using the provided models.

    Parameters:
    - trained_models: A list of trained models with 'model' and 'factor' keys.
    - test_data: A DataFrame representing the input data.
    - treatment_plans: List of treatment plan columns.

    Returns:
    - data: DataFrame with predicted probabilities for each treatment plan.
    """
    data = test_data.copy()

    # Initialize 'CURRENT_TP' column if it doesn't exist
    if 'CURRENT_TP' not in data.columns:
        data['CURRENT_TP'] = ''

    for model_info in trained_models:
        model = model_info['model']
        factor = model_info['factor']

        # Set 'CURRENT_TP' for each row based on the factor
        for index, row in data.iterrows():
            if row[factor] == 1:
                data.at[index, 'CURRENT_TP'] = factor

        # Define covariates excluding other treatment plans and including the current factor
        covariates = list((set(test_data.columns) - set(treatment_plans)).union({factor}))
        X_test = test_data[covariates].to_numpy()  # Use the full dataset without reshaping

        # Predict probabilities for the positive class
        probabilities = model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class

        # Assign the predicted probabilities for each treatment plan
        data[factor] = probabilities

    return data

def current_protocol(base_folder, method, fold, test_size, data_name, threshold):
    # Create output directories
    output_dir = os.path.join(base_folder, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    method_folder = os.path.join(output_dir, method)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    output_data_name = os.path.join(method_folder, data_name)
    if not os.path.exists(output_data_name):
        os.makedirs(output_data_name)

    inputPath = os.path.join(base_folder, 'input', data_name)
    input_data = f"{inputPath}/{data_name}.csv"

    dataset = pd.read_csv(input_data, encoding="ISO-8859-1", engine='python')
    # Mdataset = dataset.copy()
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


def current_protocol_validation_uplift_score(method, data_name, testFile, outcome_name, base_folder, treatment_plans, remove_attributes, threshold):
    test_data = pd.read_csv(testFile)
    test = test_data.copy()
    test = test.drop(columns=remove_attributes, axis=1)


    data = test.copy()

    data['CURRENT_TP'] = ''
    data['OPTIMAL_TP'] = ''
    data['LIFT_SCORE'] = 0.0
    data['FOLLOW_REC'] = 0
    data['Y_TREATED'] = 0
    data['N_TREATED'] = 0
    data['Y_UNTREATED'] = 0
    data['N_UNTREATED'] = 0
    data['UPLIFT'] = 0
    data['RealOUTCOME'] = 0

    for index, row in data.iterrows():
        for factor in treatment_plans:
            if row[factor] == 1:
                data.at[index, 'CURRENT_TP'] = factor
            # Assign each Treatment Plan to the current protocol values
            # data[factor] = test_data[factor]
        data.at[index, 'OPTIMAL_TP'] = data.at[index, 'CURRENT_TP']
        data.at[index, 'LIFT_SCORE'] = 1
        data.at[index, 'FOLLOW_REC'] = 1

    data['RealOUTCOME'] = test_data[outcome_name]
    data_out = estimateUplift(data, outcome_name)
    # data_out['PAM50'] = test_data['PAM50']

    # Create output directories
    output_dir = os.path.join(base_folder, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    method_folder = os.path.join(output_dir, method)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    output_data_name = os.path.join(method_folder, data_name)
    if not os.path.exists(output_data_name):
        os.makedirs(output_data_name)

    fileName = os.path.basename(testFile)

    # Split the base name by '.'
    fileNameParts = fileName.split('.')

    # Further split the first part by '_'
    secondFileNameParts = fileNameParts[0].split('_')

    # Create the new file name using the specified method and parts
    new_file_name = f"{secondFileNameParts[0]}_{secondFileNameParts[1]}_{method}_{threshold}_{secondFileNameParts[2]}.{fileNameParts[1]}"

    full_path = os.path.join(output_data_name, new_file_name)
    data_out.to_csv(full_path, index=False)

    # ✅ Create the new CSV file with test_data + FOLLOW_REC
    new_file_name_follow = f"{secondFileNameParts[0]}_{secondFileNameParts[1]}_{method}_{threshold}_{secondFileNameParts[2]}_follow.{fileNameParts[1]}"
    full_path_follow = os.path.join(output_data_name, new_file_name_follow)

    # Merge FOLLOW_REC into original test_data
    test_data_with_rec = test_data.copy()
    test_data_with_rec["FOLLOW_REC"] = data["FOLLOW_REC"].values

    # Save new file
    test_data_with_rec.to_csv(full_path_follow, index=False)


def current_protocol_validation_fold(data_name, treatment_plans, outcome_name, remove_attributes, threshold, base_folder,
                                     random_seed):
    input_data_folder = os.path.join(base_folder, 'input', data_name)
    # Using 5 folds Cross validation for dataset 222 samples
    for timecn in range(1, 6):
        # Testing phase
        testFile = f"{input_data_folder}/{data_name}_test_{timecn}.csv"

        method = 'CurrentProtocol'
        current_protocol_validation_uplift_score(method, data_name, testFile, outcome_name, base_folder,
                                            treatment_plans, remove_attributes, threshold)


        testFile = f"{input_data_folder}/{data_name}_train_{timecn}.csv"
        method = 'CurrentProtocol'
        current_protocol_validation_uplift_score(method, data_name, testFile, outcome_name, base_folder,
                                                 treatment_plans, remove_attributes, threshold)

