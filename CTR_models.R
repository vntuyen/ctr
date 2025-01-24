install_r_packages <- function(packages) {
  # Install and load R packages from the CRAN repository
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      tryCatch(
        {
          install.packages(pkg, repos = 'https://cloud.r-project.org', dependencies = TRUE)
          if (!require(pkg, character.only = TRUE)) {
            stop(paste("Failed to install or load package:", pkg))
          }
        },
        error = function(e) {
          message(paste("Error installing package:", pkg, "\n", e$message))
        }
      )
    }
  }
}

# List of R packages to install and load
r_packages <- c("dplyr", "graph", "rpart")

# Install and load all required packages
install_r_packages(r_packages)

# Load all packages
lapply(r_packages, library, character.only = TRUE)



# Install R package from Github repository
Rpackage5 = c("causalTree")
package.check <- lapply(
  Rpackage5,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      library(remotes)
      remotes::install_github("susanathey/causalTree")
      library(x, character.only = TRUE)
    }
  }
)


cross_validation_r <- function(
    input_data_folder,
    output_folder,
    data_name,
    causal_factors,
    outcome_name,
    remove_attributes,
    threshold) {



  for(timecn in 1: 5){
    trainingFile<- paste (input_data_folder, '/', data_name, '_train_' ,timecn,'.csv',sep='')
    trainingData <-read.csv(file = trainingFile)

    training_treatment <- c()
    for (i in 1:length(causal_factors)) {
      sum_of_column <- sum(trainingData[[causal_factors[[i]]]], na.rm = TRUE)
      if (sum_of_column > 0){
        training_treatment <- c(training_treatment, causal_factors[[i]])
      }
    }
    print(training_treatment)
    #Training phase
    trainingData <- dplyr::select (trainingData, -remove_attributes)

    CTRmodel <- build_causal_tree_model(data_name, trainingData, causal_factors, outcome_name)
    # CFmodel <- build_causal_forest_model(trainingData, training_treatment, outcome_name)

    #Testing phase
    testFile <- paste (input_data_folder, '/', data_name, '_test_' ,timecn,'.csv',sep='')
    method = 'CTR'
    estimateUpLiftScore_TE(CTRmodel,outcome_name,training_treatment, method, testFile,output_folder, exceptAttrbute = remove_attributes, threshold)
    # method= 'CF'
    # estimateUpLiftScore_TE(CFmodel,outcome_name,training_treatment, method, testFile,output_folder, exceptAttrbute = remove_attributes, threshold)

    # testFile <- paste (input_data_folder, '/', data_name, '_train_' ,timecn,'.csv',sep='')
    # method = 'CTR'
    # estimateUpLiftScore_TE(CTRmodel,outcome_name,training_treatment, method, testFile,output_folder, exceptAttrbute = remove_attributes, threshold)
    # # #
    # method= 'CF'
    # estimateUpLiftScore_TE(CFmodel,outcome_name, training_treatment, method, testFile,output_folder, exceptAttrbute = remove_attributes, threshold)
  }
}



estimateUpLiftScore<- function(model, outComeColName, estimationType, infileName, outputBaseFolder, exceptAttrbute = c(), threshold){
# estimateUpLiftScore_TE<- function(model, outComeColName,training_treatment, estimationType, infileName, outputBaseFolder, exceptAttrbute = c(), threshold){
    input_data <-read.csv(file = infileName)
  # data <- subset(input_data, select = -exceptAttrbute)
  # data <- input_data[ , !(names(df) %in% exceptAttrbute)]
  data <- input_data

  data['CURRENT_TP'] <- ''
  data['OPTIMAL_TP'] = ''
  data['LIFT_SCORE'] = 0
  data ['FOLLOW_REC'] = 0
  # data ['Real_OUTCOME'] = 0

  data ['Y_TREATED'] = 0
  data ['N_TREATED'] = 0
  data ['Y_UNTREATED'] = 0
  data ['N_UNTREATED'] = 0
  data['UPLIFT'] = 0

  for(row in 1: nrow(data)){
    for(i in 1: length(causal_factors)){
      if (data[row,causal_factors[[i]]] == 1){
        data[row,'CURRENT_TP'] <- causal_factors[[i]]
      }
    }


    inputRow = dplyr::select (data[row, ], -c('CURRENT_TP','LIFT_SCORE', 'OPTIMAL_TP','UPLIFT',
                                              'Y_TREATED','N_TREATED','Y_UNTREATED','N_UNTREATED', 'FOLLOW_REC', outcome_name, exceptAttrbute ))
    if(estimationType == 'CTR'){
      rowTE <- predict_causal_effect_row(model, inputRow, threshold )
    }

    prevLift = -9999
    highestTreatmentName <- ''

    for(i in 1: length(causal_factors)){
      data[row,causal_factors[[i]]] <- rowTE[causal_factors[[i]]]
      # row[causal_factors[[i]]]

      if(prevLift < rowTE[causal_factors[[i]]]){
        prevLift <- rowTE[causal_factors[[i]]]
        if(prevLift > threshold) {
          highestTreatmentName <- causal_factors[[i]]
        }
      }
    }
    data[row,'LIFT_SCORE'] <- prevLift
    data[row,'OPTIMAL_TP'] <- highestTreatmentName
  }

  data <- data[order(-data$LIFT_SCORE),]

  y_treated <- 0
  n_treated <- 0
  y_untreated <- 0
  n_untreated <- 0

  for(row in 1: nrow(data)){

    data[row,'N_TREATED']<- n_treated
    data[row,'Y_TREATED']<- y_treated
    data[row,'N_UNTREATED']<- n_untreated
    data[row,'Y_UNTREATED']<- y_untreated

    if((data[row,'OPTIMAL_TP'] == data[row,'CURRENT_TP'])){
      data [row, 'FOLLOW_REC'] = 1
      n_treated <- n_treated + 1
      data[row,'N_TREATED']<- n_treated
      if(data[row,outComeColName] == 1){
        y_treated <- y_treated + 1
        data[row,'Y_TREATED']<- y_treated
      }

    }else{

      n_untreated <- n_untreated + 1
      data[row,'N_UNTREATED']<- n_untreated

      if(data[row,outComeColName] == 1){
        y_untreated <- y_untreated + 1
        data[row,'Y_UNTREATED']<- y_untreated
      }
    }

    if(n_treated == 0) {
      data[row,'UPLIFT'] = 0
    }else if(n_untreated == 0){
      data[row,'UPLIFT'] = 0
    }else{
      liftestimate = ((y_treated/n_treated) - (y_untreated/n_untreated) )*(n_treated + n_untreated)
      qiniestimate = ((y_treated) - (y_untreated*(n_treated/n_untreated) ))
      data[row,'UPLIFT'] <- liftestimate
    }
  }

  ## update uplift by percentage

  totalIncrease <- ((y_treated/n_treated) - (y_untreated/n_untreated) )

  for(row in 1: nrow(data)){

    n_treated <- data[row,'N_TREATED']
    y_treated <- data[row,'Y_TREATED']
    n_untreated <- data[row,'N_UNTREATED']
    y_untreated <- data[row,'Y_UNTREATED']

    liftestimate <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))
    liftestimateWithBase <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))/totalIncrease
    data[row,'UPLIFT'] <- liftestimate

  }

  data['RealOUTCOME'] <- data[outcome_name]
  ##Output
  # ##Create Output Folder
  outputDir <- file.path(outputBaseFolder, 'output')
  if (!dir.exists(outputDir)){   #Check existence of directory and create it if it doesn't exist
    dir.create(outputDir)
  }
  methodFolder<-file.path(outputDir , estimationType)
  if (!dir.exists(methodFolder)){   #Check existence of directory and create it if it doesn't exist
    dir.create(methodFolder)
  }
  # outputDataFolder <- paste0(data_name, '_Out')
  outputdata_name <-file.path(methodFolder, data_name)
  if (!dir.exists(outputdata_name)){ #Check existence of directory and create it if it doesn't exist
    dir.create(outputdata_name)
  }

  fileName = basename(infileName)
  fileNameParts <- strsplit(fileName ,'\\.')
  fileNameParts <- unlist(fileNameParts)

  secondFileNameParts <- strsplit(fileNameParts[1] ,'_')
  secondFileNameParts <- unlist(secondFileNameParts)

  newFileName <- paste(c(secondFileNameParts[1],'_', secondFileNameParts[2], '_', estimationType, '_', threshold, '_', secondFileNameParts[3],
                         '.', fileNameParts[2]), collapse = "")

  fullPath <- paste(c(outputdata_name,'/',newFileName ), collapse = "")
  write.csv(data,fullPath, row.names = FALSE)

}

estimateUpLiftScore_TE<- function(model, outComeColName,training_treatment, estimationType, infileName, outputBaseFolder, exceptAttrbute = c(), threshold){
    input_data <-read.csv(file = infileName)

  data <- input_data

  data['CURRENT_TP'] <- ''
  data['OPTIMAL_TP'] = ''
  data['LIFT_SCORE'] = 0
  data ['FOLLOW_REC'] = 0
  # data ['Real_OUTCOME'] = 0

  data ['Y_TREATED'] = 0
  data ['N_TREATED'] = 0
  data ['Y_UNTREATED'] = 0
  data ['N_UNTREATED'] = 0
  data['UPLIFT'] = 0

  for(row in 1: nrow(data)){
    for(i in 1: length(causal_factors)){
      if (data[row,causal_factors[[i]]] == 1){
        data[row,'CURRENT_TP'] <- causal_factors[[i]]
      }
    }


    inputRow = dplyr::select (data[row, ], -c('CURRENT_TP','LIFT_SCORE', 'OPTIMAL_TP','UPLIFT',
                                              'Y_TREATED','N_TREATED','Y_UNTREATED','N_UNTREATED', 'FOLLOW_REC', outcome_name, exceptAttrbute ))
    if(estimationType == 'CTR'){
      rowTE <- predict_causal_effect_row(model, inputRow, threshold )
    }
    if(estimationType == 'CF'){
      # covariates = unique(setdiff(colnames(inputRow), c(outcome_name, training_treatment)))
      inputRow <- dplyr::select (inputRow, -training_treatment)
      rowTE <- predict_causal_effect_row(model, inputRow, threshold )
    }


    prevLift = -9999
    highestTreatmentName <- ''

    for(i in 1: length(causal_factors)){
      data[row,causal_factors[[i]]] <- rowTE[causal_factors[[i]]]

      if(prevLift < rowTE[causal_factors[[i]]]){
        prevLift <- rowTE[causal_factors[[i]]]
        if(prevLift > threshold) {
          highestTreatmentName <- causal_factors[[i]]
        }
      }
    }
    data[row,'LIFT_SCORE'] <- prevLift
    data[row,'OPTIMAL_TP'] <- highestTreatmentName
  }

  # data <- data[order(-data$LIFT_SCORE),]

  y_treated <- 0
  n_treated <- 0
  y_untreated <- 0
  n_untreated <- 0

  for(row in 1: nrow(data)){
    # TREATMENT_NAME = data[row,'TREATMENT_NAME']
    # TREATMENT_NAME <- toString(TREATMENT_NAME)

    data[row,'N_TREATED']<- n_treated
    data[row,'Y_TREATED']<- y_treated
    data[row,'N_UNTREATED']<- n_untreated
    data[row,'Y_UNTREATED']<- y_untreated

    # if((TREATMENT_NAME != 'NA')&&(data[row,TREATMENT_NAME] == 1)){
    if((data[row,'OPTIMAL_TP'] == data[row,'CURRENT_TP'])){
      data [row, 'FOLLOW_REC'] = 1
      n_treated <- n_treated + 1
      data[row,'N_TREATED']<- n_treated
      if(data[row,outComeColName] == 1){
        y_treated <- y_treated + 1
        data[row,'Y_TREATED']<- y_treated
      }

    }else{

      n_untreated <- n_untreated + 1
      data[row,'N_UNTREATED']<- n_untreated

      if(data[row,outComeColName] == 1){
        y_untreated <- y_untreated + 1
        data[row,'Y_UNTREATED']<- y_untreated
      }
    }

    if(n_treated == 0) {
      data[row,'UPLIFT'] = 0
    }else if(n_untreated == 0){
      data[row,'UPLIFT'] = 0
    }else{
      liftestimate = ((y_treated/n_treated) - (y_untreated/n_untreated) )*(n_treated + n_untreated)
      qiniestimate = ((y_treated) - (y_untreated*(n_treated/n_untreated) ))
      data[row,'UPLIFT'] <- liftestimate
    }
  }

  ## update uplift by percentage

  totalIncrease <- ((y_treated/n_treated) - (y_untreated/n_untreated) )

  for(row in 1: nrow(data)){

    n_treated <- data[row,'N_TREATED']
    y_treated <- data[row,'Y_TREATED']
    n_untreated <- data[row,'N_UNTREATED']
    y_untreated <- data[row,'Y_UNTREATED']

    liftestimate <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))
    liftestimateWithBase <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))/totalIncrease
    data[row,'UPLIFT'] <- liftestimate

  }

  data['RealOUTCOME'] <- data[outcome_name]
  # data['PAM50'] <- data['PAM50']
  ##Output
  ##Output
  # ##Create Output Folder
  outputDir <- file.path(outputBaseFolder, 'output')
  if (!dir.exists(outputDir)){   #Check existence of directory and create it if it doesn't exist
    dir.create(outputDir)
  }
  methodFolder<-file.path(outputDir , estimationType)
  if (!dir.exists(methodFolder)){   #Check existence of directory and create it if it doesn't exist
    dir.create(methodFolder)
  }
  # outputDataFolder <- paste0(data_name, '_Out')
  outputdata_name <-file.path(methodFolder, data_name)
  if (!dir.exists(outputdata_name)){ #Check existence of directory and create it if it doesn't exist
    dir.create(outputdata_name)
  }

  fileName = basename(infileName)
  fileNameParts <- strsplit(fileName ,'\\.')
  fileNameParts <- unlist(fileNameParts)

  secondFileNameParts <- strsplit(fileNameParts[1] ,'_')
  secondFileNameParts <- unlist(secondFileNameParts)

  newFileName <- paste(c(secondFileNameParts[1],'_', secondFileNameParts[2], '_', estimationType, '_', threshold, '_', secondFileNameParts[3],
                         '.', fileNameParts[2]), collapse = "")

  fullPath <- paste(c(outputdata_name,'/',newFileName ), collapse = "")
  # write.csv(data,fullPath, row.names = FALSE)

}



#######################
## # build Causal Tree model
build_causal_tree_model <- function(data_name, trainingData, causal_factors, outcome_name, splitRule = 'CT', cvRule = 'CT') {
  results <- list()
  output_folder = getwd()
  out_put <- file.path(output_folder, 'output')

  if (!dir.exists(out_put)) {
    dir.create(out_put)
  }
  outputDir <- file.path(out_put, 'TrainedModel')

  if (!dir.exists(outputDir)) {
    dir.create(outputDir)
  }

  for (i in 1:length(causal_factors)) {
    reg <- glm(as.formula(paste(causal_factors[[i]], ' ~ . -', outcome_name, sep = "")),
               family = binomial,
               data = trainingData)

    propensity_scores <- reg$fitted
    tree <- causalTree(as.formula(paste(outcome_name, ' ~ . ', sep = "")),
                       data = trainingData,
                       treatment = trainingData[[causal_factors[[i]]]],
                       split.Rule = splitRule,
                       cv.option = cvRule,
                       split.Honest = TRUE,
                       cv.Honest = TRUE,
                       split.Bucket = FALSE,
                       xval = 5,
                       cp = 0,
                       minsize = 3L,
                       propensity = propensity_scores)


    opcp <- tree$cptable[, 2][which.min(tree$cptable[, 4])]
    opfit <- prune(tree, opcp)


    treeFileName <- paste(data_name, causal_factors[[i]], '_tree.png', sep = '')
    treeFile <- file.path(outputDir, treeFileName)
    png(file = treeFile, width = 1200, height = 900)

    rpart.plot(opfit)
    dev.off()

    treeModel <- list()
    treeModel$model <- opfit
    treeModel$factor <- causal_factors[[i]]
    results <- append(results, list(treeModel))

    trainingModel <- opfit
    trainedModelFileName <- paste(data_name, causal_factors[[i]], '_trainedModel.RDS', sep = '')

    trainedModelFile <- paste (outputDir, '/', trainedModelFileName,sep='')
    saveRDS(trainingModel, file=trainedModelFile)
  }

  return(results)
}


#######################
## # build Causal Tree model for each Treatment Plan
build_causal_tree_model_e <- function(trainingData, causal_factors, outcome_name, splitRule = 'CT', cvRule = 'CT') {
  results <- list()
  output_folder = getwd()
  out_put <- file.path(output_folder, 'output')

  if (!dir.exists(out_put)) {
    dir.create(out_put)
  }
  outputDir <- file.path(out_put, 'Tree')

  if (!dir.exists(outputDir)) {
    dir.create(outputDir)
  }

  covariates = unique(setdiff(colnames(trainingData), causal_factors))

  # for (i in 1:length(causal_factors)) {
  #
  #   forest <- causal_forest(X = trainingData[covariates], Y = trainingData[[outcome_name]],
  #                           W = trainingData[[causal_factors[[i]]]], num.trees = 4, sample.fraction=0.5, min.node.size=2
  #                           )



  for (i in 1:length(causal_factors)) {
    reg <- glm(as.formula(paste(causal_factors[[i]], ' ~ . -', outcome_name, sep = "")),
               family = binomial,
               data = trainingData)

    propensity_scores <- reg$fitted
    tree <- causalTree(as.formula(paste(outcome_name, ' ~ . ', sep = "")),
                       data = trainingData[covariates],
                       treatment = trainingData[[causal_factors[[i]]]],
                       split.Rule = splitRule,
                       cv.option = cvRule,
                       split.Honest = TRUE,
                       cv.Honest = TRUE,
                       split.Bucket = FALSE,
                       xval = 5,
                       cp = 0,
                       minsize = 3L,
                       propensity = propensity_scores)


    # tree1 <- causalTree(as.formula(paste(outcomeName, ' ~. ', sep= ""))
    #                     , data = trainingData, treatment = trainingData[[causalFactor]],
    #                     split.Rule = splitRule, cv.option = cvRule,
    #                     split.Honest = T, cv.Honest = T, split.Bucket = F,
    #                     xval = 5, cp = 0, propensity = propensity_scores)


    opcp <- tree$cptable[, 2][which.min(tree$cptable[, 4])]
    opfit <- prune(tree, opcp)

    treeFileName <- paste(causal_factors[[i]], '_tree.png', sep = '')
    treeFile <- file.path(outputDir, treeFileName)
    png(file = treeFile, width = 1200, height = 900)

    rpart.plot(opfit)
    dev.off()

    treeModel <- list()
    treeModel$model <- opfit
    treeModel$factor <- causal_factors[[i]]
    results <- append(results, list(treeModel))
  }

  return(results)
}


#######################
## # build Causal Forest model
build_causal_forest_model <- function(trainingData, causal_factors, outcome_name, num.trees = 2000, sample.fraction=0.5, min.node.size=50) {
  results <- list()
  output_folder = getwd()
  outputDir <- file.path(output_folder, 'output')

  if (!dir.exists(outputDir)) {
    dir.create(outputDir)
  }
  covariates = unique(setdiff(colnames(trainingData), c(outcome_name, causal_factors)))

  for (i in 1:length(causal_factors)) {

    forest <- causal_forest(X = trainingData[covariates], Y = trainingData[[outcome_name]],
                            W = trainingData[[causal_factors[[i]]]], num.trees = 3, sample.fraction=0.5, min.node.size=3
                            )


    treeModel <- list()
    treeModel$model <- forest
    treeModel$factor <- causal_factors[[i]]
    results <- append(results, list(treeModel))
  }
  return(results)
}




## Predict the Highest treatment Effect
predict_causal_effect <-function(models, recordForEstimate, threshold){
  prevLift = -9999
  # treatmentName = 'NA'
  highestTreatmentName = ''
  result <- c()
  interRow <- recordForEstimate.copy()

  for(i in 1: length(models)){

    treeModel = models[[i]]
    result[[i]] <- predict(treeModel$model, recordForEstimate)

    if(prevLift < result[[i]]){
      prevLift <- result[[i]]
      if(prevLift > threshold) {
        highestTreatmentName <- treeModel$factor
      }

    }
  }
  return (list(prevLift, highestTreatmentName, result))
}


## Predict the Highest treatment Effect
predict_causal_effect_row <-function(models, recordForEstimate, threshold){
  interRow <- recordForEstimate

  for(i in 1: length(models)){

    treeModel = models[[i]]
    interRow[treeModel$factor] <- predict(treeModel$model, recordForEstimate)

  }
  return (interRow)
}



