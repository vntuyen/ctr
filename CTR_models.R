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
r_packages <- c("dplyr", "rpart")

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


    set.seed(42)

    # tuned <- tune_causal_tree(
    #   data_name     = data_name,
    #   trainingData  = trainingData,      # your preprocessed df
    #   causal_factors= causal_factors,
    #   outcome_name  = outcome_name,
    #   seed          = 42
    # )
    #
    # # Best models
    # CTRmodel <- lapply(tuned$best_models, identity)
    CTRmodel <- build_causal_tree_model(data_name, trainingData, causal_factors, outcome_name)

    #Testing phase
    testFile <- paste (input_data_folder, '/', data_name, '_test_' ,timecn,'.csv',sep='')
    method = 'CTR'
    estimateUpLiftScore_TE(CTRmodel,data_name, outcome_name,training_treatment, method, testFile,output_folder, exceptAttrbute = remove_attributes, threshold)
   }
}



estimateUpLiftScore<- function(model, outComeColName, estimationType, infileName, outputBaseFolder, exceptAttrbute = c(), threshold){
# estimateUpLiftScore_TE<- function(model, outComeColName,training_treatment, estimationType, infileName, outputBaseFolder, exceptAttrbute = c(), threshold){
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

estimateUpLiftScore_TE<- function(model, data_name, outComeColName,training_treatment, estimationType, infileName, outputBaseFolder, exceptAttrbute = c(), threshold){
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



    # ✅ Create the new CSV file name
  new_file_name_follow <-   newFileName <- paste(c(secondFileNameParts[1],'_', secondFileNameParts[2], '_', estimationType, '_', threshold, '_', secondFileNameParts[3],
                         '_follow.', fileNameParts[2]), collapse = "")

  full_path_follow <- paste(c(outputdata_name,'/',new_file_name_follow ), collapse = "")

  # ✅ Merge FOLLOW_REC into original test_data
  test_data_with_rec <- input_data
  test_data_with_rec$FOLLOW_REC <- data$FOLLOW_REC

  # ✅ Save as CSV
  # write.csv(test_data_with_rec, full_path_follow, row.names = FALSE)

}



# #######################
# ## # build Causal Tree model
# build_causal_tree_model <- function(data_name, trainingData, causal_factors, outcome_name, splitRule = 'CT', cvRule = 'CT') {
#   results <- list()
#   output_folder = getwd()
#   out_put <- file.path(output_folder, 'output')
#
#   if (!dir.exists(out_put)) {
#     dir.create(out_put)
#   }
#   outputDir <- file.path(out_put, 'TrainedModel')
#
#   if (!dir.exists(outputDir)) {
#     dir.create(outputDir)
#   }
#
#   for (i in 1:length(causal_factors)) {
#     reg <- glm(as.formula(paste(causal_factors[[i]], ' ~ . -', outcome_name, sep = "")),
#                family = binomial,
#                data = trainingData)
#
#     propensity_scores <- reg$fitted
#     tree <- causalTree(as.formula(paste(outcome_name, ' ~ . ', sep = "")),
#                        data = trainingData,
#                        treatment = trainingData[[causal_factors[[i]]]],
#                        split.Rule = splitRule,
#                        cv.option = cvRule,
#                        split.Honest = TRUE,
#                        cv.Honest = TRUE,
#                        split.Bucket = FALSE,
#                        xval = 5,
#                        cp = 0,
#                        minsize = 3L,
#                        propensity = propensity_scores)
#
#
#     opcp <- tree$cptable[, 2][which.min(tree$cptable[, 4])]
#     opfit <- prune(tree, opcp)
#
#
#     treeFileName <- paste(data_name, causal_factors[[i]], '_tree.png', sep = '')
#     treeFile <- file.path(outputDir, treeFileName)
#     png(file = treeFile, width = 1200, height = 900)
#
#     rpart.plot(opfit)
#     dev.off()
#
#     treeModel <- list()
#     treeModel$model <- opfit
#     treeModel$factor <- causal_factors[[i]]
#     results <- append(results, list(treeModel))
#
#     trainingModel <- opfit
#     trainedModelFileName <- paste(data_name, causal_factors[[i]], '_trainedModel.RDS', sep = '')
#
#     trainedModelFile <- paste (outputDir, '/', trainedModelFileName,sep='')
#     saveRDS(trainingModel, file=trainedModelFile)
#   }
#
#   return(results)
# }
tune_causal_tree <- function(
  data_name,
  trainingData,
  causal_factors,
  outcome_name,
  seed = 42,
  grid = expand.grid(
    splitRule   = c("CT", "TOT"),
    minsize     = c(3L, 10L, 20L, 30L),
    splitBucket = c(FALSE, TRUE),
    bucketNum   = c(5L, 10L),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  ),
  xval = 10,
  trim_eps = 0.05,          # propensity clipping in [trim_eps, 1-trim_eps]
  use_one_se_rule = TRUE    # TRUE = 1-SE pruning; FALSE = min-xerror pruning
) {
  # Coerce grid columns once after expand.grid(...)
  grid$splitRule   <- as.character(grid$splitRule)
  grid$minsize     <- as.integer(grid$minsize)
  grid$splitBucket <- as.logical(grid$splitBucket)
  grid$bucketNum   <- as.integer(grid$bucketNum)

  stopifnot(outcome_name %in% names(trainingData))
  if (!all(causal_factors %in% names(trainingData))) {
    stop("Some causal_factors are not columns of trainingData.")
  }

  # ---- Stable contrasts to avoid surprises in model.matrix ----
  old_contr <- options(contrasts = c("contr.treatment", "contr.poly"))
  on.exit(options(old_contr), add = TRUE)

  # ---- Helpers ----
  to01 <- function(x) {
    if (is.logical(x)) return(as.integer(x))
    if (is.factor(x))  return(as.integer(x) - 1L)
    if (is.numeric(x)) {
      ux <- sort(unique(na.omit(x)))
      if (identical(ux, c(0, 1))) return(as.integer(x))
      if (identical(ux, c(1, 2))) return(as.integer(x - 1L))
      return(as.integer(x != 0))
    }
    stop("Outcome/treatment must be coercible to numeric 0/1.")
  }

  # ---- Coerce outcome & treatments to 0/1 ----
  trainingData[[outcome_name]] <- to01(trainingData[[outcome_name]])
  for (f in causal_factors) trainingData[[f]] <- to01(trainingData[[f]])

  # ---- Drop constant predictors (except outcome/treatments) ----
  is_const <- vapply(trainingData, function(z) length(unique(na.omit(z))) <= 1, logical(1))
  protect  <- names(trainingData) %in% c(outcome_name, causal_factors)
  keep     <- !(is_const & !protect)
  trainingData <- trainingData[, keep, drop = FALSE]

  # ---- Coerce grid column types once (avoid factor/length issues) ----
  grid$splitRule   <- as.character(grid$splitRule)
  grid$minsize     <- as.integer(grid$minsize)
  grid$splitBucket <- as.logical(grid$splitBucket)
  grid$bucketNum   <- as.integer(grid$bucketNum)

  # ---- Output holders ----
  all_results <- list()
  best_models <- list()

  set.seed(seed)  # global seed

  for (fac in causal_factors) {
    message(sprintf("Tuning CausalTree for %s ...", fac))

    # ----- Build propensity: fac ~ all covariates EXCEPT outcome & this treatment -----
    rhs_vars <- setdiff(names(trainingData), c(outcome_name, fac))
    rhs      <- if (length(rhs_vars) > 0) paste(rhs_vars, collapse = " + ") else "1"
    frm_prop <- as.formula(paste(fac, "~", rhs))

    use_brglm <- requireNamespace("brglm2", quietly = TRUE)
    if (use_brglm) {
      reg <- brglm2::brglm(
        formula = frm_prop,
        family  = binomial(link = "logit"),
        data    = trainingData,
        method  = "brglmFit"
      )
    } else {
      reg <- glm(
        formula = frm_prop,
        family  = binomial(link = "logit"),
        data    = trainingData,
        control = glm.control(maxit = 100)
      )
      if (!isTRUE(reg$converged)) {
        warning(sprintf("Propensity glm for %s did not fully converge. Consider installing {brglm2}.", fac))
      }
    }
    p <- as.numeric(reg$fitted.values)
    # Clip extreme propensities to stabilise splits
    p <- pmax(pmin(p, 1 - trim_eps), trim_eps)

    # ---- Grid search over tree hyperparams ----
    grid_scores <- data.frame(
      splitRule   = character(),
      minsize     = integer(),
      splitBucket = logical(),
      bucketNum   = integer(),
      cp          = numeric(),
      xerror      = numeric(),
      xstd        = numeric(),
      stringsAsFactors = FALSE
    )

    best_fit   <- NULL
    best_xerr  <- Inf
    best_cp    <- NA_real_
    best_cfg   <- NULL

    for (row in seq_len(nrow(grid))) {

      # 1) Coerce each value from the grid row to a scalar of the correct type
      split_rule   <- as.character(grid[row, "splitRule"][[1]])   # "CT" or "TOT"
      minsize_val  <- as.integer(  grid[row, "minsize"][[1]]   )
      split_bucket <- isTRUE(      grid[row, "splitBucket"][[1]] )
      bucket_num   <- as.integer(  grid[row, "bucketNum"][[1]]   )
      xval_k       <- as.integer(xval)  # fixed K folds

      # Safety: assert scalars
      stopifnot(length(split_rule)   == 1L)
      stopifnot(length(minsize_val)  == 1L)
      stopifnot(length(split_bucket) == 1L)
      stopifnot(length(bucket_num)   == 1L)
      stopifnot(length(xval_k)       == 1L)

      # 2) Honesty only applies to CT
      use_honesty <- identical(split_rule, "CT")

      # 3) Build args list selectively to avoid passing irrelevant params
      ct_args <- list(
        formula    = as.formula(paste(outcome_name, "~ .")),
        data       = trainingData,
        treatment  = trainingData[[fac]],
        split.Rule = split_rule,
        cv.option  = split_rule,
        xval       = xval_k,
        cp         = 0,
        minsize    = minsize_val,
        propensity = p
      )

      # Add honesty flags only for CT
      if (use_honesty) {
        ct_args$split.Honest <- TRUE
        ct_args$cv.Honest    <- TRUE
      }

      # Add bucket controls only when requested
      if (isTRUE(split_bucket)) {
        ct_args$split.Bucket <- TRUE
        ct_args$bucketNum    <- bucket_num
      } else {
        ct_args$split.Bucket <- FALSE
      }

      # 4) Deterministic CV fold assignment (seed) then fit
      set.seed(seed + row)
      tree <- do.call(causalTree::causalTree, ct_args)

      # 5) Robust CP selection (1-SE rule with safe fallbacks)
      cpt <- tree$cptable
      needed <- c("CP", "xerror", "xstd")

      if (!all(needed %in% colnames(cpt))) {
        opfit <- tree
        cp_sel <- NA_real_
        xerr   <- Inf
        xstd   <- NA_real_
      } else {
        ok <- is.finite(cpt[, "CP"]) & is.finite(cpt[, "xerror"]) & is.finite(cpt[, "xstd"])
        cpt_ok <- cpt[ok, , drop = FALSE]

        if (nrow(cpt_ok) == 0) {
          opfit <- tree
          cp_sel <- NA_real_
          xerr   <- Inf
          xstd   <- NA_real_
        } else {
          min_row <- which.min(cpt_ok[, "xerror"])
          pick_idx <- min_row

          if (isTRUE(use_one_se_rule)) {
            thresh <- cpt_ok[min_row, "xerror"] + cpt_ok[min_row, "xstd"]
            le_idx <- which(cpt_ok[, "xerror"] <= thresh)
            pick_idx <- if (length(le_idx) > 0) max(le_idx) else min_row
          }

          cp_sel <- as.numeric(cpt_ok[pick_idx, "CP"])
          if (!is.finite(cp_sel)) {
            opfit <- tree
          } else {
            opfit <- prune(tree, cp = cp_sel)
          }
          xerr <- as.numeric(cpt_ok[pick_idx, "xerror"])
          xstd <- as.numeric(cpt_ok[pick_idx, "xstd"])
        }
      }

      # Record scores as before
      grid_scores <- rbind(
        grid_scores,
        data.frame(
          splitRule   = split_rule,
          minsize     = minsize_val,
          splitBucket = split_bucket,
          bucketNum   = bucket_num,
          cp          = cp_sel,
          xerror      = xerr,
          xstd        = xstd,
          stringsAsFactors = FALSE
        )
      )

      if (is.finite(xerr) && (xerr < best_xerr)) {
        best_xerr <- xerr
        best_cp   <- cp_sel
        best_cfg  <- list(
          splitRule   = split_rule,
          minsize     = minsize_val,
          splitBucket = split_bucket,
          bucketNum   = bucket_num,
          cp          = best_cp
        )
        best_fit  <- opfit
      }

    } # end grid loop

    # If nothing was finite, fall back to last unpruned tree (rare)
    if (is.null(best_fit)) {
      warning(sprintf("[%s - %s] No finite xerror across grid; returning last unpruned model.",
                      data_name, fac))
      # Refit a simple default to return something
      set.seed(seed + 999)
      best_fit <- causalTree::causalTree(
        as.formula(paste(outcome_name, "~ .")),
        data         = trainingData,
        treatment    = trainingData[[fac]],
        split.Rule   = "CT",
        cv.option    = "CT",
        split.Honest = TRUE,
        cv.Honest    = TRUE,
        split.Bucket = FALSE,
        xval         = as.integer(xval),
        cp           = 0,
        minsize      = 3L,
        propensity   = p
      )
      best_cfg <- list(splitRule="CT", minsize=3L, splitBucket=FALSE, bucketNum=5L, cp=NA_real_)
      best_xerr <- NA_real_
    }

    # Save best for this factor
    best_models[[fac]] <- list(
      model     = best_fit,
      factor    = fac,
      cfg       = best_cfg,
      cv_xerror = best_xerr
    )
    all_results[[fac]] <- grid_scores
  }

  return(list(best_models = best_models, tuning_tables = all_results))
}




build_causal_tree_model <- function(
  data_name, trainingData, causal_factors, outcome_name,
  splitRule = "CT", cvRule = "CT", seed = 42, K = 5
) {
  stopifnot(outcome_name %in% names(trainingData))
  # Stable contrasts
  old_contr <- options(contrasts = c("contr.treatment", "contr.poly"))
  on.exit(options(old_contr), add = TRUE)

  # Helper: coerce to 0/1
  to01 <- function(x) {
    if (is.logical(x)) return(as.integer(x))
    if (is.factor(x))  return(as.integer(x) - 1L)
    if (is.numeric(x)) {
      ux <- sort(unique(na.omit(x)))
      if (identical(ux, c(0, 1))) return(as.integer(x))
      if (identical(ux, c(1, 2))) return(as.integer(x - 1L))
      return(as.integer(x != 0))
    }
    stop("Outcome/treatment must be coercible to numeric 0/1.")
  }

  # Coerce outcome + treatments
  trainingData[[outcome_name]] <- to01(trainingData[[outcome_name]])
  for (f in causal_factors) {
    stopifnot(f %in% names(trainingData))
    trainingData[[f]] <- to01(trainingData[[f]])
  }

  # Drop constant predictors to avoid model.matrix warnings
  is_constant <- vapply(trainingData, function(col) length(unique(na.omit(col))) <= 1, logical(1))
  # Never drop outcome or treatments
  protect <- names(trainingData) %in% c(outcome_name, causal_factors)
  keep <- !(is_constant & !protect)
  trainingData <- trainingData[, keep, drop = FALSE]

  # Output dirs
  output_folder <- getwd()
  out_put  <- file.path(output_folder, "output")
  if (!dir.exists(out_put)) dir.create(out_put)
  outputDir <- file.path(out_put, "TrainedModel")
  if (!dir.exists(outputDir)) dir.create(outputDir)

  results <- list()
  set.seed(seed)  # global seed once

  for (i in seq_along(causal_factors)) {
    fac <- causal_factors[[i]]

    # Deterministic RNG for this factor
    set.seed(seed + i)

    # ---- Propensity: fac ~ all covariates except outcome + current treatment ----
    rhs_vars <- setdiff(names(trainingData), c(outcome_name, fac))
    # If there are no covariates left, use intercept-only model
    rhs <- if (length(rhs_vars) > 0) paste(rhs_vars, collapse = " + ") else "1"
    frm_prop <- as.formula(paste(fac, "~", rhs))

    # Fit logistic propensity (robust settings)
    use_brglm <- requireNamespace("brglm2", quietly = TRUE)
    if (use_brglm) {
      reg <- brglm2::brglm(
        formula = frm_prop,
        family  = binomial(link = "logit"),
        data    = trainingData,
        method  = "brglmFit"  # bias-reduced, handles separation
      )
    } else {
      reg <- glm(
        formula = frm_prop,
        family  = binomial(link = "logit"),
        data    = trainingData,
        control = glm.control(maxit = 100)
      )
      if (!reg$converged) {
        warning(sprintf("Propensity glm for %s did not fully converge. Consider installing {brglm2}.", fac))
      }
    }
    propensity_scores <- as.numeric(reg$fitted.values)

    # ---- Causal tree (no honest.frac; set seed before call; keep honest splitting) ----
    tree <- causalTree::causalTree(
      formula       = as.formula(paste(outcome_name, "~ .")),
      data          = trainingData,
      treatment     = trainingData[[fac]],
      split.Rule    = splitRule,
      cv.option     = cvRule,
      split.Honest  = TRUE,
      cv.Honest     = TRUE,
      split.Bucket  = FALSE,
      xval          = K,         # keep numeric K folds; seed controls fold assignment
      cp            = 0,
      minsize       = 5L,
      propensity    = propensity_scores
    )

    # Prune by min xerror (deterministic with fixed seed)
    opcp  <- tree$cptable[, "CP"][which.min(tree$cptable[, "xerror"])]
    opfit <- prune(tree, cp = opcp)

    # Save plot + model
    # treeFile <- file.path(outputDir, paste0(data_name, "_", fac, "_tree.png"))
    # png(file = treeFile, width = 1200, height = 900)
    # rpart.plot::rpart.plot(opfit)
    # dev.off()

    saveRDS(opfit, file = file.path(outputDir, paste0(data_name, "_", fac, "_trainedModel.RDS")))

    results <- append(results, list(list(model = opfit, factor = fac)))
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



