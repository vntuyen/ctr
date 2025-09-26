source("CTR_models.R")

projectFolder <- getwd()
outputBaseFolder <- projectFolder
output_folder <- outputBaseFolder

# -----------------------------
# Define dataset configurations
# -----------------------------
configs <- list(
  list(
    data_name = "NEO",
    outcome_name = "resp.pCR",
    causal_factors = c("TP1","TP2","TP3","TP4"),
    remove_attributes = c(
      "Trial.ID","resp.Chemosensitive","resp.Chemoresistant","RCB.score",
      "RCB.category","Chemo.NumCycles","Chemo.first.Taxane","Chemo.first.Anthracycline",
      "Chemo.second.Taxane","Chemo.second.Anthracycline",
      "Chemo.any.Anthracycline","Chemo.any.antiHER2"
    )
  ),
  list(
    data_name = "DUKE",
    outcome_name = "pCR",
    causal_factors = c("TP1","TP2"),
    remove_attributes = c(
      "pid","Chemotheray.Adjuvant",
      "rfs_event","rfs_time","lrfs_event","lrfs_time",
      "drfs_event","drfs_time"
    )
  )
)

black_list <- NULL
threshold <- -999 ##A threshold -999 (very small) means using all predited treatment effect values to compare.

message("Project folder: ", projectFolder)

# -----------------------------
# Run pipeline for each dataset
# -----------------------------
for (cfg in configs) {
  data_name       <- cfg$data_name
  outcome_name    <- cfg$outcome_name
  causal_factors  <- cfg$causal_factors
  remove_attributes <- cfg$remove_attributes

  input_data_folder <- file.path(projectFolder, "input", data_name)

  message("\n==============================")
  message("Running cross_validation_r for: ", data_name)
  message("Outcome: ", outcome_name)
  message("Causal factors: ", paste(causal_factors, collapse = ", "))
  message("Input folder: ", input_data_folder)
  message("==============================")

  # Wrap in tryCatch so one failure doesn’t stop the other dataset
  tryCatch(
    {
      cross_validation_r(
        input_data_folder = input_data_folder,
        output_folder     = output_folder,
        data_name         = data_name,
        causal_factors    = causal_factors,
        outcome_name      = outcome_name,
        remove_attributes = remove_attributes,
        threshold         = threshold
      )
      message("✅ Finished: ", data_name)
    },
    error = function(e) {
      message("❌ Error while running ", data_name, ": ", conditionMessage(e))
    }
  )
}

message("\nAll tasks attempted. Check the output folders for results.")
