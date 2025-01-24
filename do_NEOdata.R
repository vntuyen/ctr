source("CTR_models.R")
projectFolder = getwd()
print(projectFolder)
outputBaseFolder = projectFolder
output_folder=outputBaseFolder


data_name = 'NEO'

outcome_name = 'resp.pCR'
# remove_attributes = c('Trial.ID','resp.Chemosensitive','resp.Chemoresistant','RCB.score','RCB.category','Chemo.NumCycles','Chemo.first.Taxane','Chemo.first.Anthracycline', 'Chemo.second.Taxane', 'Chemo.second.Anthracycline', 'Chemo.any.Anthracycline','Chemo.any.antiHER2','PAM50')
remove_attributes = c('Trial.ID','resp.Chemosensitive','resp.Chemoresistant','RCB.score','RCB.category','Chemo.NumCycles','Chemo.first.Taxane','Chemo.first.Anthracycline', 'Chemo.second.Taxane', 'Chemo.second.Anthracycline', 'Chemo.any.Anthracycline','Chemo.any.antiHER2')
black_list = NULL
threshold <- -999

input_data_folder = paste(projectFolder,'/input/',data_name,sep='')

causal_factors = c('TP1','TP2','TP3','TP4')
# causal_factors = c('TP1','TP2','TP3')


cross_validation_r(input_data_folder,
                   output_folder,
                   data_name,
                   causal_factors,
                   outcome_name,
                   remove_attributes,
                   threshold)