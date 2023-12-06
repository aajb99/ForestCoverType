####### Load Libraries #######
#install.packages('tidyverse')
library(tidyverse)
#install.packages('tidymodels')
library(tidymodels)
#install.packages('DataExplorer')
#install.packages("poissonreg")
# library(poissonreg)
#install.packages("glmnet")
library(glmnet)
#library(patchwork)
# install.packages("rpart")
#install.packages('ranger')
library(ranger)
#install.packages('stacks')
library(stacks)
#install.packages('vroom')
library(vroom)
#install.packages('parsnip')
library(parsnip)
# install.packages('dbarts')
# library(dbarts)
#install.packages('embed')
library(embed)
library(themis)
library(ggplot2)
library(parsnip)
##############################

###############
##### EDA #####
###############

# data_train <- vroom("./data/train.csv")
# view(data_train)
# 
# cor_matrix <- cor(data_train)
# 
# # Create corr heatmap
# heatmap(cor_matrix,
#         col = colorRampPalette(c('blue', 'white', 'green'))(100),
#         margins = c(5, 5),
#         Rowv = NA,
#         Colv = NA)
# 
# cor_vals_foresttype <- cor_matrix[, 'Cover_Type']
# ordered_indeces <- order(-cor_vals_foresttype)
# ordered_cor_vals_foresttype <- cor_vals_foresttype[ordered_indeces]
# ordered_cor_vals_foresttype
# 
# boxplot(data_train$Soil_Type10 ~ data_train$Cover_Type,
#         col='steelblue',
#         main='Cover type by soil 10',
#         xlab='Cover Type',
#         ylab='Soil 10')
# 
# boxplot(data_train$Soil_Type29 ~ data_train$Cover_Type,
#         col='steelblue',
#         main='Cover type by soil 29',
#         xlab='Cover Type',
#         ylab='Soil 29')
# 
# boxplot(data_train$Wilderness_Area1 ~ data_train$Cover_Type,
#         col='steelblue',
#         main='Cover type by WA 1',
#         xlab='Cover Type',
#         ylab='WA 1')
# 
# boxplot(data_train$Wilderness_Area3 ~ data_train$Cover_Type,
#         col='steelblue',
#         main='Cover type by WA 3',
#         xlab='Cover Type',
#         ylab='WA 3')
# 
# boxplot(data_train$Hillshade_Noon ~ data_train$Cover_Type,
#         col='steelblue',
#         main='Cover type by noon shade',
#         xlab='Cover Type',
#         ylab='noon shade')

##########
# multicollinearity EDA
##########
# data_wild_areas <- data_train %>%
#   select(c('Wilderness_Area1',
#            'Wilderness_Area2',
#            'Wilderness_Area3',
#            'Wilderness_Area4'))
# 
# cor_matrix_wild_areas <- cor(data_wild_areas)
# 
# # Create corr heatmap
# heatmap(cor_matrix_wild_areas,
#         col = colorRampPalette(c('blue', 'white', 'green'))(100),
#         margins = c(5, 5),
#         Rowv = NA,
#         Colv = NA,
#         cexRow = 0.6,
#         cexCol = 0.6)


#############################################
##### Find value frequencies of factors #####
#############################################

# data_train['Id']
# 
# column_unique_df <- data.frame(column_name = character(0), frequency = list())
# # column_unique_df$frequency <- NA
# 
# for(column in colnames(data_train)){
#     unique_vals <- table(data_train[[column]])
#     sorted_vals <- sort(unique_vals, decreasing = TRUE)
#     second_largest <- as.numeric(sorted_vals[2])[1]
#     
#     new_row <- data.frame(column_name = column, frequency = second_largest)
#     
#     colnames(new_row) <- colnames(column_unique_df)
#     
#     column_unique_df <- rbind(column_unique_df, new_row)
#     
# }
# 
# colnames(column_unique_df) <- c('column_name', 'second_freq')
# view(column_unique_df)
# 
# sort(table(data_train[['Horizontal_Distance_To_Roadways']]), decreasing = TRUE)

#######################
##### Recipe/Bake #####
#######################

# Import Dataset:
data_train <- vroom("./data/train.csv") %>%
  mutate(Cover_Type=factor(Cover_Type))# grab training data

ncol(data_train)
# view(data_train)
# data_train$Cover_Type


rFormula <- Cover_Type ~ .

## For target encoding/Random Forests: ###
class_rf_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(c(12:55), fn = factor) %>%
  #step_other(all_nominal_predictors(), threshold = .001) %>%
  # step_zv(all_predictors()) %>% # eliminate zero variance predictors %>%
  step_nzv(freq_cut = 15070/50) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(Cover_Type)) #%>%
  #step_pca(all_predictors(), threshold = 0.8) %>% # Threshold between 0 and 1, test run for classification rf
  # step_smote(all_outcomes(), neighbors = 5)

prepped_recipe <- prep(class_rf_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)


########################################
##### Classification Random Forest #####
########################################

########## The following should be uncommented for SMOTE ##############

class_rf_mod <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 500) %>% #Type of model
  set_engine('ranger') %>%
  set_mode('classification')

pretune_workflow <- workflow() %>%
  add_recipe(class_rf_recipe) %>%
  add_model(class_rf_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(2,ncol(data_train)-1)),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(data_train, v = 3, repeats = 1)

# Run CV
CV_results <- pretune_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- pretune_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)

data_test <- vroom("./data/test.csv") # grab testing data

fct_predictions <- predict(final_wf,
                              new_data=data_test,
                              type="class") %>% # "class" or "prob"
  mutate(Id = data_test$Id) %>%
  #mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  mutate(Cover_Type = .pred_class) %>%
  select(Id, Cover_Type)

vroom_write(fct_predictions, "./data/for_covertype_pred_rf2.csv", delim = ",")
# save(file = 'amazon_penalized_wf.RData', list = c('final_wf'))
# load('amazon_penalized_wf.RData')





