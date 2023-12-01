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
##############################


# Import Dataset:
data_train <- vroom("./data/train.csv") %>%
  mutate(Cover_Type=factor(Cover_Type))# grab training data

# data_train


###############
##### EDA #####
###############

cor_matrix <- cor(data_train)

# Create corr heatmap
heatmap(cor_matrix,
        col = colorRampPalette(c('blue', 'white', 'green'))(100),
        margins = c(5, 5),
        Rowv = NA,
        Colv = NA)

cor_vals_foresttype <- cor_matrix[, 'Cover_Type']
ordered_indeces <- order(-cor_vals_foresttype)
ordered_cor_vals_foresttype <- cor_vals_foresttype[ordered_indeces]
ordered_cor_vals_foresttype

boxplot(data_train$Soil_Type10 ~ data_train$Cover_Type,
        col='steelblue',
        main='Cover type by soil 10',
        xlab='Cover Type',
        ylab='Soil 10')

boxplot(data_train$Soil_Type29 ~ data_train$Cover_Type,
        col='steelblue',
        main='Cover type by soil 29',
        xlab='Cover Type',
        ylab='Soil 29')

boxplot(data_train$Wilderness_Area1 ~ data_train$Cover_Type,
        col='steelblue',
        main='Cover type by WA 1',
        xlab='Cover Type',
        ylab='WA 1')

boxplot(data_train$Wilderness_Area3 ~ data_train$Cover_Type,
        col='steelblue',
        main='Cover type by WA 3',
        xlab='Cover Type',
        ylab='WA 3')

boxplot(data_train$Hillshade_Noon ~ data_train$Cover_Type,
        col='steelblue',
        main='Cover type by noon shade',
        xlab='Cover Type',
        ylab='noon shade')


#######################
##### Recipe/Bake #####
#######################

rFormula <- Cover_Type ~ .

## For target encoding/Random Forests: ###
class_rf_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(c(12:55), fn = factor) %>%
  #step_other(all_nominal_predictors(), threshold = .001) %>%
  step_zv(all_predictors()) %>% # eliminate zero variance predictors
  step_lencode_glm(all_nominal_predictors(), outcome = vars(Cover_Type)) #%>% # get hours
  #step_pca(all_predictors(), threshold = 0.8) %>% # Threshold between 0 and 1, test run for classification rf
  # step_smote(all_outcomes(), neighbors = 5)

prepped_recipe <- prep(class_rf_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)

# ncol(baked_data1)

########################################
##### Classification Random Forest #####
########################################

########## The following should be uncommented for SMOTE ##############

class_rf_mod <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 800) %>% #Type of model
  set_engine('ranger') %>%
  set_mode('classification')

pretune_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(class_rf_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,ncol(data_train)-1)),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(data_train, v = 10, repeats = 1)

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

amazon_predictions <- predict(final_wf,
                              new_data=data_test,
                              type="prob") %>% # "class" or "prob"
  mutate(Id = data_test$id) %>%
  #mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  mutate(ACTION = .pred_1) %>%
  select(-.pred_0, -.pred_1)

vroom_write(amazon_predictions, "./data/amazon_pred_rf3.csv", delim = ",")
save(file = 'amazon_penalized_wf.RData', list = c('final_wf'))
load('amazon_penalized_wf.RData')





