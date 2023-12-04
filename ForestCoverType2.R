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


# Import Dataset:
data_train <- vroom("./data/train.csv") %>%
  mutate(Cover_Type=factor(Cover_Type))# grab training data

# data_train


#######################
##### Recipe/Bake #####
#######################

# ncol(data_train)
# # view(data_train)
# data_train$Cover_Type


rFormula <- Cover_Type ~ .

## For target encoding/Random Forests: ###
class_nb_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(c(12:55), fn = factor) %>%
  #step_other(all_nominal_predictors(), threshold = .001) %>%
  step_zv(all_predictors()) %>% # eliminate zero variance predictors
  step_lencode_glm(all_nominal_predictors(), outcome = vars(Cover_Type)) #%>%
#step_pca(all_predictors(), threshold = 0.8) %>% # Threshold between 0 and 1, test run for classification rf
  #step_smote(all_outcomes(), neighbors = 5)

prepped_recipe <- prep(class_nb_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)


################################
##### Naive Bayes Approach #####
################################

########## The following should be uncommented for SMOTE ##############

install.packages('discrim')
library(discrim)
install.packages('naivebayes')
library(naivebayes)

# nb model
nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
                        set_mode('classification') %>%
                        set_engine('naivebayes')

nb_wf <- workflow() %>%
  add_recipe(class_nb_recipe) %>%
  add_model(nb_mod)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(data_train, v = 10, repeats = 1)

# Run CV
CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)

data_test <- vroom("./data/test.csv") # grab testing data

fct_predictions <- predict(final_wf,
                              new_data=data_test,
                              type="class") %>% # "class" or "prob"
  mutate(Id = data_test$Id) %>%
  mutate(Cover_Type = .pred_class) %>%
  select(Id, Cover_Type)

vroom_write(fct_predictions, "./data/fct_pred_nb2.csv", delim = ",")
# save(file = 'amazon_penalized_wf.RData', list = c('final_wf'))
# load('amazon_penalized_wf.RData')