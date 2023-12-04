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

################################################################################
# Recipe/Bake

rFormula <- Cover_Type ~ .

fct_bart_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(c(12:55), fn = factor) %>%
  step_zv(all_predictors()) %>% # eliminate zero variance predictors
  step_lencode_glm(all_nominal_predictors(), outcome = vars(Cover_Type)) #%>%

prepped_recipe <- prep(fct_bart_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)


################################################################################
# Bayesian Additive Regression Trees (BART)

bart_mod <- parsnip::bart(mode = 'classification',
                          engine = 'dbarts',
                          trees = 30)

## Set Workflow
bart_wf <- workflow() %>%
  add_recipe(fct_bart_recipe) %>%
  add_model(bart_mod) %>%
  fit(data=data_train)

data_test <- vroom("./data/test.csv") # input test data

final_bart_preds <- predict(bart_wf, new_data = data_test) %>% # predictions
  mutate(Id = data_test$Id) %>%
  #mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  mutate(Cover_Type = .pred_class) %>%
  select(Id, Cover_Type)

vroom_write(final_bart_preds, "fct_predictions_bart.csv", delim = ",")




