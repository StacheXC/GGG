library(tidymodels)
library(vroom)
library(embed)

train = vroom("train.csv")
test = vroom("test.csv")
train$type = as.factor(train$type)

my_recipe = recipe(type ~ ., train) |> 
  step_rm(id) |> 
  step_mutate(color = as.factor(color)) |> 
  step_lencode_mixed(color, outcome = vars(type)) |> 
  step_normalize(all_numeric_predictors())
  
prepped_recipe = prep(my_recipe)
baked_recipe = bake(prepped_recipe, new_data = train)

model = nearest_neighbor(neighbors = tune()) |> 
  set_mode("classification") |> 
  set_engine("kknn")

wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(model)

tuning_grid = grid_regular(neighbors(),
                           levels = 10)

folds = vfold_cv(train, v = 10, repeats = 1)

CV_results = wf |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune = CV_results |> 
  select_best()

final_wf = wf |> 
  finalize_workflow(bestTune) |> 
  fit(data = train)

predictions = predict(final_wf,
                      new_data = test,
                      type = "class")

kaggle_submission = bind_cols(test["id"], predictions[".pred_class"]) |> 
  rename("id" = id, "type" = .pred_class)

vroom_write(x = kaggle_submission, 
            file = "knnpred.csv",
            delim = ",")






