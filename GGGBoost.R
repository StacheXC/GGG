library(tidymodels)
library(vroom)
library(bonsai)
library(lightgbm)

train = vroom("train.csv")
test = vroom("test.csv")
train$type = as.factor(train$type)

my_recipe = recipe(type ~ ., train) |> 
  step_rm(id) |> 
  step_mutate(color = as.factor(color))

prepped_recipe = prep(my_recipe)
baked_recipe = bake(prepped_recipe, new_data = train)

boost_model = boost_tree(tree_depth=tune(),
                         trees=tune(),
                         learn_rate=tune()) |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(boost_model)

tuning_grid = grid_regular(tree_depth(),
                           trees(),
                           learn_rate(),
                           levels = 5)

folds = vfold_cv(train, v = 10, repeats = 1)

CV_results = wf |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy),
            control=control_grid(verbose=TRUE))

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
            file = "boostpred.csv",
            delim = ",")

