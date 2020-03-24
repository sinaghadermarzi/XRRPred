  library(keras)


test <- read.csv("test_set_resolution.txt", sep="\t")

excluded = c()
target_col = c("OUTPUT_Resolution")
feature_cols = !names(test) %in% c(excluded,target_col)

x_test = data.matrix(test[feature_cols])
y_test = data.matrix(test[target_col])

# load_model_hdf5("NN_model_dump.hdf5", compile = TRUE)
model_from_saved_model("model_dump")

test_predictions <- model %>% predict(x_test)
test_predictions[ , 1] 
corr = cor(test_predictions,y_test,method = "spearman")
print(corr)