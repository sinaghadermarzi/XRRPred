library(keras)

data <- read.csv("smote_1.csv")
test <- read.csv("test_set_resolution.txt",sep = "\t")

data <- scale(data) 

col_means_train <- attr(data, "scaled:center") 
col_stddevs_train <- attr(data, "scaled:scale")
test <- scale(test, center = col_means_train, scale = col_stddevs_train)



build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(data)[2]) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )
  
  model
}

model <- build_model()


print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

epochs <- 200

# Fit the model and store training stats
history <- model %>% fit(
  data,
  data$OUTPUT_Resolution,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)


test_predictions <- model %>% predict(test)
test_predictions[ , 1]