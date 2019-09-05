setwd('~/git/Telofish_NN/')

pred_vs_data <- read.csv("pred_vs_data.csv")

test_set <- pred_vs_data[pred_vs_data$set..1.test. == 1,]
train_set <- pred_vs_data[pred_vs_data$set..1.test. == 2,]


scatter.smooth(x=test_set$X..actual, y=test_set$predicted, main="Actual ~ Predicted [Test Set]")  # scatterplot
scatter.smooth(x=train_set$X..actual, y=train_set$predicted, main="Actual ~ Predicted [Train Set]")  # scatterplot


test_r2 = lm(X..actual ~ predicted, data=test_set)
#Then we extract the coefficient of determination from the r.squared attribute of its summary.

summary(test_r2)$r.squared 

### NOTE:
## This is at 26hrs training on ~600 images, validation on 68 test images. Need to implement resampling strat.