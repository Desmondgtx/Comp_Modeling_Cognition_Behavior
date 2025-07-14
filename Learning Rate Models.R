## Modeling Learning Rate 
## Computational Modeling of Cognition and Behavior

# Libraries
library(ggplot2)

# Seed
set.seed(123)

# Power Law Function model
power_law = function(N, beta, scale = 6000) 
{ # Where N = number of trial, beta = learning rate
    scale * N^(-beta)
}

# Exponential Function model
exp_model = function(N, alpha, scale = 6000)
{ # Where alpha = learning rate
  scale * exp(-alpha*N)
}

# Trials
trials = 1:150

# Parameters
beta = 0.4 # Power Law Function, learning rate
alpha = 0.5 # Exponential Function, learning rate

# Model predictions
power_predictions = power_law(trials, beta)
exp_predictions = exp_model(trials, alpha)

