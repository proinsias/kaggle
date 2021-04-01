import os

import neptune.new as neptune

run = neptune.init(
    project='proinsias/sandbox',
    api_token=os.environ['NEPTUNE_API_TOKEN'],
)

# Track metadata and hyperparameters of your run
run["JIRA"] = "NPT-952"
run["algorithm"] = "ConvNet"

params = {
    "batch_size": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "optimizer": "Adam"
}
run["parameters"] = params


# Track the training process by logging your training metrics
for epoch in range(100):
    run["train/accuracy"].log(epoch * 0.6)
    run["train/loss"].log(epoch * 0.4)

# Log the final results
run["f1_score"] = 0.66

