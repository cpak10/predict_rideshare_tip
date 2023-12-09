# Predict Tipping in NYC Rideshare Trips
The script trains tree-based machine learning methods to predict whether a NYC rideshare trip ends in a tip of any amount. Compare how the machine learning predictions perform compared to random assignment.

## Data
The dataset captures all trips facilitated by Uber, Lyft, Juno, and Via in New York City within the years 2019-2022. Each table contains detailed information about each trip, including pick-up and drop-off times and locations, fare, and trip distance. The data can be found at the following link: https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data

## Deployment
1. Download and save data into operative folder
2. pip3 install -r requirements.txt
3. python3 predict_tip.py

## Results
The experiment designed in the script puts forward the hypothesis that using machine learning to predict which rides resulted in a tip of any amount would be more accurate than random prediction, based on f1-score. To create the scenario to disprove the null hypothesis, 100 samples of 100,000 rides each were bucketed. 99 of the samples were randomly assigned prediction values of tip or no tip. The last remaining sample was used to test the machine learning models’ predictions. To disprove the null hypothesis, the machine learning prediction would need to result in an f1-score above the mean f1-score of the randomly assigned samples with a p-score below 0.05.

Three decision tree based models were trained and tuned: simple decision tree, XGBoost, and random forest. These tree based models were chosen due to their ability to effectively handle the dataset, which boasts a large number of features and observations. Tree based models would be more robust than linear models and more efficient than SVMs. All three models produced statistically significant predictions of tip likelihood at p < 0.05. However, only the random forest model proved to have an f1-score higher than the mean f1-score of the randomly assigned samples.

![experiment](https://github.com/cpak10/predict_rideshare_tip/assets/64233202/ba815528-3cd8-404f-9ccb-2f45c8937955)

As seen in the figure above, the randomly assigned samples form a normal-like distribution in the middle with the three machine learning test results far outside of the distribution. While the simple decision tree (“tree”) and XGBoost (“xgb”) perform worse than the randomly assigned samples, the random forest (“rf”) model performs far better. The random forest produces a recall of 0.51, precision of 0.28, and an f1-score of 0.36.

While the random forest result disproves the null hypothesis with its statistically significant f1-score, it does not appear to be meaningfully significant. With neither recall nor precision greatly surpassing the odds of a coin flip, it would be difficult to deploy this model in the world expecting meaningful results.
