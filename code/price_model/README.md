# Price model training (regression)

`train_price_model.py` trains and compares five regressors (Random Forest,
Gradient Boosting, LightGBM, CatBoost and a Voting Ensemble) for the PickSpace
house and condo price prediction.

What the script does
- Splits the data **by project** (`GroupShuffleSplit`), not by row, so units from the
  same project never appear in both train and test (prevents data leakage).
- Drops second-hand records that share a project with the test set.
- Selects features with backward elimination using `GroupKFold` cross-validation on the training set only.
- Tunes each model with `RandomizedSearchCV` (25 iterations, 5 group folds), scored by MAE in baht.
- Reports MAE, MAPE, R² and the train/test R² gap, then saves the best model.

Result on the 119 held-out new-build units: Gradient Boosting, MAE 496,828 THB,
MAPE 13.3%, R² 0.861.

Run: `pip install -r requirements.txt`, put the feature files (`X_features.csv`,
`y_target.csv`, `reference_ids.csv`) in a `data_plus/` folder next to the script, then `python train_price_model.py`.
The training data is not included in this repository (the listings were scraped
and are kept private).
