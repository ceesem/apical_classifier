# apical_classifier
Tools for training and applying an apical classifier on skeleton data from the chunkedgraph

How to use:
```python
rfc = joblib.load(f"{model_dir}/point_model_current.pkl")
feature_cols = joblib.load(f"{model_dir}/feature_cols_current.pkl")
branch_params = joblib.load(f"{model_dir}/branch_params_current.pkl")

BranchClassifier = BranchClassifierFactory(rfc, feature_cols)
branch_classifier = BranchClassifier(**branch_params)
```

This generates a classifier that runs on point features extracted from:

```python
point_features_df = process_apical_features(nrn, peel_threshold=0.1)
branch_df = branch_classifier.fit_predict_data(
    point_features_df, "base_skind"
)
```

I'll explain more later.
