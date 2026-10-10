# scikit-learn Cookbook (3rd Edition) — Reproduced Notebooks, Chapters 1–5

Executable Jupyter/Colab notebooks that **reproduce the code**, **summarize each recipe**, and **explain the theory** behind *scikit-learn Cookbook, Third Edition* (over 80 recipes for machine learning in Python). Each chapter is one self-contained notebook, runnable top to bottom.

> Status: Chapters 1–5 done. Chapters 6–13 planned.

## Notebooks

| # | Notebook | Topics |
|---|----------|--------|
| 1 | [`ch01_sklearn_api.ipynb`](ch01_sklearn_api.ipynb) | Design philosophy, estimators, transformers, custom estimators, pipelines, common attributes/methods, grid & randomized search, tags and metadata routing, API best practices |
| 2 | [`ch02_preprocessing.ipynb`](ch02_preprocessing.ipynb) | Impact of raw data, missing-value imputation (simple, KNN), scaling, categorical encoding, `Pipeline`/`ColumnTransformer`, data leakage, feature engineering and selection (RFE, `SelectFromModel`), end-to-end exercise |
| 3 | [`ch03_dimensionality_reduction.ipynb`](ch03_dimensionality_reduction.ipynb) | Curse of dimensionality, PCA (variance explained, scree), LDA (supervised), t-SNE for visualisation, choosing a technique and trade-offs |
| 4 | [`ch04_knn_distance_metrics.ipynb`](ch04_knn_distance_metrics.ipynb) | Distance metrics (Euclidean, Manhattan, Chebyshev, Minkowski, cosine), KNN, effect of metric choice, tuning `k`/weights/metric, learning curves, confusion matrix, precision/recall/F1 |
| 5 | [`ch05_linear_models_regularization.ipynb`](ch05_linear_models_regularization.ipynb) | OLS, Ridge (L2), Lasso (L1), ElasticNet, bias–variance trade-off, coefficient paths, polynomial regression, splines (`SplineTransformer`), `RidgeCV`/`LassoCV`/`ElasticNetCV` |

## Chapter summaries

### Chapter 1 — Common Conventions and API Elements
scikit-learn is built around a small, consistent API: **estimators** (`fit`), **predictors** (`predict`), **transformers** (`transform`), and **meta-estimators** (pipelines, search). Learned attributes end with `_`, hyperparameters are set in the constructor, and everything composes through `Pipeline`. Custom components subclass `BaseEstimator` plus the relevant mixin. Hyperparameters are tuned with `GridSearchCV` / `RandomizedSearchCV`, and newer features (estimator tags, metadata routing) make behaviour explicit.

### Chapter 2 — Pre-Model Workflow and Data Preprocessing
Model quality is bounded by data quality. The notebook covers imputation (mean/median/most-frequent, `KNNImputer`), scaling (standard, min-max, robust), encoding (one-hot, ordinal; `LabelEncoder` is for targets only), `ColumnTransformer` for mixed data, and feature engineering/selection. The key principle is **avoid data leakage**: fit every preprocessing step on training data only, which `Pipeline` enforces automatically.
*Note:* California Housing needs internet; the notebook falls back to a synthetic regression dataset with injected missing values if the download fails.

### Chapter 3 — Dimensionality Reduction
**PCA** finds orthogonal directions of maximum variance (unsupervised). **LDA** finds directions that maximise class separation (supervised). **t-SNE** preserves local neighbourhoods for 2-D/3-D visualisation only (not for downstream modelling, and distances between clusters are not meaningful). Reducing dimensions trades some accuracy for speed and interpretability; the notebook measures this trade-off on real runs.

### Chapter 4 — Distance Metrics and Nearest Neighbors
KNN is a lazy, non-parametric learner that predicts from the `k` closest training points, so **scaling is essential**. On a structured (checkerboard) dataset, Chebyshev distance clearly underperformed (~0.66 vs ~0.80), while the other metrics were within ~0.005 of each other. Tuning covers `n_neighbors`, `weights`, and `metric`; evaluation uses learning curves, the confusion matrix, and precision/recall/F1.

### Chapter 5 — Linear Models and Regularization
Regularization penalises large coefficients to reduce variance: **Ridge** shrinks, **Lasso** shrinks and can zero coefficients (feature selection), **ElasticNet** blends both. In the book's experiment ElasticNet had the best R² (≈0.144 vs Ridge 0.089, Lasso 0.068, OLS 0.065), though margins are small. At `alpha=10` Lasso zeroed no coefficients because of the target scale. Non-linearity is handled with polynomial features (degree 5 overfit in this run) or splines. On a fresh dataset, `LassoCV` kept 18 and `ElasticNetCV` 25 of 30 coefficients at R² ≈ 0.981.

## Running the notebooks

**Google Colab:** upload a notebook (or open from GitHub) and run all cells.

**Locally:**
```bash
pip install scikit-learn numpy pandas matplotlib seaborn jupyter
jupyter lab
```
Tested with scikit-learn 1.9.1 and NumPy 2.5.3. Each notebook was executed end to end with no errors.

## Notes on differences from the book

- Notebook text is in English, matching the book.
- A few small slips in the book (version numbers, a typo, `LabelEncoder` used for features, `KNNImputer` described as majority vote) are corrected or flagged inline.
- Narrative claims were checked against the actual outputs; where results differ from the book's prose, the notebook states what was observed.
- Some older book code needs adjustment on recent scikit-learn (e.g. `multi_class` was removed from `LogisticRegression`; relevant from Chapter 6).

## Reference

*scikit-learn Cookbook, Third Edition* — over 80 recipes for machine learning in Python with scikit-learn. These notebooks are study material that summarize and re-implement the recipes; refer to the book for the original text.
