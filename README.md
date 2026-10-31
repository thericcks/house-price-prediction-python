# House Price Prediction Using Python

A beginner-friendly Python project that explores a residential housing dataset and compares two Linear Regression models for estimating house prices.

## 📄 Project Documentation

For a detailed explanation of the project's methodology, 
data analysis, Python implementation, visualizations, and 
machine learning results, refer to the project thesis below.

📘 [Read the Project Thesis](https://drive.google.com/file/d/1oHKGsttYEevvK6-i8EeNrBnKddLkvJZM/view?usp=sharing)

## Project overview

This project uses exploratory data analysis (EDA), data visualisation, correlation analysis, and supervised machine learning to examine housing attributes and predict the `price` target variable.

The analysis was performed in a Jupyter Notebook using a dataset of **545 records and 13 columns**: one target (`price`) and 12 predictor features. The supplied dataset contains no recorded missing values.

## Objectives

- Inspect and understand the housing dataset.
- Summarise house prices using descriptive statistics.
- Visualise price distributions, potential outliers, and relationships between features.
- Compare price distributions across numerical, binary, and categorical features.
- Examine Pearson correlations between price and numerical/binary features.
- Train and compare Simple Linear Regression and Multiple Linear Regression models.
- Evaluate predictions with MAE, RMSE, and R², then inspect actual-versus-predicted values and residuals.

## Dataset

Place the original CSV file named **`Housing.csv`** in the project root, next to the notebook. The notebook reads it with:

```python
df = pd.read_csv("Housing.csv")
```

The dataset contains the following columns:

| Role | Features |
|---|---|
| Target | `price` |
| Numerical predictors | `area`, `bedrooms`, `bathrooms`, `stories`, `parking` |
| Yes/No predictors | `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` |
| Categorical predictor | `furnishingstatus` |

The six Yes/No features are mapped to numeric indicators (`yes` = 1 and `no` = 0). `furnishingstatus` has three categories (`furnished`, `semi-furnished`, and `unfurnished`) and is one-hot encoded inside the Model B preprocessing pipeline.

**Dataset note:** The source, geographical coverage, collection date, price currency, and redistribution licence were not documented in the supplied project materials. Confirm the dataset's original attribution and permission before redistributing `Housing.csv` publicly.

## Tools and libraries

- **Python** — programming language
- **Jupyter Notebook** — interactive development environment
- **Pandas** — data loading, inspection, filtering, and summaries
- **NumPy** — numerical computing library (imported in the notebook)
- **Matplotlib** — histograms, boxplots, scatter plots, and residual visualisation
- **Seaborn** — correlation heatmap
- **Scikit-learn** — preprocessing, train-test splitting, Linear Regression, and evaluation metrics

## Analysis workflow

1. Load `Housing.csv` into a Pandas DataFrame.
2. Inspect dimensions, columns, data types, missing values, and descriptive statistics.
3. Explore price distribution and feature relationships with plots.
4. Map the six Yes/No features to 0/1 and examine a Pearson correlation matrix. `furnishingstatus` is excluded from this correlation matrix because it is still categorical text at that stage.
5. Split the data into training and testing subsets (80:20, `random_state=42`).
6. Train two regression models and compare their test-set predictions.
7. Review actual-versus-predicted and residual plots for Model B.

## Models and results

Both models use the same 80:20 train-test split with `random_state=42`.

- **Model A — Simple Linear Regression:** uses `area` as the only predictor.
- **Model B — Multiple Linear Regression:** uses the remaining housing features. A Scikit-learn `ColumnTransformer` and `Pipeline` apply `OneHotEncoder` to `furnishingstatus` while passing the other prepared features through to the regressor.

Recorded test-set results:

| Metric | Model A: Area only | Model B: Multiple features |
|---|---:|---:|
| MAE | 1,474,748.13 | 970,043.40 |
| RMSE | 1,917,103.70 | 1,324,506.96 |
| R² | 0.2729 | 0.6529 |

On this particular test split, Model B reduced MAE by approximately 34.2% and RMSE by approximately 30.9% compared with Model A. Its R² was also higher. These results indicate better performance on the recorded split; they do not guarantee performance on other samples or establish causation.

**Price values are reported in dataset units.** The currency is not specified in the supplied project files. R² is a measure of explained variation relative to a test-set mean baseline, not a prediction-accuracy percentage.

## How to run the project

1. Clone or download this repository.
2. Ensure `Housing.csv` and the notebook are in the same folder.
3. (Recommended) Create and activate a Python virtual environment.
4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `house_price_prediction.ipynb` and run the cells from top to bottom.

The notebook expects the CSV filename to be exactly `Housing.csv` (including capitalisation).

## Repository structure

```text
house-price-prediction-python/
├── README.md
├── requirements.txt
├── .gitignore
├── Housing.csv
└── house_price_prediction.ipynb
```

## Limitations

- The dataset contains 545 records and a limited set of predictors.
- No house-age or construction-year feature is available.
- Some categories contain very few observations, so group comparisons for those categories should be interpreted cautiously.
- The reported evaluation uses one 80:20 train-test split; cross-validation and hyperparameter tuning were not recorded.
- Some individual predictions have large errors, including for certain higher-priced houses.
- The worst-predicted house categories were not investigated in detail.
- The model is an educational estimation exercise, not a professional property valuation service.

## Future improvements

Possible future work includes cross-validation, testing other regression algorithms, more detailed residual/error analysis, and creating a separate web interface. These improvements are not part of the current notebook implementation.

## License

No licence is currently specified. Add a licence only after deciding how the code may be reused and confirming the dataset's redistribution terms. A repository licence for your code does not automatically grant rights to redistribute the dataset.
