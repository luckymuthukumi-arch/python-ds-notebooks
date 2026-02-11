import json
from pathlib import Path

ROOT = Path(__file__).parent

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "name": "python",
        "version": "3.x",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    },
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells):
    return {"cells": cells, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}


extra = [
    (
        "09_exercise_solutions_foundations.ipynb",
        [
            md("""
# 09 — Exercise Solutions (Foundations)

Solutions for practice tasks from notebooks 01–04.
"""),
            md("## Solution: Favorite number and city"),
            code("""
favorite_number = 7
city = "Nairobi"
print(f"My favorite number is {favorite_number} and my city is {city}.")
"""),
            md("## Solution: is_even(n)"),
            code("""
def is_even(n):
    return n % 2 == 0

for value in [1, 2, 15, 20]:
    print(value, is_even(value))
"""),
            md("## Solution: Number → cube dictionary"),
            code("""
cubes = {n: n**3 for n in range(1, 6)}
cubes
"""),
            md("## Solution: safe_divide(a, b)"),
            code("""
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero"

print(safe_divide(10, 2))
print(safe_divide(10, 0))
"""),
        ],
    ),
    (
        "10_exercise_solutions_data_science.ipynb",
        [
            md("""
# 10 — Exercise Solutions (Data Science)

Solutions for practice tasks from notebooks 05–08.
"""),
            md("## Solution: 100 random numbers summary"),
            code("""
import numpy as np

rng = np.random.default_rng(42)
nums = rng.normal(loc=50, scale=10, size=100)
print("Min:", nums.min())
print("Max:", nums.max())
print("Mean:", nums.mean())
"""),
            md("## Solution: Pass/Fail column in pandas"),
            code("""
import pandas as pd

df = pd.DataFrame({
    "name": ["Amina", "Brian", "Chao", "Dina", "Eli"],
    "score": [88, 72, 95, 67, 81],
})

df["result"] = df["score"].apply(lambda s: "Pass" if s >= 75 else "Fail")
df
"""),
            md("## Solution: Histogram of scores"),
            code("""
import matplotlib.pyplot as plt

scores = [70, 75, 80, 85, 90, 95]
plt.hist(scores, bins=5, edgecolor="black")
plt.title("Histogram of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

print("Interpretation: scores are skewed toward higher values in this tiny sample.")
"""),
            md("## Solution: Compare test sizes in ML"),
            code("""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

df = pd.DataFrame({
    "hours": [1,2,3,4,5,6,7,8,9,10],
    "score": [50,55,60,63,68,72,78,84,88,93]
})

for ts in [0.2, 0.3, 0.4]:
    X_train, X_test, y_train, y_test = train_test_split(df[["hours"]], df["score"], test_size=ts, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"test_size={ts}: MAE={mean_absolute_error(y_test, pred):.2f}, R2={r2_score(y_test, pred):.3f}")
"""),
        ],
    ),
    (
        "11_mini_project_exploratory_analysis.ipynb",
        [
            md("""
# 11 — Mini Project: Exploratory Data Analysis

## Project brief
Analyze a synthetic retail dataset and answer business questions.
"""),
            code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(7)

n = 200
categories = ["Electronics", "Clothing", "Home", "Sports"]

df = pd.DataFrame({
    "category": rng.choice(categories, size=n, p=[0.3, 0.25, 0.25, 0.2]),
    "price": rng.normal(80, 25, size=n).clip(5, 300),
    "quantity": rng.integers(1, 6, size=n),
})
df["revenue"] = df["price"] * df["quantity"]
df.head()
"""),
            code("""
# Questions:
# 1) Which category generates highest total revenue?
# 2) What is average order value by category?
# 3) Are there outliers in price?

summary = df.groupby("category").agg(
    total_revenue=("revenue", "sum"),
    avg_order_value=("revenue", "mean"),
    avg_price=("price", "mean"),
).sort_values("total_revenue", ascending=False)

summary
"""),
            code("""
sns.boxplot(data=df, x="category", y="price")
plt.title("Price Distribution by Category")
plt.xticks(rotation=15)
plt.show()
"""),
            md("""
## Deliverable
Write a short 3–5 bullet summary of findings and one recommendation.
"""),
        ],
    ),
    (
        "12_mini_project_prediction_pipeline.ipynb",
        [
            md("""
# 12 — Mini Project: Prediction Pipeline

## Project brief
Build and evaluate a model to predict house prices from synthetic features.
"""),
            code("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng(123)
n = 500

df = pd.DataFrame({
    "sqft": rng.normal(1600, 500, n).clip(400, 5000),
    "bedrooms": rng.integers(1, 6, n),
    "age": rng.integers(0, 60, n),
    "neighborhood": rng.choice(["A", "B", "C", "D"], n),
})

base = 50000 + df["sqft"] * 180 + df["bedrooms"] * 10000 - df["age"] * 800
neigh_boost = df["neighborhood"].map({"A": 90000, "B": 50000, "C": 20000, "D": 0})
noise = rng.normal(0, 25000, n)

df["price"] = base + neigh_boost + noise

df.head()
"""),
            code("""
X = df.drop(columns=["price"])
y = df["price"]

num_cols = ["sqft", "bedrooms", "age"]
cat_cols = ["neighborhood"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols),
])

model = RandomForestRegressor(n_estimators=200, random_state=42)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", model),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("R2:", round(r2_score(y_test, pred), 3))
"""),
            md("""
## Extension ideas
- Try GradientBoostingRegressor or XGBoost
- Add cross-validation
- Inspect feature importances
"""),
        ],
    ),
]

for filename, cells in extra:
    (ROOT / filename).write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")

readme = ROOT / "README.md"
content = readme.read_text(encoding="utf-8")
append = """

## Added Tracks

### Exercise Solutions
9. 09_exercise_solutions_foundations.ipynb
10. 10_exercise_solutions_data_science.ipynb

### Mini Projects
11. 11_mini_project_exploratory_analysis.ipynb
12. 12_mini_project_prediction_pipeline.ipynb
"""
if "## Added Tracks" not in content:
    readme.write_text(content + append, encoding="utf-8")

print("Added", len(extra), "notebooks")
