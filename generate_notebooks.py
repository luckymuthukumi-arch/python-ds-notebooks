import json
from pathlib import Path

ROOT = Path(__file__).parent

KERNEL = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
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
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": KERNEL,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = [
    (
        "01_python_fundamentals.ipynb",
        [
            md("""
# 01 — Python Fundamentals

## Learning goals
- Understand variables, basic data types, and expressions
- Practice input/output and type conversion
- Write simple control flow
"""),
            code("""
# Variables and basic types
name = "Kevin"
age = 28
height_m = 1.75
is_learning = True

print(name, age, height_m, is_learning)
print(type(name), type(age), type(height_m), type(is_learning))
"""),
            code("""
# Arithmetic and string formatting
x, y = 12, 5
print("Addition:", x + y)
print("Division:", x / y)
print("Floor division:", x // y)
print("Power:", x ** y)

print(f"{name} is {age} years old.")
"""),
            code("""
# Practice task:
# 1) Create variables for your favorite number and city
# 2) Print a sentence that combines both
"""),
        ],
    ),
    (
        "02_control_flow_and_functions.ipynb",
        [
            md("""
# 02 — Control Flow and Functions

## Learning goals
- Use if/elif/else for decisions
- Use loops for repetition
- Write reusable functions
"""),
            code("""
score = 83

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"

print("Grade:", grade)
"""),
            code("""
# Loops
nums = [2, 4, 6, 8]
total = 0
for n in nums:
    total += n

print("Total:", total)
"""),
            code("""
# Functions
def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

print(celsius_to_fahrenheit(25))
"""),
            code("""
# Practice task:
# Write a function called is_even(n) that returns True/False.
"""),
        ],
    ),
    (
        "03_collections_and_comprehensions.ipynb",
        [
            md("""
# 03 — Collections and Comprehensions

## Learning goals
- Work with lists, tuples, dictionaries, and sets
- Use list/dict comprehensions
"""),
            code("""
fruits = ["apple", "banana", "mango", "apple"]
fruit_set = set(fruits)
print("Unique fruits:", fruit_set)

prices = {"apple": 1.2, "banana": 0.8, "mango": 1.5}
print("Mango price:", prices["mango"])
"""),
            code("""
numbers = list(range(1, 11))
squares = [n**2 for n in numbers]
even_squares = [n**2 for n in numbers if n % 2 == 0]

print("Squares:", squares)
print("Even squares:", even_squares)
"""),
            code("""
# Practice task:
# Build a dictionary where keys are numbers 1-5 and values are cubes.
"""),
        ],
    ),
    (
        "04_files_exceptions_and_modules.ipynb",
        [
            md("""
# 04 — Files, Exceptions, and Modules

## Learning goals
- Read and write files
- Handle errors with try/except
- Import and use modules
"""),
            code("""
from pathlib import Path

sample = Path("sample_notes.txt")
sample.write_text("Python is practical.\\nData is powerful.\\n")

text = sample.read_text()
print(text)
"""),
            code("""
# Exception handling
raw = ["10", "20", "oops", "30"]
parsed = []

for item in raw:
    try:
        parsed.append(int(item))
    except ValueError:
        print(f"Skipping non-numeric value: {item}")

print("Parsed:", parsed)
"""),
            code("""
# Practice task:
# Create a function safe_divide(a, b) that handles division by zero.
"""),
        ],
    ),
    (
        "05_numpy_basics.ipynb",
        [
            md("""
# 05 — NumPy Basics

## Learning goals
- Create and manipulate arrays
- Perform vectorized math
- Understand axis-based operations
"""),
            code("""
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Mean:", arr.mean())
print("Std:", arr.std())
"""),
            code("""
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Matrix:\n", matrix)
print("Column sums:", matrix.sum(axis=0))
print("Row sums:", matrix.sum(axis=1))
"""),
            code("""
# Practice task:
# Generate 100 random numbers and compute min, max, mean.
"""),
        ],
    ),
    (
        "06_pandas_data_wrangling.ipynb",
        [
            md("""
# 06 — Pandas Data Wrangling

## Learning goals
- Create DataFrames
- Clean/filter/sort data
- Group and aggregate
"""),
            code("""
import pandas as pd

data = {
    "name": ["Amina", "Brian", "Chao", "Dina", "Eli"],
    "score": [88, 72, 95, 67, 81],
    "hours": [10, 7, 12, 5, 9],
}

df = pd.DataFrame(data)
df
"""),
            code("""
# Filter and derive columns
filtered = df[df["score"] >= 80].copy()
filtered["efficiency"] = filtered["score"] / filtered["hours"]
filtered.sort_values("efficiency", ascending=False)
"""),
            code("""
# Grouping example
bins = pd.cut(df["hours"], bins=[0, 6, 9, 12], labels=["low", "mid", "high"])
df.groupby(bins)["score"].mean()
"""),
            code("""
# Practice task:
# Add a pass/fail column where pass is score >= 75.
"""),
        ],
    ),
    (
        "07_data_visualization.ipynb",
        [
            md("""
# 07 — Data Visualization (Matplotlib + Seaborn)

## Learning goals
- Create line, bar, and histogram charts
- Use visualization for storytelling
"""),
            code("""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

sales = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "revenue": [1200, 1350, 1280, 1500, 1650, 1720],
})

plt.figure(figsize=(8, 4))
plt.plot(sales["month"], sales["revenue"], marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.show()
"""),
            code("""
students = pd.DataFrame({
    "group": ["A", "A", "B", "B", "C", "C"],
    "score": [70, 75, 80, 85, 90, 95]
})

plt.figure(figsize=(6, 4))
sns.boxplot(data=students, x="group", y="score")
plt.title("Score Distribution by Group")
plt.show()
"""),
            code("""
# Practice task:
# Create a histogram of the score column and explain what it shows.
"""),
        ],
    ),
    (
        "08_intro_machine_learning.ipynb",
        [
            md("""
# 08 — Intro to Machine Learning (Scikit-learn)

## Learning goals
- Understand train/test split
- Train a basic regression model
- Evaluate with MAE / R²
"""),
            code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Synthetic study-hours dataset
df = pd.DataFrame({
    "hours": [1,2,3,4,5,6,7,8,9,10],
    "score": [50,55,60,63,68,72,78,84,88,93]
})

X = df[["hours"]]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R²:", r2_score(y_test, pred))
"""),
            code("""
# Make a prediction
hours = [[7.5]]
predicted_score = model.predict(hours)[0]
print(f"Predicted score for 7.5 study hours: {predicted_score:.2f}")
"""),
            code("""
# Practice task:
# Try a different test_size and compare MAE/R².
"""),
        ],
    ),
]

for filename, cells in NOTEBOOKS:
    content = notebook(cells)
    (ROOT / filename).write_text(json.dumps(content, indent=2), encoding="utf-8")

readme = ROOT / "README.md"
readme.write_text(
    """# Python to Data Science Notebook Series

A progressive set of Jupyter notebooks to teach Python from fundamentals to introductory data science.

## Notebook Path
1. 01_python_fundamentals.ipynb
2. 02_control_flow_and_functions.ipynb
3. 03_collections_and_comprehensions.ipynb
4. 04_files_exceptions_and_modules.ipynb
5. 05_numpy_basics.ipynb
6. 06_pandas_data_wrangling.ipynb
7. 07_data_visualization.ipynb
8. 08_intro_machine_learning.ipynb

## Recommended Environment
```bash
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
jupyter notebook
```

## Teaching Notes
- Each notebook includes learning goals, runnable examples, and a practice task.
- Difficulty increases gradually for classroom progression.
""",
    encoding="utf-8",
)

print("Generated", len(NOTEBOOKS), "notebooks + README.md")
