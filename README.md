# **F1 2025 Championship Predictor & Data Analysis**

<!-- Hero Image -->
![F1 Performance Analysis](https://images.pexels.com/photos/29406740/pexels-photo-29406740.jpeg)

Welcome to the **F1 2025 Championship Predictor**! This project is a data analysis focused on the 2025 Formula 1 season. The goal is not just to understand what happened, but to predict what could happen using the official **FastF1** library.

All the work lives inside a single Jupyter Notebook, designed to act as the complete brain of the project: data extraction from the Formula 1 API, cleaning, feature engineering, machine learning training, and championship simulation.

---

## **Table of Contents**

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Extraction (FastF1)](#data-extraction-fastf1)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Championship Simulation](#championship-simulation)
- [Installation](#installation)
- [Running the Project](#running-the-project)

---

## **Overview**

The system uses the Python `fastf1` library to pull data directly from Formula 1's servers — race results, points, grid positions, and the full 2025 calendar. From there, the notebook trains a model to estimate driver performance and projects the championship outcome.

## **Architecture**
- **Jupyter Notebook (`notebooks/f1_2025_predictor.ipynb`)**: All the logic lives here.
- **Local Cache (`data/cache/`)**: Stores FastF1 responses so subsequent runs require no network access.
- **Fallback Data (`data/historical/`)**: Historical CSV files used as a backup if the API is unavailable.

## **Data Extraction (FastF1)**

`fastf1` is the industry standard for F1 telemetry and timing analysis in Python. It provides access to:
* `get_event_schedule(2025)`: The official season calendar.
* `get_session()`: Precise race session data (official results, lap times).

## **Machine Learning Methodology**

The problem is framed as **finish position regression** — rather than a binary "who wins" classification, we use a **Random Forest Regressor** (also compared against XGBoost) to predict each driver's exact finishing position, based on their historical rolling form and starting grid position.

A **temporal split** is used for validation: earlier rounds train the model, later rounds test it. This mirrors real-world prediction and prevents data leakage.

## **Championship Simulation**

The model provides each driver's expected base performance. To capture the natural unpredictability of F1 (engine failures, collisions, strategy calls), the notebook runs thousands of Abu Dhabi simulations with probabilistic noise around the model's predictions. The result is a probability distribution showing each driver's realistic chance of becoming World Champion.

---

## **Installation**

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/priscillalea/f1-winner-predictor.git
cd f1-winner-predictor
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install fastf1 pandas scikit-learn xgboost plotly matplotlib seaborn jupyter
```

---

## **Running the Project**

1. Start Jupyter Notebook from the project root:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/f1_2025_predictor.ipynb`.
3. Run all cells from top to bottom.
4. On the first run, FastF1 will download the race data into `data/cache/` — this may take a moment. All subsequent runs will be near-instant.
