# **F1 2025 Performance Analysis**

<!-- Hero Image / Capa -->
![F1 Performance Analysis](https://images.pexels.com/photos/29406740/pexels-photo-29406740.jpeg)

A data analysis of Formula 1 in the year 2025, focusing on pit stops, qualifying, and race results. This project visualizes patterns, highlights top teams and drivers, and prepares the foundation for a future machine learning prediction of race winners. This project will be updated in the near future.

---

## **Table of Contents**

- [Motivation & Story](#motivation--story)
- [What This Project Does](#what-this-project-does)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data Source](#data-source)
- [Results & Visualizations](#results--visualizations)
- [Roadmap](#roadmap)
- [How This Fits My Learning Path](#how-this-fits-my-learning-path)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## **Motivation & Story**

I’m a passionate F1 fan and a Data Science student. I love analyzing data, identifying patterns, and telling stories through numbers. Watching the 2025 season, I wondered:

- Who is the fastest team in pit stops?
- Who excels in qualifying sessions?
- Who is actually winning races, especially in a competitive year with drivers like Oscar Piastri and Lando Norris?

As a Red Bull Racing fan, I was shocked to see Max Verstappen only won 2 races so far 😅. But this analysis helped me organize thoughts and explore trends in a structured way, reinforcing the belief I have in data as a fundamental aspect in everyday life and projects.

This is my third project, after a **K-Drama Recommender** and a **Spam Detector**.

---

## **What This Project Does**

- Collects and cleans F1 2025 data (manually scraped from [Formula1.com](https://www.formula1.com/en/results/2025/races)).
- Normalizes team names and colors for consistent visualization.
- Computes and visualizes:
  - Average pit stop times per team
  - Pole positions per driver
  - Race wins per driver
- Generates clear, horizontal bar charts for each metric.
- Prepares the groundwork for future ML models to predict race winners.

---

## **Project Structure**

f1-winner-predictor/
├─ f1.ipynb # Main analysis notebook
├─ csv/
│ ├─ pitstop.csv
│ ├─ qualifying.csv
│ └─ raceresult.csv
├─ plots/ # Saved PNG and SVG charts
└─ README.md

yaml
Copy code

**Tech Stack:** Python, Pandas, NumPy, Matplotlib, IPython display.

---

## **Getting Started**

### **Prerequisites**

- Python 3.10+
- pip
- Jupyter Notebook or Google Colab (recommended)

### **Installation**

Clone the repository:

```bash
git clone https://github.com/priscillalea/f1-winner-predictor.git
cd f1-winner-predictor
Create and activate a virtual environment (optional but recommended):

bash
Copy code
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
Install dependencies:

bash
Copy code
pip install pandas matplotlib numpy
Ensure CSV files are in place:

bash
Copy code
csv/pitstop.csv
csv/qualifying.csv
csv/raceresult.csv
Run the Notebook
Jupyter Notebook:

bash
Copy code
jupyter notebook f1.ipynb
Open the notebook in your browser and execute cells sequentially to generate the analysis and plots.

Google Colab:

Open f1.ipynb in Colab.

Upload CSV files to the environment or link from Google Drive.

Run cells as usual.

Notes:

Charts will be saved in the plots/ folder in both PNG and SVG formats.

You can modify team colors, add new metrics, or update data directly in the notebook.

Future updates will include ML-based predictions for race winners.

Usage
Visualize pit stop performance, qualifying results, and race wins.

Customize team colors or add additional metrics by modifying the notebook.

Prepare for future ML-based winner predictions using historical data.

Data Source
All data collected manually from the official Formula 1 website: Formula 1 Results 2025

Currently, the code includes only pit stop, qualifying, and race day results. Other data saved in this repository will be used in future updates for ML predictions.

Results & Visualizations
Charts generated:

Average Pit Stop Time per Team

Number of Pole Positions per Driver

Number of Race Wins per Driver

Example visualizations will be displayed inline in the notebook.

Roadmap
Add machine learning models to predict next GP winners.

Include more datasets: lap times, weather conditions, tire strategies, etc.

Improve visualizations with interactive plots (Plotly / Altair).

Automate data scraping for future seasons.

How This Fits My Learning Path
Weekend ML Projects:

K-Drama Recommender → TF-IDF & similarity, HTML/CSS frontend.

Spam Detector → NLP pipeline with Multinomial Naive Bayes.

F1 Performance Analysis → Tabular data, metrics, visualization, story-telling.

This project allows me to combine passion (F1) with technical learning (Python, data analysis, visualization).

Contributing
Fork the repository

Create a feature branch: git checkout -b feat/my-feature

Commit changes with clear messages

Open a pull request

License
MIT License — see LICENSE for details.

Acknowledgments
Official Formula 1 website for data

Python, Pandas, Matplotlib communities for guidance

My own weekend learning journey ❤️
