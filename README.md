# Stock Portfolio Risk Intelligence

**Final Year Project** — B.E. / B.Tech in Computer Science / Information Technology

---

## Abstract

Stock Portfolio Risk Intelligence is a machine learning–based system that predicts **risk categories** (Low / Medium / High) for equities using fundamental, technical, and analyst-derived features. The system helps investors and portfolio managers visualize portfolio risk, compare stocks by sector, and get per-stock risk predictions with class probabilities. A Streamlit web application provides an interactive dashboard for portfolio overview and single-stock risk prediction.

**Keywords:** Stock risk prediction, Random Forest, portfolio analytics, fundamental analysis, technical indicators, Streamlit.

---

## Objectives

- To build a supervised ML model for classifying stocks into risk categories from financial and technical features.
- To design an interactive web dashboard for portfolio risk overview and single-stock risk prediction.
- To compute and visualize a composite RiskScore (Beta + volatility) and sector-wise risk metrics.
- To support reproducibility via a clear pipeline (data → features → model → UI).

---

## Features

- **Portfolio risk overview:** Table of companies with Sector, Risk Category, and Growth Category.
- **RiskScore:** Normalized composite score from Beta and 3M Volatility with scatter and sector bar charts.
- **Single-stock prediction:** Select a ticker and get predicted Risk Category with class probabilities (Random Forest).
- **Dataset:** Multi-stock dataset with fundamentals (P/E, ROE, Debt/Equity, etc.), technicals (RSI, MACD, volatility), and analyst ratings.

---

## Tech Stack

| Component        | Technology        |
|-----------------|-------------------|
| Language        | Python 3.8+       |
| ML Framework    | scikit-learn      |
| Model           | Random Forest (stratified, class_weight='balanced') |
| Web UI          | Streamlit         |
| Data            | Pandas, NumPy     |
| Notebook        | Jupyter (EDA & experimentation) |

---

## Prerequisites (Basics)

Before cloning or running this project, ensure you have:

| Prerequisite | Purpose |
|--------------|---------|
| **Python 3.8+** | Required to run the app and notebooks |
| **pip** | To install dependencies from `requirements.txt` |
| **Terminal / Command line** | To run commands (e.g. `pip install`, `streamlit run`) |
| **Web browser** | To use the Streamlit app (Chrome, Firefox, Edge, etc.) |
| **Git** (optional) | Only if you clone the repo from GitHub |

You do not need prior experience in machine learning or finance; the app and this README explain usage.

---

## Project Structure

```
ml/
├── app.py                 # Streamlit application (run this)
├── stockprofil.ipynb      # EDA, feature engineering, model experimentation
├── data/
│   ├── Stock_Dataset.csv      # Main stock dataset
│   ├── TCS_Price_History.csv  # TCS historical prices (sample)
│   ├── TCS_Summary.csv        # TCS summary stats
│   ├── Summary_Stats.csv      # Dataset summary statistics
│   └── Feature_Guide.csv      # Feature definitions
├── docs/
│   └── PROJECT_REPORT.md      # Project report outline / draft
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup & Run

### 1. Clone or download the project

If you have the repo on GitHub:

```bash
git clone https://github.com/YOUR_USERNAME/ml.git
cd ml
```

Or download the project as a ZIP and extract it, then:

```bash
cd path/to/ml
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

From the project root (where `app.py` and `data/` are):

```bash
streamlit run app.py
```

The app will open in the browser (default: http://localhost:8501).

### 5. Run the notebook (optional)

Open `stockprofil.ipynb` in Jupyter or VS Code to reproduce EDA and model experiments.

---

## Dataset Description

- **Stock_Dataset.csv:** Companies with features such as Market Cap, P/E, P/B, ROE, Debt/Equity, RSI, MACD, 3M Volatility, Analyst Buy/Hold/Sell, and target **Risk Category** (Low/Medium/High).
- **Feature_Guide.csv:** Short description and type of each feature.
- **TCS_Price_History.csv / TCS_Summary.csv:** Example time-series and summary for TCS (optional for report/case study).

---

## Results (Summary)

- **Target:** Risk Category (Low / Medium / High).
- **Model:** Random Forest Classifier (300 trees, stratified split, balanced class weights).
- **Metric:** Test set accuracy is displayed in the app sidebar (run the app to see the current value).
- **UI:** Portfolio table, RiskScore scatter chart, sector-wise average RiskScore bar chart, and single-stock prediction with probabilities.

---

## Future Scope

- Add more stocks and longer history; integrate live or periodic data feeds.
- Extend to **Growth Category** or **Buy/Hold/Sell** prediction using the same or additional targets.
- Hyperparameter tuning (e.g. GridSearchCV) and cross-validation reporting.
- User upload of custom portfolio CSV and risk report export (PDF/Excel).
- Optional: simple authentication and saved portfolios.

---

## Author / Team

- **[Your Name]** — Final Year, [Your College], [Branch]
- Guide: [Guide Name], [Department]

---

## References

1. Scikit-learn: Machine Learning in Python — https://scikit-learn.org/
2. Streamlit Documentation — https://docs.streamlit.io/
3. Pandas Documentation — https://pandas.pydata.org/docs/
4. (Add papers/books on stock risk, portfolio theory, or ML in finance as per your report.)

---

## Posting this project to GitHub

1. Create a new repository on GitHub (do not add a README or .gitignore if you already have them).
2. In your project folder (`ml`), run:

```bash
git init
git add .
git commit -m "Initial commit: Stock Portfolio Risk Intelligence"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your GitHub username and repository name.

---

## License

This project is submitted as a final year academic project. Use for learning and evaluation only.
