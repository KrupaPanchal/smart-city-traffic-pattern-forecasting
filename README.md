# Smart City Traffic Pattern Forecasting

**Author:** Krupa Panchal  
**Domain:** Data Science / Machine Learning  
**Current stage:** Week 1 — Strategic Planning & Problem Definition

## Project overview
This project is planned to explore hourly traffic-volume data, identify patterns related to time and available weather variables, and establish a baseline model for estimating traffic volume. The analysis will be carried out in Python with a reproducible workflow.

> **Status note:** This repository currently contains the Week 1 planning materials and project scaffold. Dataset inspection, exploratory analysis, modeling, and evaluation have not yet been completed. Results will be added only after they have been produced and validated.

## Research question
How do time-related and available weather-related variables relate to observed hourly traffic volume, and how accurately can a baseline machine-learning model estimate traffic volume using those variables?

## Dataset candidate
- **Name:** Metro Interstate Traffic Volume
- **Source:** UCI Machine Learning Repository
- **Dataset page:** https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
- **Target candidate:** `traffic_volume`
- **Important limitation:** The data represents a particular section of Interstate 94 in the Minneapolis–Saint Paul area; findings should not automatically be generalized to every city.

The dataset will be obtained from the original source and its columns, time range, units, missing values, and licensing/usage terms will be checked before analysis. The raw dataset is not included in this repository.

## Planned workflow
1. Acquire the dataset and document its source.
2. Inspect columns, data types, row count, time coverage, missing values, duplicates, and unusual values.
3. Clean and validate the data; document each decision.
4. Conduct exploratory data analysis (EDA) with clearly labelled charts.
5. Prepare time-based features where appropriate.
6. Establish a simple baseline and compare it with suitable regression models.
7. Evaluate with MAE, RMSE, and R², using a validation strategy appropriate to the forecasting objective.
8. Summarize findings, limitations, and reproducible steps.

## Repository structure
```text
smart-city-traffic-pattern-forecasting/
├── data/
│   └── README.md
├── figures/
│   └── .gitkeep
├── notebooks/
│   └── README.md
├── reports/
│   └── Krupa_Panchal_Week_1_Strategic_Planning_Traffic_Forecasting.docx
├── src/
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting started
1. Create and activate a Python virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from the UCI page above and follow the data-source instructions in `data/README.md`.
4. Keep raw data out of Git unless the source terms and repository requirements clearly permit redistribution.

## Evaluation plan
Use a simple baseline before comparing machine-learning models. Candidate metrics are:
- **MAE:** average absolute prediction error.
- **RMSE:** gives larger errors more weight.
- **R²:** proportion of target variance explained by the model, interpreted in context.

A chronological split may be more appropriate than a random split if the objective is to estimate future traffic. The final split and feature set will be decided after the actual dataset is inspected.

## Planned deliverables
- Week 1 project planning report
- Dataset quality notes and EDA notebook
- Baseline model and evaluation summary
- Documented final conclusions and limitations

## Author
Krupa Panchal
