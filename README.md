# WDI CO2 Efficiency Dashboard

An interactive Streamlit dashboard for analyzing CO2 emissions efficiency across countries using World Development Indicators data.

## Live Demo

[Open the dashboard](https://wdi-co2-efficiency-dashboard-javpthbtu9xmq2rkzgayxx.streamlit.app/)

## Project Overview

This project evaluates whether countries emit more or less CO2 than expected based on:

- GDP per capita
- Population
- Year
- Income group
- Region

A Random Forest model is used to estimate expected emissions, and the difference between actual and predicted emissions is used as a residual-based efficiency measure.

## Features

- Global country-level CO2 efficiency map
- GDP vs CO2 scatter plot
- Top 10 best and worst performers
- Detailed country table
- Country-level time trend comparison
- Interactive filters by year, region, and income group

## Project Files

- `wdi_co2_dashboard_app.py` → Streamlit dashboard
- `WDI_dashboard_project.ipynb` → analysis notebook
- `dashboard_data.parquet` → prepared dashboard dataset
- `requirements.txt` → required dependencies

## Project Report

Full project report is available here:

[View Report (PDF)](./AmirhosseinLN_CO2analyse.pdf)

## Data Source

World Development Indicators dataset from the World Bank:

https://www.kaggle.com/datasets/theworldbank/world-development-indicators

Note: Raw datasets are not included in the repository due to file size limitations.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

## Key Insights

- Income level is a strong predictor of emissions behavior
- Some high-income countries are significantly more efficient than expected
- Certain resource-dependent economies exhibit persistent over-emission
- Efficiency varies not only across countries but also over time

## Author

Amirhossein Latifi Navid  
LinkedIn: https://www.linkedin.com/in/amirhossein-latifinavid-5923272a7
