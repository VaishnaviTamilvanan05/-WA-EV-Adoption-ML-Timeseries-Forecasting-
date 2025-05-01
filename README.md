# ⚡ Washington State EV Forecasting & Charging Infrastructure Planning (2025–2028)

This project delivers a data-driven strategy to forecast **electric vehicle (EV) adoption trends** and predict the **charging infrastructure needed** across Washington State through 2028. Using advanced **time series models** (Prophet, SARIMA, ARIMA) and **machine learning**, it provides actionable insights to help planners and policymakers address EV infrastructure deficits and plan equitable charger deployment.

---

## 🎯 Key Objectives

- Forecast monthly EV registrations from 2010–2028 at **state and county levels**
- Integrate **average EV sale price** as an external regressor to improve forecast realism
- Identify **infrastructure gaps** using **DOE/NREL benchmarks**
- Predict **annual station needs** by county (2025–2028)
- Allocate stations at the **ZIP-code level** based on demand/supply ratios
- Provide insights through an interactive **Power BI dashboard**

---

## 🔁 Process Implemented

1. **Data Collection**
   - EV registration data from Washington Open Data Portal (2010–2024)
   - Charging station data from OpenChargeMap

2. **Data Preprocessing**
   - Cleaning, feature selection, time indexing, and handling outliers
   - County and ZIP-level aggregation for EV trends and infrastructure

3. **Exploratory Data Analysis (EDA)**
   - Trend visualization, correlation analysis, and hypothesis testing

4. **Forecast Modeling**
   - ARIMA/SARIMA for baseline trend and seasonality modeling
   - Prophet for flexible forecasting, with and without external regressors
   - Fine-tuning via hyperparameter grid search and walk-forward validation

5. **County-Level Forecasting**
   - Individual models trained for top 10 counties using historical patterns and sale price influence

6. **Charging Infrastructure Gap Analysis**
   - EV-to-station ratios and comparison with DOE/NREL policy thresholds

7. **Machine Learning for Station Prediction**
   - Linear regression model trained on historical EV/station data
   - Station forecasts generated for 2025–2028 per county

8. **ZIP-Level Prioritization**
   - Normalized score based on EV demand vs. current supply
   - Proportional station allocation to underserved ZIPs

9. **Dashboard & Output Generation**
   - Interactive Power BI dashboard + downloadable CSV reports
   - Visuals, summaries, and maps per county and statewide

---

## 📊 Tools & Technologies

- **Python**: Prophet, statsmodels, scikit-learn, pandas, matplotlib, seaborn  
- **Power BI**: Dynamic dashboard for regional insights  
- **Data Sources**:  
  - [Washington EV Registration Data](https://data.wa.gov)  
  - [OpenChargeMap](https://openchargemap.org)

---

## 🧠 Models Implemented

- **Prophet (with & without regressors)** for adaptive time series modeling
- **SARIMA/ARIMA** for capturing trend + seasonality baselines
- **Linear Regression** to estimate station additions from EV growth
- **Walk-forward cross-validation** for model robustness

---

## 📦 Outputs

- County-level EV forecasts (2025–2028)
- Predicted station needs and gaps based on policy benchmarks
- ZIP-level priority station allocation strategy
- Downloadable CSVs with clean and forecasted data
- Summary reports and Power BI dashboard

---

## 📈 Power BI Dashboard Highlights

- Interactive county maps and slicers for each forecast year
- EV-to-station ratio insights and visual growth comparisons
- Dynamic bar charts, summary boxes, and decision-ready visuals

---

## 📚 References

- Nicholas et al. (2019), ICCT — *EV Charging Infrastructure Gap*
- Browne (2024), GWU — *Equity in EV Infrastructure*
- Neubauer & Wood (2014) — *Range Anxiety & EV Utility*
- Meta Forecasting Team — *[Prophet](https://facebook.github.io/prophet/)*
- [OpenChargeMap](https://openchargemap.org)
- [Washington State EV Dataset](https://data.wa.gov)

