# 🏥 SDI Health Analyzer
African Healthcare Service Delivery & Clinical Competency Analysis
---
# 📌 Project Overview
This project analyzes Service Delivery Indicators (SDI) Health data across 10 African countries to evaluate:
 - Clinical competency of health workers
 - Health worker absenteeism
 - Maternal health preparedness
 - Rural vs Urban disparities
 - Country-level performance differences
 - Predictors of clinical competency.
The goal is to generate data-driven policy recommendations to improve healthcare quality and reduce preventable maternal mortality.
---
🌍 Dataset
**Source**: World Bank – Service Delivery Indicators (SDI)

**Facilities**: 5,223 health facilities

**Countries**: 10 African countries

**Variables**: 24 original features (26 after feature engineering)

Countries included:
 - Kenya (2012)
 - Nigeria (2013)
 - Tanzania (2014 & 2016)
 - Uganda (2013)
 - Mozambique (2014)
 - Niger (2015)
 - Madagascar (2016)
 - Togo (2013)
 - Senegal (2010)
---

| Metric                      | Value |
| --------------------------- | ----- |
| Average Clinical Competency | 22.4% |
| Average Absenteeism Rate    | 40.2% |
| Rural Facilities            | 75%   |
| Public Facilities           | 81%   |

---

<img width="5951" height="3595" alt="dashboard_02_country_comparison" src="https://github.com/user-attachments/assets/20d9af7a-8cf1-425b-b3dc-00ba1498264c" />
<img width="5960" height="4191" alt="dashboard_01_overview" src="https://github.com/user-attachments/assets/e449e197-1fc6-4174-ad8e-bfc6d9dc6ba5" />

---
# 🔍 Exploratory Data Analysis

The analysis included:
 - Missing value audit and imputation
 - Outlier detection (IQR method)
 - Country performance ranking
 - Rural vs Urban disparity analysis
 - Public vs Private comparison
 - Statistical testing (T-test for high vs low performers)
 - 12 comprehensive EDA visualizations

## 🏆 Country Performance Highlights
 - **Best Performing**: Kenya (38.5% competency)
 - **Lowest Performing**: Nigeria (22.1% competency)
 - **Competency gap**: 16+ percentage points
T-test confirmed statistically significant performance differences between high and low performing countries (p < 0.001).

---
# 🚨 Major Findings
## 1️⃣ Clinical Competency Crisis
Average competency across all facilities is only 22.4%.

## 2️⃣ Health Worker Absenteeism Emergency
Average absenteeism rate: 40.2%

##3️⃣ Maternal Health Emergency
| Disease               | Avg Competency |
| --------------------- | -------------- |
| Eclampsia             | ~5%            |
| Pregnancy Care        | ~6%            |
| Postpartum Hemorrhage | ~18%           |

---
#🦠 Disease-Specific Competency Ranking
Lowest to highest:
1. Eclampsia (5%)

2. Pregnancy care (6%)

3. PID (15%)

4. PPH (18%)

5. Diabetes (19%)

6. Malaria (23%)

7. TB (24%)

8. Pneumonia (25%)

9. Diarrhea (25%)

This indicates severe systemic gaps in maternal and emergency obstetric care.

---
#🔧 Data Preprocessing

- Median imputation for numerical variables
 - Mode imputation for categorical variables
 - Dropped 3 columns with 100% missing data
 - IQR-based outlier capping (6,989 outliers capped)
 - One-hot style encoding:
      1. *is_rural*
      2. *is_public*

 - Feature engineering:
      1. *rural_x_absence*
      2. *public_x_absence*

 - Created competency classification categories:
      1. *Low (47.7%)*
      2. *Medium (36.4%)*
      3. *High (15.9%)*

---
# 🤖 Predictive Modeling
## Target:
*avg_competency* (regression)

## Models Trained:
 - Random Forest
 - Gradient Boosting
 - Linear Regression

## Best Model:
Random Forest
| Metric  | Value |
| ------- | ----- |
| Test R² | 0.991 |
| RMSE    | 0.80  |
| MAE     | 0.42  |

Feature importance showed disease-specific competencies strongly explain overall competency.
Note: High R² reflects strong internal correlation between aggregate competency and disease-level competency components.

---
# 🏛 Policy Recommendations
## For Governments
 - Declare maternal health a national priority
 - Mandatory biannual competency assessments
 - Performance-linked incentives
 - Absenteeism accountability systems

## For Health Facilities
 - Emergency obstetric simulation training
 - Weekly case-based clinical review
 - Peer mentorship programs
 - Attendance monitoring systems

## For International Partners
 - Fund maternal health emergency training
 - Rural facility strengthening
 - Cross-country knowledge exchange

---
📈 Projected Impact (Scenario Modeling)
If implemented:
 - 5,223 facilities strengthened
 - ~26 million patients reached annually
 - 50% reduction in maternal mortality (modeled scenario)
 - Estimated ROI: 5–10x per dollar invested

--- 
# 🛠 Tech Stack

Python

Pandas

NumPy

Matplotlib / Seaborn

Scikit-learn

Statistical testing (SciPy)

--- 
🎯 Skills Demonstrated
 - Healthcare data analysis
 - Public health analytics
 - Data cleaning & preprocessing
 - Statistical hypothesis testing
 - Feature engineering
 - Machine learning modeling
 - Policy translation of analytics
 - Stakeholder-specific recommendations
 - Impact modeling

---
# 👤 Author

Francis Affoah
Clinical Data Science | Public Health Analytics | Healthcare Data Analyst















