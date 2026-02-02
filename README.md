\# Cross-Border Fraud Risk Decision System



This repository contains a complete end-to-end cross-border fraud risk decision system built using synthetic data, rule-based signals, machine learning risk scores, customer trust scores, and multi-month monitoring.



The purpose of this project is to demonstrate how real-world fraud systems make decisions, not just how models are trained.



---



\## Project Overview



Cross-border payment systems must balance fraud prevention with customer experience. Not every risky transaction should be blocked. Some should be reviewed, and trusted customers should be allowed to transact with minimal friction.



This project simulates how a fraud risk team designs, monitors, and tunes a transaction decision system over time.



Key goals:

\- Detect high-risk transactions

\- Reduce false positives using customer trust

\- Monitor system behavior across months

\- Allow policy tuning without retraining models



---



\## Scope



\- Transaction type: Outbound cross-border transactions

\- Source country: India

\- Destination countries: Multiple international corridors

\- Data: Fully synthetic (no real customer data)

\- Time period: Multi-month (Jan to Mar)



---



\## System Architecture



The system is divided into three main stages.



1\) Synthetic Data Generation  

Customer profiles and transactions are generated with realistic behavior patterns such as onboarding date, transaction frequency, device change, and corridor risk.



Notebook:

notebooks/01\_synthetic\_data.ipynb



2\) Risk Scoring and Decision Engine  

Each transaction is evaluated using:

\- Rule-based risk signals

\- Machine learning risk score

\- Customer trust score



Final decisions are assigned as:

\- ALLOW

\- REVIEW

\- BLOCK



Notebook:

notebooks/02\_risk\_scoring.ipynb



3\) Multi-Month Monitoring  

The system is analyzed over multiple months to study:

\- Decision distribution stability

\- ML risk score drift

\- Trust score evolution

\- Corridor concentration risk



Notebook:

notebooks/03\_multi\_month\_analysis.ipynb



---



\## Streamlit Risk Decision Tool



A Streamlit application is included to simulate how fraud teams interact with the system.



The app allows:

\- High-level system overview

\- Decision policy simulation

\- Risk distribution analysis

\- Transaction-level exploration



File:

streamlit\_app.py



---



\## Folder Structure



cross-border-shortlist/

├── notebooks/

│   ├── 01\_synthetic\_data.ipynb

│   ├── 02\_risk\_scoring.ipynb

│   └── 03\_multi\_month\_analysis.ipynb

├── OUTPUTS/

│   └── RISK\_SCORE\_TXNS/

│       ├── risk\_scored\_transactions\_2025\_01.csv

│       ├── risk\_scored\_transactions\_2025\_02.csv

│       └── risk\_scored\_transactions\_2025\_03.csv

├── streamlit\_app.py

├── requirements.txt

└── README.md



---



\## How to Run the Project



Install dependencies:



pip install -r requirements.txt



Run the Streamlit app:



streamlit run streamlit\_app.py



---



\## Key Learnings



\- Fraud systems are decision systems, not just ML models

\- Trust scores help control false positives

\- Policy thresholds are as important as model accuracy

\- Monitoring over time is critical in production environments



---



\## Disclaimer



This project uses only synthetic data.

No real customer or transaction data is included.



