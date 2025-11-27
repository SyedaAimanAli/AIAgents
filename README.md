# AIAgents — Multi-Agent Data Analysis System

AIAgents is a modular, asynchronous data-analysis pipeline built using Python.  
It processes CSV datasets through multiple “agents”, each responsible for a specific analytical task—cleaning, EDA, anomaly detection, ML modeling, business insights, and report generation.

The system can run standalone via `main.py` or through a clean Flask-based UI.

---

## Key Features

- **Multi-Agent Architecture**  
  Each step of the pipeline is handled by a dedicated agent:
  - Data Cleaning Agent  
  - EDA (Exploratory Data Analysis) Agent  
  - Anomaly Detection Agent  
  - Machine Learning Agent  
  - Business Insights Agent  
  - Report Generation Agent  

- **Automatic PDF Reporting**  
  Generates a neatly formatted PDF summarizing:
  - Insights  
  - Visualizations  
  - Key findings  
  - Model analysis  
  - Anomaly summaries  

- **Optional AI (Google ADK/Gemini) Enhancements**
  - Executive summary generation  
  - Data-cleaning strategy suggestions  
  - Insight/interpretation generation  

- **Flask Web UI**  
  Allows uploading a CSV file → running pipeline → downloading report.

- **Sample Dataset Generator**  
  Useful for testing the pipeline instantly without uploading anything.

---

## Project Structure

```

AIAgents/
│
├── project/
│   ├── agents/
│   │   ├── anomaly.py
│   │   ├── data_cleaning.py
│   │   ├── eda_agent.py
│   │   ├── insights.py
│   │   ├── ml_model.py
│   │   └── report.py
│   │
│   ├── dataset.py
│   └── main.py              # Orchestrates the entire pipeline
│
├── webapp/
│   ├── app.py               # Flask server
│   ├── templates/
│   │   ├── index.html
│   │   └── results.html
│   └── static/              # CSS/JS (optional)
│
├── README.md
└── uv.lock / pyproject.toml  # Dependencies

````

---

## How to Run (Local Machine)

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
````

### **2. (Optional) Add ADK API Key**

```bash
export GOOGLE_ADK_API_KEY="your_api_key"
```

### **3. Run Flask UI**

```bash
python webapp/app.py
```

Then open in your browser:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Agents Overview

### **Data Cleaning Agent**

Cleans missing values, fixes types, removes duplicates, prepares dataset.

### **EDA Agent**

Generates visualizations and dataset summaries.

### **Anomaly Detection Agent**

Identifies numeric outliers using IQR-based thresholds.

### **Machine Learning Agent**

Builds a simple model (regression or classification) and outputs feature importance.

### **Business Insights Agent**

Creates executive summary + actionable insights (ADK optional).

### **Report Generation Agent**

Compiles everything into a polished multi-page PDF.

---

## Why This Project Exists

* Demonstrate multi-agent architecture
* Provide fast automated dataset analysis
* Show integration of classical ML + AI models
* Serve as a reusable pipeline for analytics, hackathons, prototypes

---

## 🛠️ Future Improvements (Planned)

* Add more visualizations
* Support Excel/JSON uploads
* Improve Flask UI dashboard
* Add background processing

---

## License

MIT License — free to use in research, commercial or personal projects.

```


Just tell me the style you prefer.
```
