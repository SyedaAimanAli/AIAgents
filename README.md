# Multi-Agent Automated Data Analysis Pipeline

### End-to-End Analysis • Business Insights • PDF Reporting • Flask UI • ADK (Gemini) Enhanced

This project is a **complete automated data-analysis system** powered by a **multi-agent architecture**, Google’s **ADK (Gemini)** intelligence, and a clean **Flask web interface**.

Upload any CSV file, and the system performs:

✔ Data Cleaning
✔ Exploratory Data Analysis (EDA)
✔ Anomaly Detection
✔ Machine-Learning Modeling
✔ Business Insights (via ADK)
✔ Exported PDF Report
✔ All via an orchestrated asynchronous pipeline

---

## Features

### ** 1. Multi-Agent AI Architecture**

Each part of the workflow is handled by an independent agent:

| Agent                               | Responsibilities                                                |
| ----------------------------------- | --------------------------------------------------------------- |
| **Data Cleaning Agent**             | Missing values, duplicates, column fixes, summary generation    |
| **EDA Agent**                       | Visualizations (encoded), distributions, correlations           |
| **Anomaly Agent**                   | Outlier detection + anomaly summary                             |
| **ML Agent**                        | Auto-ML model training, prediction analysis, feature importance |
| **Insights Agent (Gemini-powered)** | Business-level insights + executive summary                     |
| **Report Agent**                    | Creates a professional PDF report                               |

---

## 🧠 Powered by Google ADK (Gemini)

If an ADK API key is provided, agents additionally use Gemini to:

* Suggest better cleaning strategies
* Provide structured insights
* Generate a human-friendly **Executive Summary**
* Assist in report creation

The system works **with or without** ADK enabled.

---

## Flask Web Interface

A clean UI allows users to:

### ✔ Upload CSV file

### ✔ (Optional) Specify a target column (for supervised ML)

### ✔ See real-time "Processing…" loader

### ✔ Download the generated PDF

### ✔ View agent execution summary + insights

---

## Project Structure

```
/project
├── agents/ 
├── main.py
├── dataset.py # (optional) dataset generator for demo/sample
├── report logic, etc.

/webapp 
├── app.py 
├── templates/ # HTML templates (index, results)
└── static/ # (optional) CSS / assets
```

---

## Installation & Setup

### **1. Clone the repository**

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. (Optional) Set ADK API Key**

```bash
export GOOGLE_ADK_API_KEY="YOUR_API_KEY"
```

### **4. Run the Flask app**

```bash
python app.py
```

Then visit:

```
http://127.0.0.1:5000
```

---

##  How It Works

### **Step 1 — Upload CSV**

The user uploads a dataset through the web interface.

### **Step 2 — Multi-Agent Execution**

`run_pipeline()` orchestrates all agents asynchronously.

### **Step 3 — Insights + Report**

Gemini generates insights and a natural language executive summary.

### **Step 4 — PDF Output**

A clean, structured PDF is created by the Report Agent using ReportLab.

---

## Example Output (Summary)

* Dataset shape
* Cleaning steps performed
* EDA visuals & findings
* Anomaly detection results
* ML feature importance
* AI-generated Executive Summary
* Full insights section
* Agent execution time table

---

## Why This Project Is Valuable

This project demonstrates:

* Real multi-agent design
* Practical AI integration
* Clean full-stack development
* Async orchestration
* Automated reporting
* Real-world dataset processing pipeline

Perfect for:

* Kaggle pipelines
* AI engineering portfolios
* Automated analytics tools
* Enterprise internal dashboards
* Capstone projects

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you want to add.

---

## License

MIT License — feel free to use and modify.

