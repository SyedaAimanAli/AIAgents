from flask import Flask, render_template, request, redirect, send_file, flash
import os
import asyncio
import uuid
import sys
import pandas as pd

sys.path.append("..")
from main import run_pipeline

app = Flask(__name__)
# app.secret_key = 'your-secret-key-here-change-in-production'  
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        flash("No file uploaded.", "error")
        return redirect("/")

    file = request.files["file"]
    target = request.form.get("target", "").strip() or None

    if file.filename == "":
        flash("No file selected.", "error")
        return redirect("/")
    
    if not file.filename.endswith('.csv'):
        flash("Please upload a CSV file.", "error")
        return redirect("/")

    # Save uploaded file
    file_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.csv")
    file.save(filepath)

    # Validate target column if provided
    try:
        df = pd.read_csv(filepath)
        
        if target:
            if target not in df.columns:
                flash(f"Target column '{target}' not found in CSV. Available columns: {', '.join(df.columns.tolist())}", "error")
                os.remove(filepath)  # Clean up
                return redirect("/")
            
            # Check if target column is suitable (numeric for regression)
            if df[target].dtype not in ['int64', 'float64']:
                flash(f"Warning: Target column '{target}' is not numeric. ML predictions may not work as expected.", "warning")
        
    except Exception as e:
        flash(f"Error reading CSV file: {str(e)}", "error")
        os.remove(filepath)
        return redirect("/")

    # Run pipeline
    try:
        results = asyncio.run(run_pipeline(filepath, target))
        pdf_path = results["report"].data.get("pdf_path")
        
        # Pass target info to results page
        return render_template(
            "results.html",
            pdf_path=pdf_path,
            results=results["agent_results"],
            insights=results["insights"],
            target_column=target,
            dataset_shape=df.shape
        )
    
    except Exception as e:
        flash(f"Error during analysis: {str(e)}", "error")
        if os.path.exists(filepath):
            os.remove(filepath)
        return redirect("/")

@app.route("/download")
def download():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        flash("File not found.", "error")
        return redirect("/")
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)