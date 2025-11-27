# %%bash
# cat > project/main.py <<'PY'
import asyncio
import pandas as pd
import time
import os
import sys
import chardet
sys.path.append(os.path.dirname(__file__))
os.environ['GOOGLE_ADK_API_KEY'] = 'AIzaSyCCxO72g3PQ-IzjMro0UglRoqGWhStVjk0'


from agents.data_cleaning import DataCleaningAgent, AgentResult as CleaningResult
from agents.eda import EDAAgent, AgentResult as EDAResult
from agents.anomaly import AnomalyAgent, AgentResult as AnomalyResult
from agents.ml_model import MLAgent, AgentResult as MLResult
from agents.insights import BusinessInsightsAgent, AgentResult as InsightsResult
from agents.report import ReportAgent, AgentResult as ReportResult
from dataset import generate_sample_dataset
from google import genai
from google.genai import types


# Optional ADK client wiring (only used if environment variable GOOGLE_ADK_API_KEY is set)
ADK_CLIENT = None
ADK_MODEL = None
try:
    if os.getenv('GOOGLE_ADK_API_KEY'):
        from google import genai
        ADK_CLIENT = genai.Client(api_key=os.getenv('GOOGLE_ADK_API_KEY'))
        ADK_MODEL = 'gemini-2.0-flash'
except Exception:
    ADK_CLIENT = None
    ADK_MODEL = None

resp = ADK_CLIENT.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        {
            "role": "user",
            "parts": [
                {"text": "Write one sentence about data analysis."}
            ]
        }
    ],
    config=types.GenerateContentConfig(
        temperature=0.2
    )
)
print(resp.text)


def load_csv_with_fallback(path):
    # Read raw bytes
    with open(path, "rb") as f:
        raw = f.read()

    # Detect encoding
    enc = chardet.detect(raw)["encoding"] or "latin1"

    # Load with detected encoding
    try:
        return pd.read_csv(path, encoding=enc)
    except Exception:
        return pd.read_csv(path, encoding="latin1")


async def run_pipeline(csv_path: str, target: str = None):
    start_total = time.time()
    print('\n' + '='*60)

    print('Multi-Agent Pipeline Starting')
    print('='*60)

    # df = pd.read_csv(csv_path)
    df = load_csv_with_fallback(csv_path)
    print(f'Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns')

    # instantiate agents (pass ADK client if available)
    cleaning_agent = DataCleaningAgent(adk_client=ADK_CLIENT, adk_model_id=ADK_MODEL)
    eda_agent = EDAAgent(adk_client=ADK_CLIENT, adk_model_id=ADK_MODEL)
    anomaly_agent = AnomalyAgent(adk_client=ADK_CLIENT, adk_model_id=ADK_MODEL)
    ml_agent = MLAgent(adk_client=ADK_CLIENT, adk_model_id=ADK_MODEL)
    insights_agent = BusinessInsightsAgent(adk_client=ADK_CLIENT, adk_model_id=ADK_MODEL)
    report_agent = ReportAgent(
        output_pdf_dir='project',
        adk_client=ADK_CLIENT, 
        adk_model_id=ADK_MODEL
    )

    # Phase 1: parallel execution
    tasks = {
        'cleaning': cleaning_agent.execute(df),
        'eda': eda_agent.execute(df),
        'anomaly': anomaly_agent.execute(df),
        'ml': ml_agent.execute(df, target)
    }

    keys = list(tasks.keys())
    completed = await asyncio.gather(*tasks.values(), return_exceptions=True)

    results = {}
    for k, res in zip(keys, completed):
        if isinstance(res, Exception):
            results[k] = type('R', (), {'agent_name': k, 'status': 'error', 'data': None, 'execution_time': 0, 'error': str(res)})()
        else:
            results[k] = res
        status = 'SUCCESS' if results[k].status=='success' else 'ERROR'
        print(f"{results[k].agent_name:<25} {status:<8} ({results[k].execution_time:.2f}s)")

    # Phase 2: insights
    insights = await insights_agent.execute(results)
    print(f"{insights.agent_name:<25} {insights.status.upper():<8} ({insights.execution_time:.2f}s)")

    # Phase 3: report
    report = await report_agent.execute(results, insights)
    print(f"{report.agent_name:<25} {report.status.upper():<8} ({report.execution_time:.2f}s)")

    total_time = time.time() - start_total
    print(f'='*60)
    print(f'Pipeline complete - total time: {total_time:.2f}s')
    print(f'PDF report at:', report.data.get('pdf_path'))
    return {'agent_results': results, 'insights': insights, 'report': report}

if __name__ == '__main__':

    df = generate_sample_dataset()
    path = 'project/sample_data.csv'
    df.to_csv(path, index=False)
    print(f"Sample dataset created at '{path}'")
    import asyncio
    asyncio.run(run_pipeline(path, target='profit'))