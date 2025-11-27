# %%bash
# cat > project/agents/anomaly.py <<'PY'
import time
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Latest Google AI SDK (ADK Replacement)
try:
    from google import genai
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False


@dataclass
class AgentResult:
    agent_name: str
    status: str
    data: any
    execution_time: float
    error: str = None


class AnomalyAgent:
    def __init__(self, adk_client: "genai.Client" = None, adk_model_id: str = None):
        self.name = 'Anomaly Detection Agent'
        self.adk_client = adk_client
        self.adk_model_id = adk_model_id

    async def call_adk(self, prompt: str, context: dict = None) -> str | None:
        """Call Google ADK asynchronously using new API format."""
        if not ADK_AVAILABLE or not self.adk_client:
            return None

        try:
            full_prompt = prompt
            if context:
                full_prompt = f"Context:\n{json.dumps(context, default=str, indent=2)}\n\n{prompt}"

            import asyncio
            loop = asyncio.get_event_loop()

            def blocking_call():
                response = self.adk_client.models.generate_content(
                    model=self.adk_model_id,
                    contents=[{
                        "role": "user",
                        "parts": [{"text": full_prompt}]
                    }],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=800
                    )
                )
                return getattr(response, "text", None)

            return await loop.run_in_executor(None, blocking_call)

        except Exception:
            return None

    async def execute(self, df: pd.DataFrame) -> AgentResult:
        start_time = time.time()

        try:
            report = {
                'outliers_by_column': {},
                'anomaly_summary': [],
                'total_anomalies': 0,
                'adk_insights': None
            }

            numeric_df = df.select_dtypes(include=[np.number])

            # Detect IQR outliers
            for col in numeric_df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

                if len(outliers) > 0:
                    report['outliers_by_column'][col] = {
                        'count': int(len(outliers)),
                        'percentage': float(len(outliers) / len(df) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'values': outliers[col].tolist()[:10]
                    }

                    report['anomaly_summary'].append(
                        f"{col}: {len(outliers)} outliers "
                        f"({len(outliers) / len(df) * 100:.2f}%)"
                    )

                    report['total_anomalies'] += int(len(outliers))

            # Google ADK insights
            if self.adk_client:
                try:
                    adk_prompt = (
                        "Provide a concise business-oriented summary and recommendations "
                        "based on the detected anomalies."
                    )
                    report['adk_insights'] = await safe_call_adk(
                        self, adk_prompt, report['outliers_by_column']
                    )
                except Exception:
                    report['adk_insights'] = None


            execution_time = time.time() - start_time

            return AgentResult(
                agent_name=self.name,
                status='success',
                data=report,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                agent_name=self.name,
                status='error',
                data=None,
                execution_time=execution_time,
                error=str(e)
            )