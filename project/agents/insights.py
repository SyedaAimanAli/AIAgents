# %%bash
# cat > project/agents/insights.py <<'PY'
import time
import json
from dataclasses import dataclass

# Latest Google AI SDK (ADK replacement)
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


class BusinessInsightsAgent:
    def __init__(self, adk_client: "genai.Client" = None, adk_model_id: str = None):
        self.name = 'Business Insights Agent'
        self.adk_client = adk_client
        self.adk_model_id = adk_model_id

    async def call_adk(self, prompt: str, context: dict = None) -> str | None:
        """Call Google ADK asynchronously for business insights."""
        if not ADK_AVAILABLE or not self.adk_client:
            return None

        try:
            full_prompt = prompt
            if context:
                full_prompt = (
                    "Context:\n"
                    f"{json.dumps(context, default=str, indent=2)}\n\n"
                    f"{prompt}"
                )

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
                        temperature=0.3,
                        max_output_tokens=1000
                    )
                )
                return getattr(response, "text", None)

            return await loop.run_in_executor(None, blocking_call)

        except Exception:
            return None

    async def execute(self, all_results: dict) -> AgentResult:
        start_time = time.time()

        try:
            insights = {
                'executive_summary': '',
                'key_findings': [],
                'recommendations': []
            }

            # Prepare context for ADK
            context = {k: (v.data if v.status == 'success' else None)
                       for k, v in all_results.items()}

            if self.adk_client:
                prompt = (
                    "Generate a concise executive summary, top key findings, "
                    "and actionable recommendations based on the dataset analysis results."
                )
                adk_text = await self.call_adk(prompt, context)

                if adk_text:
                    lines = [line.strip() for line in adk_text.split("\n") if line.strip()]
                    # First line as executive summary, next 5 lines key findings, next 5 lines recommendations
                    if lines:
                        insights['executive_summary'] = lines[0]
                        insights['key_findings'] = lines[1:6] if len(lines) > 1 else []
                        insights['recommendations'] = lines[6:11] if len(lines) > 6 else []

            # Fallback executive summary
            if not insights['executive_summary'] and 'cleaning' in all_results and all_results['cleaning'].status == 'success':
                cr = all_results['cleaning'].data.get('report', {})
                rows, cols = cr.get('original_shape', [0, 0])
                insights['executive_summary'] = f"Processed dataset with {rows} rows and {cols} columns."

            # Include anomaly insights
            if 'anomaly' in all_results and all_results['anomaly'].status == 'success':
                an = all_results['anomaly'].data
                total_anomalies = an.get('total_anomalies', 0)
                if total_anomalies > 0:
                    insights['key_findings'].append(f"Detected {total_anomalies} anomalies across numeric columns.")
                    insights['recommendations'].append("Investigate outliers for data quality or business trends.")

            # Include ML insights
            if 'ml' in all_results and all_results['ml'].status == 'success':
                ml = all_results['ml'].data
                ml_insights = ml.get('insights', [])
                if ml_insights:
                    insights['key_findings'].extend(ml_insights[:5])
                    insights['recommendations'].append(
                        "Leverage top features to improve predictive modeling and decision-making."
                    )

            # Final fallback
            if not insights['executive_summary']:
                insights['executive_summary'] = "No major findings detected. Dataset appears structured normally."

            # Combined ADK insights for PDF
            combined_lines = [insights['executive_summary']] + insights['key_findings'] + insights['recommendations']
            insights['adk_insights'] = "\n".join([line for line in combined_lines if line])

            # Structured insights for PDF section
            insights['adk_insights_structured'] = {
                'Executive Summary': insights['executive_summary'],
                'Key Findings': insights['key_findings'],
                'Recommendations': insights['recommendations']
            }

            execution_time = time.time() - start_time

            return AgentResult(
                agent_name=self.name,
                status='success',
                data=insights,
                execution_time=execution_time
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                status='error',
                data=None,
                execution_time=time.time() - start_time,
                error=str(e)
            )
