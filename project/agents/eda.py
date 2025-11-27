# %%bash
# cat > project/agents/eda_agent.py <<'PY'
import os
import io
import json
import base64
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


class EDAAgent:
    def __init__(self, charts_dir='project/charts', adk_client: "genai.Client" = None, adk_model_id: str = None):
        self.name = "EDA Agent"
        self.charts_dir = charts_dir
        os.makedirs(self.charts_dir, exist_ok=True)
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
                    contents=[{"role": "user", "parts": [{"text": full_prompt}]}],
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
            eda_report = {
                'basic_stats': df.describe(include='all').to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'correlations': {},
                'visualizations': [],
                'adk_insights': None
            }

            numeric_df = df.select_dtypes(include=[np.number])

            # Correlation matrix + heatmap
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                eda_report['correlations'] = corr_matrix.to_dict()

                fig, ax = plt.subplots(figsize=(8, 6))
                cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

                ax.set_xticks(range(len(corr_matrix.columns)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                ax.set_yticks(range(len(corr_matrix.columns)))
                ax.set_yticklabels(corr_matrix.columns)

                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', fontsize=7)

                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)

                eda_report['visualizations'].append({
                    'type': 'correlation_heatmap',
                    'data': base64.b64encode(buf.read()).decode()
                })
                plt.close(fig)

            # Distribution plots for first 3 numeric columns
            for col in numeric_df.columns[:3]:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {col}')

                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)

                eda_report['visualizations'].append({
                    'type': f'distribution_{col}',
                    'data': base64.b64encode(buf.read()).decode()
                })
                plt.close(fig)

            # Google ADK insights
            if self.adk_client:
                try:
                    stats_summary = {
                        'numeric_columns': len(numeric_df.columns),
                        'statistics': df.describe().to_dict()
                    }

                    eda_report['adk_insights'] = await self.call_adk(
                        prompt="Provide 3 key insights from this dataset.",
                        context=stats_summary
                    )
                except Exception:
                    eda_report['adk_insights'] = None

            execution_time = time.time() - start_time
            return AgentResult(
                agent_name=self.name,
                status='success',
                data=eda_report,
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
