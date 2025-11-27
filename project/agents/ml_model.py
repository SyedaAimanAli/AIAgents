# %%bash
# cat > project/agents/ml_model.py <<'PY'
import time
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

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


class MLAgent:
    def __init__(self, adk_client: "genai.Client" = None, adk_model_id: str = None):
        self.name = 'ML Agent'
        self.adk_client = adk_client
        self.adk_model_id = adk_model_id

    async def call_adk(self, prompt: str, context: dict = None) -> str | None:
        """Call Google ADK asynchronously for ML insights using new API."""
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

    async def execute(self, df: pd.DataFrame, target: str = None) -> AgentResult:
        start_time = time.time()

        try:
            numeric_df = df.select_dtypes(include=[np.number])

            result = {
                'model_type': 'feature_importance',
                'feature_importance': {},
                'target': None,
                'insights': [],
                'adk_insights': None
            }

            # Auto-select target if not provided
            if target is None:
                if numeric_df.shape[1] < 2:
                    return AgentResult(
                        agent_name=self.name,
                        status='success',
                        data=result,
                        execution_time=time.time() - start_time
                    )
                target = numeric_df.columns[0]

            # Validate target
            if target not in df.columns:
                return AgentResult(
                    agent_name=self.name,
                    status='error',
                    data=None,
                    execution_time=time.time() - start_time,
                    error=f'Target column {target} not in dataframe'
                )

            result['target'] = target

            X = df.drop(columns=[target])
            y = df[target]

            # Encode categorical features
            X_pre = X.copy()
            for col in X_pre.select_dtypes(include=['object', 'category']).columns:
                X_pre[col] = LabelEncoder().fit_transform(X_pre[col].astype(str))

            # Choose ML model type
            if y.dtype.kind in 'iu' and y.nunique() <= 10:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_pre.fillna(0), y.fillna(0).astype(int))
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_pre.fillna(0), y.fillna(0))

            # Compute feature importance
            importances = dict(zip(X_pre.columns, model.feature_importances_))
            sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            result['feature_importance'] = sorted_imp

            # Local insights on top features
            for feat, score in list(sorted_imp.items())[:3]:
                result['insights'].append(f"{feat} important (score={score:.3f})")

            # Google ADK insights
            if self.adk_client:
                try:
                    adk_prompt = "Provide business-relevant insights based on the top features and model importance."
                    result['adk_insights'] = await self.call_adk(adk_prompt, sorted_imp)
                except Exception:
                    result['adk_insights'] = None

            execution_time = time.time() - start_time

            return AgentResult(
                agent_name=self.name,
                status='success',
                data=result,
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
