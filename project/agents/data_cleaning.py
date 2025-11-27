# %%bash
# cat > project/agents/data_cleaning.py <<'PY'
import os
import json
import pandas as pd
import numpy as np
import time
from dataclasses import dataclass
from adk_helper import safe_call_adk


# ADK optional import
try:
    from google import genai
    from google.genai import types
    ADK_AVAILABLE = True
except Exception:
    ADK_AVAILABLE = False


@dataclass
class AgentResult:
    agent_name: str
    status: str
    data: any
    execution_time: float
    error: str = None


class DataCleaningAgent:
    def __init__(self, adk_client=None, adk_model_id=None):
        self.name = "Data Cleaning Agent"
        self.adk_client = adk_client
        self.adk_model_id = adk_model_id

    async def call_adk(self, prompt: str, context: dict = None):
        """Call Google ADK (new API) asynchronously for cleaning suggestions."""
        if not ADK_AVAILABLE or not self.adk_client:
            return None

        try:
            # Merge context into the prompt
            full_prompt = prompt
            if context:
                full_prompt = (
                    "Context:\n"
                    + json.dumps(context, default=str, indent=2)
                    + "\n\n"
                    + prompt
                )

            loop = __import__("asyncio").get_event_loop()

            def blocking_call():
                response = self.adk_client.models.generate_content(
                    model=self.adk_model_id,
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {"text": full_prompt}
                            ]
                        }
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=800
                    )
                )
                return getattr(response, "text", str(response))

            return await loop.run_in_executor(None, blocking_call)

        except Exception:
            return None

    async def execute(self, df: pd.DataFrame):
        start = time.time()

        try:
            # Data summary to provide to ADK
            data_summary = {
                "shape": df.shape,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing": df.isnull().sum().to_dict(),
                "duplicates": int(df.duplicated().sum())
            }

            # ADK call
            adk_resp = None
            if self.adk_client:
                adk_resp = await safe_call_adk(
                    self,
                    "Provide a JSON cleaning strategy for this dataset.",
                    data_summary
                )


            cleaned_df = df.copy()

            cleaning_report = {
                "original_shape": df.shape,
                "missing_values_before": df.isnull().sum().to_dict(),
                "operations": [],
                "adk_recommendations": None,
            }

            # Parse ADK JSON if returned
            if adk_resp:
                try:
                    cleaning_report["adk_recommendations"] = json.loads(adk_resp)
                except Exception:
                    cleaning_report["adk_recommendations"] = str(adk_resp)

            # Numeric columns
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    median_val = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                    cleaning_report["operations"].append(
                        f"Filled {col} nulls with median: {median_val}"
                    )

            # Categorical columns
            categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    mode_val = (
                        cleaned_df[col].mode()[0]
                        if len(cleaned_df[col].mode()) > 0
                        else "Unknown"
                    )
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                    cleaning_report["operations"].append(
                        f"Filled {col} nulls with mode: {mode_val}"
                    )

            # Remove duplicates
            duplicates = int(cleaned_df.duplicated().sum())
            if duplicates > 0:
                cleaned_df = cleaned_df.drop_duplicates()
                cleaning_report["operations"].append(
                    f"Removed {duplicates} duplicate rows"
                )

            cleaning_report["cleaned_shape"] = cleaned_df.shape
            cleaning_report["missing_values_after"] = cleaned_df.isnull().sum().to_dict()

            exec_time = time.time() - start

            return AgentResult(
                agent_name=self.name,
                status="success",
                data={"cleaned_df": cleaned_df, "report": cleaning_report},
                execution_time=exec_time
            )

        except Exception as e:
            exec_time = time.time() - start
            return AgentResult(
                agent_name=self.name,
                status="error",
                data=None,
                execution_time=exec_time,
                error=str(e)
            )