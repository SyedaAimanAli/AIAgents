import io
import base64
import time
from dataclasses import dataclass
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import re
import os
import json

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

class ReportAgent:
    def __init__(self, output_pdf_dir='project', adk_client=None, adk_model_id=None):
        self.name = 'Report Generation Agent'
        self.output_pdf_dir = output_pdf_dir
        self.adk_client = adk_client
        self.adk_model_id = adk_model_id

    async def execute(self, all_results: dict, business_insights: AgentResult) -> AgentResult:
        start_time = time.time()
        try:
            pdf_path = await self._generate_pdf_report(all_results, business_insights)
            exec_time = time.time() - start_time
            return AgentResult(
                agent_name=self.name,
                status='success',
                data={'pdf_path': pdf_path},
                execution_time=exec_time
            )
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                status='error',
                data=None,
                execution_time=time.time() - start_time,
                error=str(e)
            )

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'#+', '', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        return text.strip()

    async def call_adk(self, prompt: str, context: dict = None):
        """Call Google ADK (new API) asynchronously for description generation."""
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
                        max_output_tokens=400
                    )
                )
                return getattr(response, "text", str(response))

            return await loop.run_in_executor(None, blocking_call)

        except Exception as e:
            print(f"ADK call error: {e}")
            return None

    async def _generate_agent_description(self, agent_name: str, agent_key: str) -> str:
        """Generate agent description using ADK"""
        
        # Define what each agent does for context
        agent_context = {
            "cleaning": {
                "purpose": "data preparation and quality assurance",
                "tasks": "handles missing values, resolves inconsistent formats, removes duplicates, standardizes data types",
                "output": "clean, reliable dataset ready for analysis"
            },
            "eda": {
                "purpose": "exploratory data analysis and pattern discovery",
                "tasks": "generates statistical summaries, creates visualizations, identifies trends, correlations, and distributions",
                "output": "insights about data structure, relationships, and quality issues"
            },
            "anomaly": {
                "purpose": "outlier detection and data quality monitoring",
                "tasks": "identifies unusual patterns using IQR-based statistical methods, flags potential errors or special events",
                "output": "summary of anomalies with affected features and counts"
            },
            "ml": {
                "purpose": "predictive modeling and feature analysis",
                "tasks": "trains baseline models, calculates feature importance scores, identifies key predictive patterns",
                "output": "ranked features showing what drives the target outcome"
            }
        }
        
        context = agent_context.get(agent_key.lower(), {
            "purpose": "data analysis",
            "tasks": "performs analysis tasks",
            "output": "analysis results"
        })
        
        if self.adk_client:
            try:
                prompt = f"""Generate a concise, professional description for the "{agent_name}" in a data analysis pipeline.

Agent Details:
- Purpose: {context['purpose']}
- Key Tasks: {context['tasks']}
- Output: {context['output']}

Requirements:
1. Write exactly 3-4 sentences
2. Use third person (e.g., "This agent...", "It performs...", "Its purpose...")
3. Vary sentence structure - don't start every sentence the same way
4. Focus on: what it does, why it matters, how it helps
5. Professional and technical tone
6. NO bullet points, NO markdown formatting, NO asterisks, NO quotes
7. Plain text only, natural prose

Example format: "This agent is responsible for preparing the dataset for analysis. It handles missing values, resolves inconsistent formats, and removes duplicate entries. Its purpose is to ensure that downstream agents work with clean, reliable data."

Generate description:"""

                response = await self.call_adk(prompt)
                
                if response:
                    description = self._clean_text(response)
                    # Additional cleanup for any remaining formatting
                    description = description.replace('*', '').replace('#', '')
                    return description
                
            except Exception as e:
                print(f"Error generating description for {agent_name}: {e}")
        
        # Fallback if ADK fails or is unavailable
        return self._get_fallback_description(agent_key)

    def _get_fallback_description(self, agent_key: str) -> str:
        """Fallback descriptions if ADK fails"""
        descriptions = {
            "cleaning": "This agent is responsible for preparing the dataset for analysis. It handles missing values, resolves inconsistent formats, removes duplicate entries, and applies recommendations when available. Its purpose is to ensure that downstream agents work with clean, reliable, and standardized data.",
            "eda": "The EDA Agent explores the dataset and generates statistical summaries and visualizations. Its role is to highlight trends, distributions, correlations, and structural patterns. It helps users understand what features matter, how values are spread, and where potential data quality issues may exist.",
            "anomaly": "The Anomaly Detection Agent identifies unusual patterns or outliers in numeric columns. These anomalies may indicate data-entry errors, special business events, or operational risks. The agent uses statistical IQR-based detection and produces a summary of affected features.",
            "ml": "The ML Agent trains a quick baseline model to uncover feature importance and predictive patterns. It supports both regression and classification tasks. The primary purpose is not model accuracy, but interpretability—understanding which features drive the target outcome the most."
        }
        return descriptions.get(agent_key.lower(), "Agent description not available.")

    async def _generate_pdf_report(self, all_results: dict, business_insights: AgentResult) -> str:
        os.makedirs(self.output_pdf_dir, exist_ok=True)

        pdf_filename = f"{self.output_pdf_dir}/analysis_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        doc = SimpleDocTemplate(
            pdf_filename,
            pagesize=letter,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1a73e8'))
        heading_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, spaceBefore=12, spaceAfter=6)
        subheading_style = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=12, spaceBefore=6, spaceAfter=3)
        normal_style = styles['Normal']

        story = [
            Paragraph('Multi-Agent Data Analysis Report', title_style),
            Paragraph(f"Generated on {datetime.now():%Y-%m-%d %H:%M:%S}", normal_style),
            Spacer(1, 0.2 * inch)
        ]

        # Safeguard
        if not business_insights or not business_insights.data:
            business_insights = AgentResult(
                agent_name="Business Insights Agent",
                status="success",
                data={'executive_summary': "No executive summary available.",
                      'adk_insights_structured': {}},
                execution_time=0.0
            )

        # Executive Summary
        exec_summary = business_insights.data.get('executive_summary', 'No executive summary available.')
        exec_summary = self._clean_text(exec_summary)
        story.append(Paragraph('Executive Summary', heading_style))
        for line in exec_summary.split('\n'):
            if line.strip():
                story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 0.05 * inch))

        # Business Insights
        insights_struct = business_insights.data.get('adk_insights_structured', {})
        if insights_struct:
            story.append(Paragraph('Insights', heading_style))
            for section, content in insights_struct.items():
                if section.lower() == 'executive summary':
                    continue
                story.append(Paragraph(section, subheading_style))
                if isinstance(content, list):
                    for item in content:
                        story.append(Paragraph(f"• {self._clean_text(item)}", normal_style))
                else:
                    story.append(Paragraph(self._clean_text(str(content)), normal_style))
                story.append(Spacer(1, 0.05 * inch))

        story.append(PageBreak())

        # Agent Summary Table
        story.append(Paragraph('Agent Execution Summary', heading_style))
        table_data = [['Agent', 'Status', 'Execution Time (s)']]
        for k, v in all_results.items():
            status_text = 'SUCCESS' if v.status == 'success' else f"ERROR: {v.error or 'Unknown'}"
            table_data.append([v.agent_name, status_text, f"{v.execution_time:.3f}"])

        t = Table(table_data, colWidths=[3*inch, 2*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a73e8')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.1 * inch))

        # Add heading before Insights Agent description
        story.append(Paragraph('Insights Agent', subheading_style))
        story.append(Paragraph(
            "The Business Insights Agent consolidates outputs from all other agents, transforms them into "
            "actionable findings, and generates recommendations. It acts as the decision-making layer of the "
            "pipeline, summarizing complex analyses into clear business-focused insights.",
            normal_style
        ))
        story.append(Spacer(1, 0.2 * inch))

        # Detailed Results - Generate descriptions dynamically
        for idx, (key, result) in enumerate(all_results.items()):
            story.append(Paragraph(f"{result.agent_name} - Results", heading_style))

            # Generate agent description using ADK
            description = await self._generate_agent_description(result.agent_name, key)
            story.append(Paragraph(description, normal_style))
            story.append(Spacer(1, 0.1 * inch))

            # Skip further processing for cleaning agent (no data to display)
            if key.lower() == "cleaning":
                continue

            data = result.data or {}

            # EDA visualizations - smaller size
            if key.lower() == 'eda' and data.get('visualizations'):
                for viz in data['visualizations']:
                    try:
                        img_bytes = base64.b64decode(viz['data'])
                        bio = io.BytesIO(img_bytes)
                        img = Image(bio, width=4.5*inch, height=3*inch)
                        story.append(Paragraph(viz.get('type', 'Chart'), subheading_style))
                        story.append(img)
                        story.append(Spacer(1, 0.05 * inch))
                    except:
                        pass

            # Anomalies
            if key.lower() == 'anomaly' and data.get('anomaly_summary'):
                story.append(Paragraph('Anomaly Summary:', subheading_style))
                for s in data['anomaly_summary']:
                    story.append(Paragraph(f"• {self._clean_text(s)}", normal_style))

            # ML features
            if key.lower() == 'ml' and data.get('feature_importance'):
                story.append(Paragraph('Top Features:', subheading_style))
                for f, score in list(data['feature_importance'].items())[:10]:
                    story.append(Paragraph(f"• {f}: {score:.4f}", normal_style))

            # Only add page break between agent sections, not after the last one
            # Skip page break after anomaly if ML is next (they're short and can share a page)
            if idx < len(all_results) - 1:
                next_key = list(all_results.keys())[idx + 1] if idx + 1 < len(all_results) else None
                # Don't add page break between anomaly and ml agents
                if not (key.lower() == 'anomaly' and next_key and next_key.lower() == 'ml'):
                    story.append(PageBreak())

        doc.build(story)
        return pdf_filename