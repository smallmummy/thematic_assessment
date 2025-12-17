"""
Insight generation service using OpenAI GPT-4o.
"""

import logging
import time
import json
from typing import List, Dict, Any
from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class InsightService:
    """Service for generating insights from clustered sentences using LLM."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_sentences_for_insights: int = 20,
        max_sentences_for_comparison: int = 15,
        max_title_length: int = 100
    ):
        """
        Initialize the insight generation service.

        Args:
            api_key: OpenAI API key
            model: Model name to use (default: gpt-4o)
            max_sentences_for_insights: Maximum sentences to send for insight generation (default: 20)
            max_sentences_for_comparison: Maximum sentences per dataset for comparison (default: 15)
            max_title_length: Maximum length for generated titles (default: 100)
        """
        logger.info("Initializing InsightService with model: %s", model)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_sentences_for_insights = max_sentences_for_insights
        self.max_sentences_for_comparison = max_sentences_for_comparison
        self.max_title_length = max_title_length

    async def generate_standalone_cluster_content_async(
        self,
        sentences: List[str],
        theme: str,
        sentiment: str
    ) -> Dict[str, Any]:
        """
        Generate both title and insights in one async call using JSON mode.

        Args:
            sentences: List of sentences in the cluster
            theme: Overall theme of the analysis
            sentiment: Sentiment of the cluster (positive/negative/neutral)

        Returns:
            Dict with 'title' and 'insights' keys
        """
        start_time = time.time()

        # Use all sentences if within limit, otherwise trim to max
        if len(sentences) > self.max_sentences_for_insights:
            sample_sentences = sentences[:self.max_sentences_for_insights]
        else:
            sample_sentences = sentences

        logger.info(
            "[PERF] Generating title + insights for cluster with %s sentences (using %s) (%s)",
            len(sentences), len(sample_sentences), sentiment
        )

        prompt = f"""Analyze this customer feedback cluster about "{theme}" (sentiment: {sentiment}).

Sentences:
{chr(10).join(f"- {s}" for s in sample_sentences)}

Generate a title and 2-3 key insights. Each insight should:
1. Be specific and evidence-based
2. Highlight important patterns or findings
3. Use **bold markdown** to emphasize 1-3 key words or phrases per insight
4. Be actionable and clear
5. Be 1-2 sentences long

Return as JSON:
{{
  "title": "Concise sub-theme title (max 10 words)",
  "insights": [
    "Insight 1 with **bold** emphasis",
    "Insight 2 with **bold** emphasis",
    "Insight 3 with **bold** emphasis"
  ]
}}"""

        try:
            api_start = time.time()
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing customer feedback. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=450
            )
            api_time = time.time() - api_start

            result = json.loads(response.choices[0].message.content)

            # Validate and clean the result
            title = result.get('title', f"{theme.title()} Theme").strip().strip('"\'')
            if len(title) > self.max_title_length:
                truncate_pos = self.max_title_length - 3
                title = title[:truncate_pos] + "..."

            insights = result.get('insights', [])
            # Ensure we have 2-3 insights
            if len(insights) < 2:
                # Record error log for further investigation
                logger.error(
                    "Only generated %s insights, expected 2-3",
                    len(insights)
                )
                # Fallback
                insights.append("Additional analysis needed for comprehensive insights.")
            elif len(insights) > 3:
                # Record error log for further investigation
                logger.error(
                    "Generated %s insights, expected max 3",
                    len(insights)
                )
                # Fallback
                insights = insights[:3]

            total_time = time.time() - start_time
            logger.info(
                "[PERF] Generated title + %s insights in %.3fs (API: %.3fs): %s",
                len(insights), total_time, api_time, title
            )

            return {
                'title': title,
                'insights': insights
            }

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(
                "[PERF] Error generating cluster content in %.3fs: %s",
                total_time, str(e)
            )
            # Record error log for further investigation
            logger.error(
                "Error generating standalone insights: %s", str(e), exc_info=True
            )
            # Fallback
            return {
                'title': f"{theme.title()} Theme",
                'insights': [
                    f"Customers expressed **{sentiment}** sentiment regarding {theme.lower()}.",
                    f"This theme appeared in **{len(sentences)} customer comments**."
                ]
            }

    async def generate_comparative_cluster_content_async(
        self,
        baseline_sentences: List[str],
        comparison_sentences: List[str],
        theme: str,
        sentiment: str
    ) -> Dict[str, Any]:
        """
        Generate title, similarities, and differences in one async call using JSON mode.

        Args:
            baseline_sentences: Sentences from baseline dataset
            comparison_sentences: Sentences from comparison dataset
            theme: Overall theme of the analysis
            sentiment: Overall sentiment of the cluster

        Returns:
            Dict with 'title', 'similarities', and 'differences' keys
        """
        start_time = time.time()

        # Use all sentences if within limit, otherwise trim to max
        if len(baseline_sentences) > self.max_sentences_for_comparison:
            sample_baseline = baseline_sentences[:self.max_sentences_for_comparison]
        else:
            sample_baseline = baseline_sentences

        if len(comparison_sentences) > self.max_sentences_for_comparison:
            sample_comparison = comparison_sentences[:self.max_sentences_for_comparison]
        else:
            sample_comparison = comparison_sentences

        logger.info(
            "[PERF] Generating title + comparative insights (baseline: %s/%s, comparison: %s/%s, %s)",
            len(sample_baseline), len(baseline_sentences),
            len(sample_comparison), len(comparison_sentences),
            sentiment
        )

        prompt = f"""Compare these two sets of customer feedback about "{theme}" (sentiment: {sentiment}).

BASELINE Dataset:
{chr(10).join(f"- {s}" for s in sample_baseline) if sample_baseline else "- (No baseline sentences)"}

COMPARISON Dataset:
{chr(10).join(f"- {s}" for s in sample_comparison) if sample_comparison else "- (No comparison sentences)"}

Generate:
1. A concise title for this theme
2. 2-3 key similarities between the datasets
3. 2-3 key differences between the datasets

Each insight should use **bold markdown** to emphasize key points.

Return as JSON:
{{
  "title": "Concise sub-theme title (max 10 words)",
  "similarities": [
    "Similarity 1 with **bold** emphasis",
    "Similarity 2 with **bold** emphasis",
    "Similarity 3 with **bold** emphasis"
  ],
  "differences": [
    "Difference 1 with **bold** emphasis",
    "Difference 2 with **bold** emphasis",
    "Difference 3 with **bold** emphasis"
  ]
}}"""

        try:
            api_start = time.time()
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at comparative analysis of customer feedback. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
                max_tokens=700
            )
            api_time = time.time() - api_start

            result = json.loads(response.choices[0].message.content)

            # Validate and clean the result
            title = result.get('title', f"{theme.title()} Sub Theme").strip().strip('"\'')
            if len(title) > self.max_title_length:
                # Truncate if needed
                truncate_pos = self.max_title_length - 3
                logger.warning(
                    "Generated title too long (%s chars), truncating to %s",
                    len(title), self.max_title_length
                )
                title = title[:truncate_pos] + "..."

            similarities = result.get('similarities', [])
            differences = result.get('differences', [])

            # Ensure we have 2-3 of each
            if len(similarities) < 2:
                # Record error log for further investigation
                logger.error(
                    "Only generated %s similarities, expected 2-3",
                    len(similarities)
                )
                # Fallback
                similarities.append(f"Both datasets mention **{title.lower()}** as a key theme.")
            if len(similarities) > 3:
                # Record error log for further investigation
                logger.error(
                    "Generated %s similarities, expected max 3",
                    len(similarities)
                )
                # Fallback
                similarities = similarities[:3]

            if len(differences) < 2:
                # Record error log for further investigation
                logger.error(
                    "Only generated %s differences, expected 2-3",
                    len(differences)
                )
                # Fallback
                differences.append(f"The datasets show **different emphasis** on aspects of {title.lower()}.")
            if len(differences) > 3:
                # Record error log for further investigation
                logger.error(
                    "Generated %s differences, expected max 3",
                    len(differences)
                )
                # Fallback
                differences = differences[:3]

            total_time = time.time() - start_time
            logger.info(
                "[PERF] Generated title + %s similarities + %s differences in %.3fs (API: %.3fs): %s",
                len(similarities), len(differences), total_time, api_time, title
            )

            return {
                'title': title,
                'similarities': similarities,
                'differences': differences
            }

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(
                "[PERF] Error generating comparative cluster content in %.3fs: %s",
                total_time, str(e)
            )
            # Record error log for further investigation
            logger.error(
                "Error generating comparative insights: %s", str(e), exc_info=True
            )

            # Fallback
            return {
                'title': f"{theme.title()} Theme",
                'similarities': [
                    f"Both datasets discuss **{theme.lower()}** with similar themes.",
                    f"Overall sentiment is **{sentiment}** in both periods."
                ],
                'differences': [
                    f"Baseline contains **{len(baseline_sentences)} mentions** while comparison has **{len(comparison_sentences)}**.",
                    "Further analysis needed to identify specific differences."
                ]
            }
