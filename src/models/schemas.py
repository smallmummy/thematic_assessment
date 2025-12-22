"""
Pydantic models for input validation and response formatting.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class SentenceInput(BaseModel):
    """Model for individual sentence input."""

    sentence: str = Field(..., min_length=1, description="The sentence text")
    id: str = Field(..., min_length=1, description="Unique identifier for the sentence")

    @field_validator('sentence')
    @classmethod
    def sentence_not_empty(cls, v: str) -> str:
        """Validate that sentence is not just whitespace."""
        if not v.strip():
            raise ValueError('Sentence cannot be empty or whitespace only')
        return v.strip()

    @field_validator('id')
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        """Validate that id is not just whitespace."""
        if not v.strip():
            raise ValueError('ID cannot be empty or whitespace only')
        return v.strip()


class StandaloneAnalysisRequest(BaseModel):
    """Model for standalone analysis request."""

    surveyTitle: str = Field(..., min_length=1, description="Title of the dataset")
    theme: str = Field(..., min_length=1, description="Overall theme of the sentences")
    baseline: List[SentenceInput] = Field(..., min_length=1, max_length=10000, description="Array of sentences to analyze")


class ComparativeAnalysisRequest(BaseModel):
    """Model for comparative analysis request."""

    surveyTitle: str = Field(..., min_length=1, description="Title of the dataset")
    theme: str = Field(..., min_length=1, description="Overall theme of the sentences")
    baseline: List[SentenceInput] = Field(..., min_length=1, max_length=10000, description="Baseline sentences")
    comparison: List[SentenceInput] = Field(..., min_length=1, max_length=10000, description="Comparison sentences")


class StandaloneCluster(BaseModel):
    """Model for a single cluster in standalone analysis."""

    title: str = Field(..., max_length=100, description="Descriptive title for the cluster")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Overall sentiment")
    sentences: List[str] = Field(..., min_length=1, description="List of sentence IDs in this cluster")
    keyInsights: List[str] = Field(..., min_length=2, max_length=3, description="2-3 key insights")


class ComparativeCluster(BaseModel):
    """Model for a single cluster in comparative analysis."""

    title: str = Field(..., max_length=100, description="Descriptive title for the cluster")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Overall sentiment")
    baselineSentences: List[str] = Field(..., description="Sentence IDs from baseline")
    comparisonSentences: List[str] = Field(..., description="Sentence IDs from comparison")
    keySimilarities: List[str] = Field(..., min_length=2, max_length=3, description="2-3 key similarities")
    keyDifferences: List[str] = Field(..., min_length=2, max_length=3, description="2-3 key differences")


class StandaloneAnalysisResponse(BaseModel):
    """Model for standalone analysis response."""

    clusters: List[StandaloneCluster] = Field(..., description="Array of thematic clusters")


class ComparativeAnalysisResponse(BaseModel):
    """Model for comparative analysis response."""

    clusters: List[ComparativeCluster] = Field(..., description="Array of thematic clusters")


class ErrorDetail(BaseModel):
    """Model for error details."""

    field: Optional[str] = Field(None, description="Field that caused the error")
    issue: str = Field(..., description="Description of the issue")


class ErrorResponse(BaseModel):
    """Model for error response."""

    error: dict = Field(..., description="Error information")

    @classmethod
    def create(cls, code: str, message: str, details: Optional[dict] = None) -> 'ErrorResponse':
        """Factory method to create error response."""
        error_dict = {
            "code": code,
            "message": message
        }
        if details:
            error_dict["details"] = details
        return cls(error=error_dict)
