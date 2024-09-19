from typing import Any, Dict, List, Optional, Union, Iterator
from pydantic import ConfigDict, Field, BaseModel
from zoneinfo import ZoneInfo
from datetime import datetime
from enum import Enum
import httpx


def convert_datetime_to_gmt(dt: datetime) -> str:
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))

    return dt.strftime("%Y-%m-%dT%H:%M:%S%z")


class CoreModel(BaseModel):
    class Config:
        json_encoders = {datetime: convert_datetime_to_gmt}
        populate_by_name = True
        from_attributes = False


class AzureOpenAIFinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALLS = "function_calls"
    RECITATION = "recitation"
    ERROR = "error"
    UNKNOWN = "unknown"


class ParseType(Enum):
    LIST = "list"
    JSON = "json"
    RAW = "raw"


class AzureOpenAITokenUsage(CoreModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(CoreModel):
    """Chat message."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict:
        data = {"role": self.role.value, "content": self.content}

        return data


class MultiGenerationsResponse(BaseModel):
    results: List[BaseModel]
    _raw_response: Any

    def __iter__(self) -> Iterator[BaseModel]:
        return iter(self.results)

    def __getitem__(self, index: int) -> BaseModel:
        return self.results[index]

    def __len__(self) -> int:
        return len(self.results)


class LiteLLMKwargs(CoreModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    timeout: Optional[Union[float, str, httpx.Timeout]] = Field(
        default=None,
        description="Request timeout",
    )
    temperature: Optional[float] = Field(
        default=0.0,
        description="Sampling temperature",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling parameter",
    )
    n: Optional[int] = Field(
        default=None,
        description="Number of completions to generate",
    )
    stream: Optional[bool] = Field(
        default=None,
        description="Whether to stream the response",
    )
    stream_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Stream options",
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Sequences where the API will stop generating",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate",
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        description="Presence penalty parameter",
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        description="Frequency penalty parameter",
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description="Logit bias dictionary",
    )
    user: Optional[str] = Field(default=None, description="User identifier")
    response_format: Optional[Union[Dict[str, Any], Any]] = Field(
        default=None,
        description="Response format",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for deterministic output",
    )
    tools: Optional[List[Any]] = Field(default=None, description="List of tools")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Tool choice",
    )
    logprobs: Optional[bool] = Field(
        default=None,
        description="Include log probabilities",
    )
    top_logprobs: Optional[int] = Field(
        default=None,
        description="Number of top log probabilities to return",
    )
    parallel_tool_calls: Optional[bool] = Field(
        default=None,
        description="Allow parallel tool calls",
    )
    deployment_id: Optional[str] = Field(
        default=None,
        description="Deployment ID for Azure OpenAI",
    )
    extra_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Extra headers for the request",
    )
    functions: Optional[List[Any]] = Field(
        default=None,
        description="List of functions (soon to be deprecated)",
    )
    function_call: Optional[str] = Field(
        default=None,
        description="Function call (soon to be deprecated)",
    )
    base_url: Optional[str] = Field(default=None, description="Base URL for the API")
    api_version: Optional[str] = Field(default=None, description="API version")
    api_key: Optional[str] = Field(default=None, description="API key")
    model_list: Optional[List[Any]] = Field(
        default=None,
        description="List of model configurations",
    )
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID",
    )


class Provider(str, Enum):
    ollama = "ollama"
    openai = "openai"
    azure = "azure"
    anthropic = "anthropic"
    claude = "claude"
    groq = "groq"
    mistral = "mistral"
    llama3 = "llama3"
    llama2 = "llama2"
    ollama_chat = "ollama_chat"


class Model(str, Enum):
    groq_llama3_1_70b_versatile = "groq/llama-3.1-70b-versatile"
    openai_gpt_4o = "gpt-4o"
    openai_gpt_3_5_turbo = "gpt-3.5-turbo"
    openai_gpt_4o_mini = "gpt-4o-mini"

class Criteria(str, Enum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"

    @classmethod
    def to_list(cls) -> list["Criteria"]:
        return list(cls)


class InclusionInfo(Enum):
    INCLUDED = "included"
    NOT_INCLUDED = "not included"
    NOT_APPLICABLE = "not applicable"
    NOT_ENOUGH_INFORMATION = "not enough information"


class ExclusionInfo(Enum):
    EXCLUDED = "excluded"
    NOT_EXCLUDED = "not excluded"
    NOT_APPLICABLE = "not applicable"
    NOT_ENOUGH_INFORMATION = "not enough information"


class PatientExtractedInfo(CoreModel):
    summary: str
    conditions: List[str]

    @classmethod
    def from_dict(cls, data: dict) -> "PatientExtractedInfo":
        return cls(**data)


class MatchingResult(CoreModel):
    inclusion: dict = Field(default_factory=dict)
    exclusion: dict = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "MatchingResult":
        return cls(**data)

    def __getitem__(self, criterion_type: Union[str, Criteria]) -> dict:
        if isinstance(criterion_type, str):
            criterion_type = Criteria(criterion_type.lower())

        if criterion_type == Criteria.INCLUSION:
            return self.inclusion
        elif criterion_type == Criteria.EXCLUSION:
            return self.exclusion
        else:
            raise KeyError(f"Invalid criterion type: {criterion_type}")


class RankingResult(CoreModel):
    relevance_explanation: str
    relevance_score_R: float
    eligibility_explanation: str
    eligibility_score_E: float

    @classmethod
    def from_dict(cls, data: dict) -> "RankingResult":
        return cls(**data)


class ScoreResult(CoreModel):
    matching_score: float
    agg_score: float