from pydantic import Field
from typing import List, Union
from enum import Enum

from service_platform.core.base_schema import CoreModel


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
