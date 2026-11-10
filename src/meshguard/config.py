from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from meshguard.agent.llm.base import Effort
from meshguard.response.policy import PolicyConfig

type Provider = Literal["anthropic", "openai", "heuristic"]


class LLMSettings(BaseModel):
    provider: Provider = "anthropic"
    model: str = "claude-opus-5-5"
    effort: Effort = "medium"
    base_url: str = "http://localhost:11434/v1"
    max_turns: int = Field(14, ge=2, le=50)
    timeout_seconds: int = Field(300, ge=10)
    language: str = "Russian"


class DetectorSettings(BaseModel):
    training_hours: float = Field(24, gt=0)
    target_fpr: float = Field(0.002, gt=0, lt=0.5)
    seed: int = 7


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MESHGUARD_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    llm: LLMSettings = LLMSettings()
    detector: DetectorSettings = DetectorSettings()
    policy: PolicyConfig = PolicyConfig()
    artifacts_dir: Path = Path("artifacts")
    var_dir: Path = Path("var")

    @property
    def detector_path(self) -> Path:
        return self.artifacts_dir / "detector.joblib"

    @property
    def incidents_dir(self) -> Path:
        return self.var_dir / "incidents"

    @property
    def outbox_dir(self) -> Path:
        return self.var_dir / "outbox"
