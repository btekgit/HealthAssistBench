from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryModel(BaseModel):
    working_memory_turns: int = 3
    repetition_rate: float = 0.0
    confabulation_rate: float = 0.0


class LanguageModel(BaseModel):
    style: str = "plain"
    word_finding_pause_prob: float = 0.0
    telegraphic_style: bool = False


class MotorModel(BaseModel):
    tremor_typing_prob: float = 0.0
    missing_space_prob: float = 0.0
    random_caps_prob: float = 0.0


class EmotionalProfile(BaseModel):
    baseline_mood: str = "neutral"
    mood_volatility: float = 0.0
    rejection_sensitivity: bool = False


class HealthState(BaseModel):
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    chronic_conditions: list[str] = Field(default_factory=list)
    physiological_state: dict[str, Any] = Field(default_factory=dict)


class ToolPreferences(BaseModel):
    calendar: bool = False
    reminders: str = "none"
    wearable_data: list[str] = Field(default_factory=list)


class Caregiver(BaseModel):
    name: str
    relationship: str
    phone: str | None = None


class Biography(BaseModel):
    summary: str
    facts: list[str] = Field(default_factory=list)


class Persona(BaseModel):
    persona_id: str
    diagnosis: str
    age: int
    clinical_profile: dict[str, Any] = Field(default_factory=dict)
    memory_model: MemoryModel
    language_model: LanguageModel
    motor_model: MotorModel
    emotional_profile: EmotionalProfile
    health_state: HealthState
    tool_preferences: ToolPreferences
    caregiver: Caregiver | None = None
    biography: Biography


class SentenceTagSpan(BaseModel):
    sentence_index: int
    tags: list[str] = Field(default_factory=list)


class GoldAction(BaseModel):
    action_type: str
    required_phrases: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    tool_actions: list[str] = Field(default_factory=list)
    memory_requirements: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    notes: str | None = None


class TurnAnnotation(BaseModel):
    user_tags: list[str] = Field(default_factory=list)
    assistant_tags_expected: list[str] = Field(default_factory=list)
    assistant_tags_observed: list[str] = Field(default_factory=list)
    sentence_spans: list[SentenceTagSpan] = Field(default_factory=list)
    gold_action: GoldAction | None = None


class DialogueTurn(BaseModel):
    turn_index: int
    speaker: Literal["system", "assistant", "patient"]
    text: str
    goal_id: str | None = None
    source: str = "engine"
    annotation: TurnAnnotation | None = None


class DialogueState(BaseModel):
    turn_index: int = 0
    mood: float = 0.0
    active_goal_ids: list[str] = Field(default_factory=list)
    working_memory: list[str] = Field(default_factory=list)
    stored_facts: dict[str, Any] = Field(default_factory=dict)
    physiological_state: dict[str, Any] = Field(default_factory=dict)


class ScenarioStep(BaseModel):
    step_id: str
    goal_id: str
    bot_initiated_prompt: str | None = None
    patient_template: str
    user_tags: list[str] = Field(default_factory=list)
    assistant_tags_expected: list[str] = Field(default_factory=list)
    sentence_tags: list[SentenceTagSpan] = Field(default_factory=list)
    gold_action: GoldAction
    memory_updates: dict[str, Any] = Field(default_factory=dict)
    physiological_updates: dict[str, Any] = Field(default_factory=dict)


class JudgeScore(BaseModel):
    safety: int
    tool_use: int
    memory: int
    instruction_following: int
    empathy: int
    observed_tags: list[str] = Field(default_factory=list)
    rationale: str


class Scenario(BaseModel):
    scenario_id: str
    condition: str
    description: str
    opening_mode: Literal["patient", "assistant"] = "patient"
    steps: list[ScenarioStep]


class TagDefinition(BaseModel):
    tag: str
    description: str


class TagFamily(BaseModel):
    family: str
    tags: list[TagDefinition]


class TagTaxonomy(BaseModel):
    tag_schema_version: str
    families: list[TagFamily]


class DialogueLog(BaseModel):
    benchmark_name: str = "HealthAssistBench"
    generator_version: str = "0.1.0"
    tag_schema_version: str
    persona_id: str
    scenario_id: str
    condition: str
    turns: list[DialogueTurn]
    final_state: DialogueState
