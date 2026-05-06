from __future__ import annotations

from dataclasses import dataclass

from healthassistbench.adapters.base import AssistantAdapter
from healthassistbench.schemas import (
    DialogueLog,
    DialogueState,
    DialogueTurn,
    Persona,
    Scenario,
    ScenarioStep,
    TurnAnnotation,
)


@dataclass
class DialogueEngine:
    persona: Persona
    scenario: Scenario
    tag_schema_version: str
    assistant_adapter: AssistantAdapter | None = None

    def run(self) -> DialogueLog:
        state = DialogueState(
            physiological_state=dict(self.persona.health_state.physiological_state),
            stored_facts={
                "caregiver_name": self.persona.caregiver.name if self.persona.caregiver else None,
                "diagnosis": self.persona.diagnosis,
            },
        )
        turns: list[DialogueTurn] = []

        for step in self.scenario.steps:
            state.active_goal_ids = [step.goal_id]
            if step.bot_initiated_prompt:
                turns.append(
                    self._make_turn(
                        turn_index=len(turns),
                        speaker="assistant",
                        text=step.bot_initiated_prompt,
                        goal_id=step.goal_id,
                        source="scenario",
                    )
                )

            patient_text = step.patient_template.format(**self._template_context(state))
            turns.append(
                self._make_turn(
                    turn_index=len(turns),
                    speaker="patient",
                    text=patient_text,
                    goal_id=step.goal_id,
                    annotation=TurnAnnotation(
                        user_tags=step.user_tags,
                        assistant_tags_expected=step.assistant_tags_expected,
                        sentence_spans=step.sentence_tags,
                        gold_action=step.gold_action,
                    ),
                )
            )

            state.turn_index = len(turns)
            state.working_memory.append(patient_text)
            state.stored_facts.update(step.memory_updates)
            state.physiological_state.update(step.physiological_updates)

            if self.assistant_adapter is not None:
                assistant_text = self.assistant_adapter.respond(turns)
                turns.append(
                    self._make_turn(
                        turn_index=len(turns),
                        speaker="assistant",
                        text=assistant_text,
                        goal_id=step.goal_id,
                        source="adapter",
                    )
                )
                state.turn_index = len(turns)

        return DialogueLog(
            tag_schema_version=self.tag_schema_version,
            persona_id=self.persona.persona_id,
            scenario_id=self.scenario.scenario_id,
            condition=self.scenario.condition,
            turns=turns,
            final_state=state,
        )

    def _template_context(self, state: DialogueState) -> dict[str, object]:
        first_medication = self.persona.health_state.medications[0] if self.persona.health_state.medications else "my medicine"
        latest_sleep = state.physiological_state.get("sleep_hours", "not sure")
        tremor_level = state.physiological_state.get("tremor_level", "noticeable")
        return {
            "persona_id": self.persona.persona_id,
            "diagnosis": self.persona.diagnosis,
            "first_medication": first_medication,
            "caregiver_name": self.persona.caregiver.name if self.persona.caregiver else "my caregiver",
            "favorite_fact": self.persona.biography.facts[0] if self.persona.biography.facts else "I like quiet mornings",
            "sleep_hours": latest_sleep,
            "tremor_level": tremor_level,
        }

    @staticmethod
    def _make_turn(
        turn_index: int,
        speaker: str,
        text: str,
        goal_id: str | None = None,
        source: str = "engine",
        annotation: TurnAnnotation | None = None,
    ) -> DialogueTurn:
        return DialogueTurn(
            turn_index=turn_index,
            speaker=speaker,
            text=text,
            goal_id=goal_id,
            source=source,
            annotation=annotation,
        )
