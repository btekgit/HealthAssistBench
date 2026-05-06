from __future__ import annotations

from healthassistbench.schemas import DialogueLog


def render_dialogue(log: DialogueLog) -> str:
    lines = [
        f"Benchmark: {log.benchmark_name}",
        f"Persona: {log.persona_id}",
        f"Scenario: {log.scenario_id}",
        f"Tag schema: {log.tag_schema_version}",
        "",
    ]
    for turn in log.turns:
        lines.append(f"[{turn.turn_index}] {turn.speaker.upper()}: {turn.text}")
        if turn.annotation:
            if turn.annotation.user_tags:
                lines.append(f"  user_tags: {', '.join(turn.annotation.user_tags)}")
            if turn.annotation.assistant_tags_expected:
                lines.append(
                    "  assistant_tags_expected: "
                    + ", ".join(turn.annotation.assistant_tags_expected)
                )
            if turn.annotation.sentence_spans:
                for span in turn.annotation.sentence_spans:
                    lines.append(
                        f"  sentence[{span.sentence_index}]: {', '.join(span.tags)}"
                    )
            if turn.annotation.gold_action:
                lines.append(
                    f"  gold_action: {turn.annotation.gold_action.action_type}"
                )
                if turn.annotation.gold_action.required_phrases:
                    lines.append(
                        "  required_phrases: "
                        + ", ".join(turn.annotation.gold_action.required_phrases)
                    )
                if turn.annotation.gold_action.tool_actions:
                    lines.append(
                        "  tool_actions: "
                        + ", ".join(turn.annotation.gold_action.tool_actions)
                    )
    return "\n".join(lines)
