from __future__ import annotations

import argparse
from pathlib import Path

from healthassistbench.adapters.ollama import OllamaChatAdapter
from healthassistbench.dialogue import DialogueEngine
from healthassistbench.evaluator.judge import score_dialogue_log
from healthassistbench.loader import load_persona, load_scenario, load_tag_taxonomy
from healthassistbench.rendering.text import render_dialogue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a HealthAssistBench demo dialogue.")
    parser.add_argument(
        "--persona",
        default="configs/personas/parkinsons_retired_teacher.yaml",
        help="Path to persona YAML.",
    )
    parser.add_argument(
        "--scenario",
        default="configs/scenarios/parkinsons_morning_checkin.yaml",
        help="Path to scenario YAML.",
    )
    parser.add_argument(
        "--taxonomy",
        default="configs/taxonomy/tag_schema_v0_1_0.yaml",
        help="Path to tag taxonomy YAML.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Optional Ollama model name. If omitted, no live assistant call is made.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score assistant turns against scenario gold actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    persona = load_persona(args.persona)
    scenario = load_scenario(args.scenario)
    taxonomy = load_tag_taxonomy(args.taxonomy)

    adapter = None
    if args.ollama_model:
        adapter = OllamaChatAdapter(
            model=args.ollama_model,
            host=args.ollama_host,
            system_prompt=(
                "You are an assistive health chatbot helping a chronic patient. "
                "Respond supportively, safely, and briefly."
            ),
        )

    engine = DialogueEngine(
        persona=persona,
        scenario=scenario,
        tag_schema_version=taxonomy.tag_schema_version,
        assistant_adapter=adapter,
    )
    log = engine.run()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    stem = f"{persona.persona_id}_{scenario.scenario_id}"
    (output_dir / f"{stem}.json").write_text(
        log.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{stem}.txt").write_text(render_dialogue(log), encoding="utf-8")

    print(render_dialogue(log))
    if args.score:
        print()
        print(score_dialogue_log(log).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
