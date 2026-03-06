"""
Prepare CPT (Continued Pre-Training) data.

Takes the raw CPT corpus and formats it for causal language modeling.
Each script becomes a training document with special tokens marking
boundaries. Also includes the grammar documentation.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_cpt_data(
    cpt_corpus: str = "data/cpt_corpus.txt",
    grammar_file: str = "dsl/grammar.lark",
    output_file: str = "data/cpt_train.jsonl",
    max_seq_len: int = 4096,
):
    """
    Format CPT data for Unsloth/HuggingFace training.

    Creates JSONL with {"text": "..."} entries:
    1. Grammar documentation (teaches syntax rules)
    2. Annotated DSL scripts (teaches structure patterns)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    documents = []

    # 1. Grammar documentation
    grammar_path = Path(grammar_file)
    if grammar_path.exists():
        grammar_text = grammar_path.read_text(encoding="utf-8")
        grammar_doc = (
            "# STAAD Pro DSL Grammar Reference\n\n"
            "The following is the formal EBNF grammar for STAAD Pro structural "
            "engineering scripts. The grammar defines the syntax for specifying "
            "joints (node coordinates), members (element connectivity), sections "
            "(steel profiles), material properties, supports, loads, and analysis "
            "commands.\n\n"
            "```ebnf\n"
            f"{grammar_text}\n"
            "```\n\n"
            "## Key Syntax Rules:\n"
            "- Scripts begin with `STAAD SPACE` and end with `FINISH`\n"
            "- `JOINT COORDINATES` block defines node positions as: ID X Y Z\n"
            "- `MEMBER INCIDENCES` block defines elements as: ID start_joint end_joint\n"
            "- Sections are assigned with `MEMBER PROPERTY` blocks\n"
            "- Supports use `SUPPORTS` block with FIXED or PINNED types\n"
            "- Loads defined in numbered `LOADING` cases\n"
            "- Units: METER KN is standard\n"
        )
        documents.append(grammar_doc)

        # Add it 3x for emphasis (small document)
        documents.append(grammar_doc)
        documents.append(grammar_doc)

    # 2. Annotated DSL scripts from CPT corpus
    corpus_path = Path(cpt_corpus)
    if corpus_path.exists():
        corpus_text = corpus_path.read_text(encoding="utf-8")
        scripts = corpus_text.strip().split("\n\n")

        for script in scripts:
            script = script.strip()
            if not script or len(script) < 50:
                continue

            # Add annotations around the script
            annotated = (
                "# STAAD Pro Structural Engineering Script\n\n"
                "The following is a valid STAAD Pro script defining a steel structure "
                "with joints, members, sections, supports, and loads.\n\n"
                f"```staad\n{script}\n```\n"
            )

            # Only include if within sequence length (rough char estimate)
            if len(annotated) < max_seq_len * 4:  # ~4 chars per token
                documents.append(annotated)

    # 3. Write JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")

    print(f"CPT data prepared: {len(documents)} documents → {output_file}")
    return {"count": len(documents), "output": output_file}


if __name__ == "__main__":
    prepare_cpt_data()
