"""The VLM and ReAct agent adapters."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

transformers = pytest.importorskip("transformers")


def test_vlm_model_runs_tiny_gemma():
    from PIL import Image
    from fakes import FakeProcessor, tiny_gemma
    from openmarkx.registry import get_model
    m = get_model("vlm", base="tiny").with_processor(FakeProcessor())
    out = m.run(tiny_gemma(), [{"image": Image.new("RGB", (28, 28)), "question": "what is shown"}], new_tokens=4)
    assert len(out) == 1 and isinstance(out[0], str)


def test_agent_parses_actions_and_stops_on_finish(monkeypatch):
    import openmarkx.models.agent as A
    from openmarkx.registry import get_model
    script = iter([["Thought: look\nAction: Search[Alpha]"], ["Thought: done\nAction: Finish[blue]"], ["The answer is blue."]])
    monkeypatch.setattr(A, "generate", lambda *a, **k: next(script))
    m = get_model("agent", base="llama-3.2-1b", max_steps=6, final_words=50)
    item = {"question": "What colour?", "paragraphs": {"Alpha": "Alpha is blue.", "Beta": "Beta is red."}}
    finals, transcripts = m.run(object(), [item], transcript=True)
    assert finals == ["The answer is blue."]
    obs = [t["content"] for t in transcripts[0] if t["role"] == "user" and t["content"].startswith("Observation:")]
    assert obs == ["Observation: [Alpha] Alpha is blue."]


def test_agent_matches_old_parse_action():
    from openmarkx.models.agent import parse_action
    S = ["Thought: x\nAction: Search[Paris]", "Action: Finish[ 42 ]", "Thought: none",
         "Action:Search[New\nYork] then Action: Finish[b]", "action: search[lower]"]
    assert [parse_action(s) for s in S] == [("Search", "Paris"), ("Finish", "42"), (None, None),
                                            ("Search", "New\nYork"), (None, None)]
