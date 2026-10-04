"""Regression test for detections beginning at character zero."""

from types import SimpleNamespace

from ablation_harness.tools.registry import PiiCodexTool


class Analyzer:
    def analyze_item(self, text):
        detection = SimpleNamespace(start=0, end=5, score=0.9)
        risk = SimpleNamespace(pii_type_detected="NAME", risk_level="high")
        finding = SimpleNamespace(detection=detection, risk_assessment=risk)
        return SimpleNamespace(
            analysis=[finding], sanitized_text="[NAME] Smith", risk_score_mean=0.9
        )


def test_detection_at_offset_zero_keeps_text():
    result = PiiCodexTool(Analyzer()).execute(text="Alice Smith")
    assert result["detections"][0]["text"] == "Alice"
    assert result["sanitized_text_context"] == "[NAME] Smith"
