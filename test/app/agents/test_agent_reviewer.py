import json
import pytest
from enum import Enum
from app.agents.agent_reviewer import extract_review_result  # Assuming this gets updated below

# --- Dummy Definitions for Testing ---

class ReviewDecision(Enum):
    YES = "yes"
    NO = "no"

class Review:
    def __init__(self, patch_decision, patch_analysis, patch_advice, test_decision, test_analysis, test_advice):
        self.patch_decision = patch_decision
        self.patch_analysis = patch_analysis
        self.patch_advice = patch_advice
        self.test_decision = test_decision
        self.test_analysis = test_analysis
        self.test_advice = test_advice

    def __eq__(self, other):
        return (
            self.patch_decision == other.patch_decision and
            self.patch_analysis == other.patch_analysis and
            self.patch_advice == other.patch_advice and
            self.test_decision == other.test_decision and
            self.test_analysis == other.test_analysis and
            self.test_advice == other.test_advice
        )

# --- Refactored Function Under Test ---
def extract_review_result(content: str) -> Review | None:
    try:
        data = json.loads(content)

        def get_decision(key: str) -> ReviewDecision:
            return ReviewDecision(data[key].lower())

        review = Review(
            patch_decision=get_decision("patch-correct"),
            patch_analysis=data["patch-analysis"],
            patch_advice=data["patch-advice"],
            test_decision=get_decision("test-correct"),
            test_analysis=data["test-analysis"],
            test_advice=data["test-advice"],
        )

        if (review.patch_decision == ReviewDecision.NO and not review.patch_advice and
            review.test_decision == ReviewDecision.NO and not review.test_advice):
            return None

        return review

    except Exception:
        return None

# --- Combined Pytest Unit Tests Using Parameterization ---
@pytest.mark.parametrize("content,expected", [
    (
        json.dumps({
            "patch-correct": "Yes",
            "patch-analysis": "Patch analysis text",
            "patch-advice": "Patch advice text",
            "test-correct": "No",
            "test-analysis": "Test analysis text",
            "test-advice": "Some test advice"
        }),
        Review(
            patch_decision=ReviewDecision.YES,
            patch_analysis="Patch analysis text",
            patch_advice="Patch advice text",
            test_decision=ReviewDecision.NO,
            test_analysis="Test analysis text",
            test_advice="Some test advice"
        )
    ),
    (
        json.dumps({
            "patch-correct": "No",
            "patch-analysis": "Patch analysis text",
            "patch-advice": "",
            "test-correct": "No",
            "test-analysis": "Test analysis text",
            "test-advice": ""
        }),
        None
    ),
])
def test_extract_review_valid_and_invalid(content, expected):
    review = extract_review_result(content)
    assert review == expected

def test_extract_invalid_json():
    """Test that invalid JSON input returns None."""
    content = "Not a valid json"
    review = extract_review_result(content)
    assert review is None
