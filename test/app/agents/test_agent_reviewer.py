import json
import pytest
from enum import Enum
from app.agents.agent_reviewer import extract_review_result

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

# --- Function Under Test ---
def extract_review_result(content: str) -> Review | None:
    try:
        data = json.loads(content)

        review = Review(
            patch_decision=ReviewDecision(data["patch-correct"].lower()),
            patch_analysis=data["patch-analysis"],
            patch_advice=data["patch-advice"],
            test_decision=ReviewDecision(data["test-correct"].lower()),
            test_analysis=data["test-analysis"],
            test_advice=data["test-advice"],
        )

        if (
            (review.patch_decision == ReviewDecision.NO) and not review.patch_advice
        ) and ((review.test_decision == ReviewDecision.NO) and not review.test_advice):
            return None

        return review

    except Exception:
        return None

# --- Pytest Unit Tests ---

def test_extract_valid_review():
    """Test that valid JSON input returns a proper Review instance."""
    content = json.dumps({
        "patch-correct": "Yes",
        "patch-analysis": "Patch analysis text",
        "patch-advice": "Patch advice text",
        "test-correct": "No",
        "test-analysis": "Test analysis text",
        "test-advice": "Some test advice"
    })

    review = extract_review_result(content)
    expected_review = Review(
        patch_decision=ReviewDecision.YES,
        patch_analysis="Patch analysis text",
        patch_advice="Patch advice text",
        test_decision=ReviewDecision.NO,
        test_analysis="Test analysis text",
        test_advice="Some test advice"
    )
    assert review == expected_review

def test_extract_invalid_due_to_empty_advice():
    """
    Test that when both patch and test decisions are 'No' and their corresponding advice are empty,
    the function returns None.
    """
    content = json.dumps({
        "patch-correct": "No",
        "patch-analysis": "Patch analysis text",
        "patch-advice": "",
        "test-correct": "No",
        "test-analysis": "Test analysis text",
        "test-advice": ""
    })

    review = extract_review_result(content)
    assert review is None

def test_extract_invalid_json():
    """Test that invalid JSON input returns None."""
    content = "Not a valid json"
    review = extract_review_result(content)
    assert review is None
