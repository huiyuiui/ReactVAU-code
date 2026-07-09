"""Prompt definitions used by the current ReactVAU pipeline."""

GRID_PROMPT_DETAIL = (
    "Analyze the 2x2 video grid. Is there any anomalous behavior, "
    "safety hazard, or event that disrupts the normal scene? Answer:"
)

PALIGEMMA_DESCRIBE_PROMPT = (
    "Briefly describe the main activity or event happening in this 2x2 "
    "video grid in one sentence."
)

STREAMFOREST_SCORE_GUIDED_PROMPT_SKEPTICAL = """A detection module has flagged this timestamp with {score_pct}% confidence.
{anomaly_context}
Note: This detector frequently produces false alarms from camera motion, lighting changes, and normal fast movements. Many of these alerts turn out to be false alarms.

You have been monitoring this video stream from the beginning and maintain continuous visual memory. Your task is to independently VERIFY whether this is a true anomaly or a false alarm.

Carefully assess:
1. What is actually happening in the current scene?
2. Compare with the normal and abnormal patterns you have observed. Is this genuinely different?
3. Is there clear evidence of anomalous behavior, such as violence, accidents, safety hazards, or events disrupting normal activity?

Do not rely solely on the detection confidence. Only answer 'Yes' if you see clear visual evidence of a genuine anomaly.

Is there a confirmed anomaly happening right now? Answer 'Yes' or 'No'."""


def get_grid_prompt(add_special_tokens: bool = False, style: str = "detail") -> str:
    """Return the PaliGemma grid prompt used by current train/eval scripts."""
    if style != "detail":
        raise ValueError(
            "ReactVAU currently keeps only the 'detail' PaliGemma prompt style."
        )

    if add_special_tokens:
        return f"<image>{GRID_PROMPT_DETAIL}"
    return GRID_PROMPT_DETAIL


def get_sf_prompt(
    trigger_mode: str,
    sf_scoring_method: str,
    prompt_style: str = "skeptical",
) -> str:
    """Return the StreamForest prompt used by current ReactVAU VAD evaluation."""
    if (
        trigger_mode == "score"
        and sf_scoring_method == "binary"
        and prompt_style == "skeptical"
    ):
        return STREAMFOREST_SCORE_GUIDED_PROMPT_SKEPTICAL

    raise ValueError(
        "ReactVAU currently keeps only trigger_mode='score', "
        "sf_scoring_method='binary', and sf_prompt_style='skeptical'."
    )
