"""Visual scale for turbidity classes, styled after ADPC's Air4Laos PM2.5 scale
(https://air4laos.adpc.net) - same 7-class pastel palette and pill/badge pattern,
re-keyed to general NTU turbidity breakpoints. These NTU cut-offs are a general
reference scale for this dashboard, not an official Thai PCD standard.
"""

CLASSES = [
    {"max": 5, "label": "Excellent", "color": "#A2C0FC",
     "advice": "Clear water. No turbidity-related concern for recreational or aquatic use."},
    {"max": 20, "label": "Good", "color": "#96C3A0",
     "advice": "Slightly turbid. Normal condition for most rivers after light rain."},
    {"max": 50, "label": "Fair", "color": "#FDEF6A",
     "advice": "Moderately turbid. Worth monitoring if it persists outside the wet season."},
    {"max": 100, "label": "Moderate", "color": "#FBCA69",
     "advice": "Turbid water. Likely runoff/erosion influence; treat before drinking use."},
    {"max": 150, "label": "Poor", "color": "#F19D77",
     "advice": "Highly turbid. Aquatic habitat and water treatment load are affected."},
    {"max": 250, "label": "Very Poor", "color": "#EA6F71",
     "advice": "Severely turbid. Investigate upstream erosion, discharge, or dam release."},
    {"max": float("inf"), "label": "Severe", "color": "#AF6E6D",
     "advice": "Extreme turbidity. Flag for immediate follow-up with PCD field data."},
]


def classify(ntu: float) -> dict:
    if ntu is None:
        return {"max": None, "label": "Unavailable", "color": "#C9CED6", "advice": ""}
    for c in CLASSES:
        if ntu <= c["max"]:
            return c
    return CLASSES[-1]


def color_for(ntu: float) -> str:
    return classify(ntu)["color"]


def label_for(ntu: float) -> str:
    return classify(ntu)["label"]
