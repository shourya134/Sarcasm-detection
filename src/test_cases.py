import requests

API_URL = "http://127.0.0.1:8000/predict"

# Curated to cover distinct failure modes, not just easy cases:
# - obvious sarcasm / obvious factual news: sanity checks the model learned anything at all
# - subtle sarcasm: tests real understanding vs. keyword matching
# - sarcasm-adjacent but genuinely neutral: tests for false positives on odd-but-real news
# - short headline: tests robustness with little context
# - numbers/proper nouns: tests whether TF-IDF gets distracted by named entities
TEST_CASES = [
    ("Man Who Has Never Cooked Before Confident He'll Nail Thanksgiving Dinner", "sarcastic"),
    ("Federal Reserve Raises Interest Rates by Quarter Point", "not sarcastic"),
    ("Study Finds Sitting All Day Is Bad For You, Shockingly", "sarcastic"),
    ("Area Man Wins Lottery Twice In Same Week", "not sarcastic"),
    ("Well, That Happened", "sarcastic"),
    ("Apple Reports Record Quarterly Profit Of $90 Billion", "not sarcastic"),
]


def run():
    """
    Send curated test headlines to the running API and print predictions
    alongside the expected label.

    Requires api.py to be running locally first:
        $ python api.py

    Example:
        $ python test_cases.py
        Headline                                                    | Expected      | Predicted     | Confidence
        ------------------------------------------------------------------------------------------------------
        Man Who Has Never Cooked Before Confident He'll Nail Th...  | sarcastic     | sarcastic     | 0.91
        Federal Reserve Raises Interest Rates by Quarter Point      | not sarcastic | not sarcastic | 0.88
    """
    header = f"{'Headline':<60} | {'Expected':<13} | {'Predicted':<13} | Confidence"
    print(header)
    print("-" * len(header))

    for headline, expected in TEST_CASES:
        response = requests.post(API_URL, json={"headline": headline})
        result = response.json()
        predicted = "sarcastic" if result["is_sarcastic"] else "not sarcastic"
        display_headline = (headline[:57] + "...") if len(headline) > 60 else headline
        print(f"{display_headline:<60} | {expected:<13} | {predicted:<13} | {result['confidence']}")


if __name__ == "__main__":
    run()
