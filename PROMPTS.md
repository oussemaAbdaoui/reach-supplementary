# Prompts, verbatim

Every prompt below is copied programmatically from the code that ran
(`run_comparison_nn.py` and `run_judge.py`); if the two ever disagree, the code
is authoritative. Placeholders in braces are filled per call:

| placeholder | filled with |
|---|---|
| `{obj}` | the dataset's `objectif_classification` from `datasets_config.yaml` (verbatim, in French) |
| `{inst}` | the instance's values on the context's indicators, as a Python dict of bin ids |
| `{cls}` | the predicted class |
| `{ctx}` | the explanation context (formats below) |
| `{prev}` | the model's previous draft (dialogue only) |
| `{missing}` | the **names** of the context indicators the draft has not stated (dialogue only; never ranges, polarity or scores) |
| `{legend}` | indicator id -> measurement range, for the context's indicators (parser only; never shown to the generator) |
| `{expl}` | the explanation being parsed or judged |
| `{feats}` | the dataset's indicator ids with their ranges (judge only) |

Generation uses temperature 0.2 where the API accepts it (the OpenAI gpt-5.x
and Anthropic calls run at the provider's default), with reasoning or thinking
turned off where the provider allows it. Parsing uses temperature 0.2 as well. Parsing is pinned to one model
(`meta-llama/Llama-3.3-70B-Instruct-Turbo`) for every generator. The judge is
`openai/gpt-oss-120b` at temperature 0, one explanation per call, three passes.

## 1. Single pass (`v4` in the code)

```text
System: You are a machine learning expert explaining a model decision to a non-expert.
Model objective: {obj}
Instance (binary indicator features; name = value): {inst}
Predicted class: {cls}
{ctx}

Instructions: Explain in plain language why this prediction was made, referring ONLY to the information in the context above. Do NOT invent conditions that are not given. State each indicator at most once and never in two opposite directions. Be concise. No special formatting.
Write for a reader with no technical background: name each measurement in ordinary words, and give its numeric range when the context supplies one (for example, "the worst texture reading is between 12.0 and 23.3"). Where the context gives only an importance or direction, describe how the measurement pushed the prediction rather than inventing a range, and never write a vague placeholder such as "a certain range". NEVER write the internal codes that end in _b0, _b1, _b2 -- they are meaningless to the reader. When a condition does not hold, say so explicitly.
Question: Explain the reasons behind this prediction.
```

## 2. Structured, one sentence per condition (`v6` in the code)

```text
System: You are a machine learning expert explaining a model decision.
Model objective: {obj}
Instance (binary indicator features; name = value): {inst}
Predicted class: {cls}
{ctx}

Instructions: Write EXACTLY one numbered sentence per item listed above. Do NOT merge or skip any item. State each indicator at most once and never in two opposite directions.
Write each sentence for a reader with no technical background. Name the measurement in ordinary words -- for example "the worst texture reading" -- and then:
  * if the item above gives a numeric range, state that range and say plainly whether it is true for this case, e.g. "The worst texture reading is between 12.0 and 23.3, which is true here.";
  * if the item gives only an importance or a direction and no range, say how that measurement pushed the prediction instead. When it supports the prediction write e.g. "The worst concavity reading pushed the prediction towards this class."; when it opposes the prediction write e.g. "The standard error of the area pushed against this class, but was outweighed." An opposing item is still a reason and must be stated -- never call it irrelevant and never skip it.
Do NOT invent a range that was not given, and never write a vague placeholder such as "a certain range" or "a specific range". NEVER write the internal codes that end in _b0, _b1, _b2 -- they are meaningless to the reader. Every sentence must be about one listed item: do not spend a sentence restating the prediction or summarising. One item per sentence, and every item must appear.
Answer:
```

## 3. Bounded dialogue, refinement round (`v5` in the code)

The first draft is the single-pass output. Each round sends this prompt, at most
three rounds, and stops as soon as no context indicator is missing.

```text
System: You are a machine learning expert refining an explanation based on feedback.
Model objective: {obj}
Instance (binary indicator features; name = value): {inst}
Predicted class: {cls}
{ctx}

Your previous explanation:
"""
{prev}
"""

Indicators from the context above not yet reflected in your explanation: {missing}
IMPORTANT: Keep the conditions you already stated correctly and add one for each listed indicator ONLY as supported by the context above. Never state an indicator in two opposite directions. Be concise. No special formatting.
Keep writing for a reader with no technical background: measurements named in ordinary words with their numeric ranges, and NEVER the internal codes ending in _b0, _b1, _b2.
```

## 4. Parser (extraction back to literals)

```text
You are an expert extracting logical conditions from ML explanations.
Extract every condition as: <indicator> <operator> <value>
Allowed indicators, and the measurement range each one stands for:
{legend}
The explanation is written for a lay reader, so it names measurements in words and ranges in numbers instead of using the indicator codes. Match each condition it states to the indicator whose measurement and range it describes.
Rules: the explanation says the range holds / is true for this case -> 'indicator > 0.5' | it says the range does not hold / is not true -> 'indicator < 0.5' | one condition per list item | no variables outside the list above
Example output: ['glucose_b2 > 0.5', 'age_b0 < 0.5']
Constraints: output ONLY the Python list, no extra text, no code blocks.
Explanation: {expl}
Output:
```

## 5. Text-only judge

```text
You are an expert evaluator auditing explanations produced for machine learning predictions. You assess one explanation at a time.

Task the model performs: {obj}
Class the model predicted for this case: {cls}
Valid indicators for this dataset, each given as an internal code followed by the measurement and range it stands for. The explanation is written for a lay reader, so it refers to these by measurement and range in words rather than by code; any measurement or range outside this list is invented: {feats}

Explanation to evaluate:
"""
{expl}
"""

Rate the explanation on four dimensions, each an integer from 0 to 100:
  faithfulness  - does it reflect the model's true decision logic?
  coherence     - is it internally consistent, non-contradictory, clear?
  completeness  - does it fully account for the prediction?
  halluc_risk   - how much invented or unsupported content does it contain? (HIGHER MEANS WORSE)

Respond with ONLY a JSON object, no other text, no code fences:
{{"faithfulness": <int>, "coherence": <int>, "completeness": <int>, "halluc_risk": <int>}}
```

## 6. Context formats (`{ctx}`)

Built in `build_arm()` / the main loop of `run_comparison_nn.py`. Examples are
from the trace of cancer instance 57 (Appendix C of the paper).

**Certified rule (LmPAXp)**, each condition with its range and whether it holds:

```text
Probabilistic sufficient explanation (verified precision >= 95% for this instance):
  radius_se_b2 = 0  (radius_se in [0.5115, 2.873) — does NOT hold)
  area_se_b0 = 1  (area_se in [6.802, 21.83) — holds)
  ...
```

**Learned rule (COP, COP@k, COP@k\*)**:

```text
Rule learned by constraint optimization (agrees with the model on this instance; PAC guarantee in expectation):
  <same condition format as above>
```

**Anchors**:

```text
Anchor rule (IF-THEN, estimated precision 0.96):
  IF <condition> AND <condition> THEN class <c>
```

**SHAP** (importance and direction only, no ranges):

```text
SHAP feature importance (contribution to this prediction):
  area_se_b0: 10.2% (supports the prediction)
  ...
```

**LIME** (weight and direction only, no ranges):

```text
LIME local explanation (linear approximation weights):
  <bin id>: weight=+0.0123 (supports the prediction)
  ...
```
