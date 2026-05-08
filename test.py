from openai import OpenAI
import httpx
import json
import math

# ── endpoints ────────────────────────────────────────────────────────────────
TOKENIZE_URL = "http://localhost:8000/tokenize"
MODEL = "Qwen/Qwen3.5-2B"

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # required by SDK but vLLM ignores it
)

# ── prompt ────────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """\
Analyze the text below and extract clause pairs that exhibit clear discourse relations.
 
VALID RELATIONS — use EXACTLY these names:
Circumstance, Solutionhood, Elaboration, Background, Enablement, Motivation,
Evidence, Justify, Volitional-Cause, Non-Volitional-Cause, Volitional-Result,
Non-Volitional-Result, Purpose, Antithesis, Concession, Condition, Otherwise,
Interpretation, Evaluation, Restatement, Summary, Sequence, Contrast
 
RELATION DEFINITIONS:
Relations are defined to hold between two non-overlapping text spans, here called the
nucleus and the satellite, denoted by N and S.Reader and Writer denoted by R and W. A relation definition consists of four fields:
1. Constraints on the Nucleus
2. Constraints on the Satellite
3. Constraints on the combination of Nucleus and Satellite
4. The Effect
- relation name: Circumstance
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: S presents a situation (not unrealized)
3. Constraints on the combination of Nucleus and Satellite: S sets a framework in the subject matter within which R is intended to interpret the situation presented in N
4. The Effect: R recognizes that the situation presented in S provides the framework for interpreting N
Locus of the effect: N and S
 
- relation name: Solutionhood
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: presents a problem
3. Constraints on the combination of Nucleus and Satellite: the situation presented in N is a solution to the problem stated in S
4. The Effect: R recognizes the situation presented in N as a solution to the problem presented in S
Locus of the effect: N and S
 
- relation name: Elaboration
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: S presents additional detail about the situation or some element of subject matter which is presented in N or inferentially accessible in N in one or more of the ways listed below. In the list, if N presents the first member of any pair, then S includes the second: 1. set: member / 2. abstract: instance / 3. whole: part / 4. process: step / 5. object: attribute / 6. generalization: specific
4. The Effect: R recognizes the situation presented in S as providing additional detail for N. R identifies the element of subject matter for which detail is provided.
Locus of the effect: N and S
 
- relation name: Background
1. Constraints on the Nucleus: R won't comprehend N sufficiently before reading text of S
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: S increases the ability of R to comprehend an element in N
4. The Effect: R's ability to comprehend N increases
Locus of the effect: N
 
- relation name: Enablement
1. Constraints on the Nucleus: presents R action (including accepting an offer), unrealized with respect to the context of N
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: R comprehending S increases R's potential ability to perform the action presented in N
4. The Effect: R's potential ability to perform the action presented in N increases
Locus of the effect: N
 
- relation name: Motivation
1. Constraints on the Nucleus: presents an action in which R is the actor (including accepting an offer), unrealized with respect to the context of N
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: comprehending S increases R's desire to perform action presented in N
4. The Effect: R's desire to perform action presented in N is increased
Locus of the effect: N
 
- relation name: Evidence
1. Constraints on the Nucleus: R might not believe N to a degree satisfactory to W
2. Constraints on the Satellite: R believes S or will find it credible
3. Constraints on the combination of Nucleus and Satellite: R's comprehending S increases R's belief of N
4. The Effect: R's belief of N is increased
Locus of the effect: N
 
- relation name: Justify
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: R's comprehending S increases R's readiness to accept W's right to present N
4. The Effect: R's readiness to accept W's right to present N is increased
Locus of the effect: N
 
- relation name: Volitional-Cause
1. Constraints on the Nucleus: presents a volitional action or else a situation that could have arisen from a volitional action
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: S presents a situation that could have caused the agent of the volitional action in N to perform that action; without the presentation of S, R might not regard the action as motivated or know the particular motivation; N is more central to W's purposes in putting forth the N-S combination than S is
4. The Effect: R recognizes the situation presented in S as a cause for the volitional action presented in N
Locus of the effect: N and S
 
- relation name: Non-Volitional-Cause
1. Constraints on the Nucleus: presents a situation that is not a volitional action
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: S presents a situation that, by means other than motivating a volitional action, caused the situation presented in N; without the presentation of S, R might not know the particular cause of the situation; a presentation of N is more central than S to W's purposes in putting forth the N-S combination
4. The Effect: R recognizes the situation presented in S as a cause of the situation presented in N
Locus of the effect: N and S
 
- relation name: Volitional-Result
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: presents a volitional action or a situation that could have arisen from a volitional action
3. Constraints on the combination of Nucleus and Satellite: N presents a situation that could have caused the situation presented in S; the situation presented in N is more central to W's purposes than is that presented in S
4. The Effect: R recognizes that the situation presented in N could be a cause for the action or situation presented in S
Locus of the effect: N and S
 
- relation name: Non-Volitional-Result
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: presents a situation that is not a volitional action
3. Constraints on the combination of Nucleus and Satellite: N presents a situation that caused the situation presented in S; presentation of N is more central to W's purposes in putting forth the N-S combination than is the presentation of S
4. The Effect: R recognizes that the situation presented in N could have caused the situation presented in S
Locus of the effect: N and S
 
- relation name: Purpose
1. Constraints on the Nucleus: presents an activity
2. Constraints on the Satellite: presents a situation that is unrealized
3. Constraints on the combination of Nucleus and Satellite: S presents a situation to be realized through the activity in N
4. The Effect: R recognizes that the activity in N is initiated in order to realize S
Locus of the effect: N and S
 
- relation name: Antithesis
1. Constraints on the Nucleus: W has positive regard for the situation presented in N
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: the situations presented in N and S are in contrast; because of an incompatibility that arises from the contrast, one cannot have positive regard for both the situations presented in N and S; comprehending S and the incompatibility between the situations presented in N and S increases R's positive regard for the situation presented in N
4. The Effect: R's positive regard for N is increased
Locus of the effect: N
 
- relation name: Concession
1. Constraints on the Nucleus: W has positive regard for the situation presented in N
2. Constraints on the Satellite: W is not claiming that the situation presented in S doesn't hold
3. Constraints on the combination of Nucleus and Satellite: W acknowledges a potential or apparent incompatibility between the situations presented in N and S; W regards the situations presented in N and S as compatible; recognizing the compatibility between the situations presented in N and S increases R's positive regard for the situation presented in N
4. The Effect: R's positive regard for the situation presented in N is increased
Locus of the effect: N and S
 
- relation name: Condition
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: S presents a hypothetical, future, or otherwise unrealized situation (relative to the situational context of S)
3. Constraints on the combination of Nucleus and Satellite: realization of the situation presented in N depends on realization of that presented in S
4. The Effect: R recognizes how the realization of the situation presented in N depends on the realization of the situation presented in S
Locus of the effect: N and S
 
- relation name: Otherwise
1. Constraints on the Nucleus: presents an unrealized situation
2. Constraints on the Satellite: presents an unrealized situation
3. Constraints on the combination of Nucleus and Satellite: realization of the situation presented in N prevents realization of the situation presented in S
4. The Effect: R recognizes the dependency relation of prevention between the realization of the situation presented in N and the realization of the situation presented in S
Locus of the effect: N and S
 
- relation name: Interpretation
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: S relates the situation presented in N to a framework of ideas not involved in N itself and not concerned with W's positive regard
4. The Effect: R recognizes that S relates the situation presented in N to a framework of ideas not involved in the knowledge presented in N itself
Locus of the effect: N and S
 
- relation name: Evaluation
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: S relates the situation in N to degree of W's positive regard toward the situation presented in N
4. The Effect: R recognizes that the situation presented in S assesses the situation presented in N and recognizes the value it assigns
Locus of the effect: N and S
 
- relation name: Restatement
1. Constraints on the Nucleus: none
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: S restates N, where S and N are of comparable bulk
4. The Effect: R recognizes S as a restatement of N
Locus of the effect: N and S
 
- relation name: Summary
1. Constraints on the Nucleus: N must be more than one unit
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: S presents a restatement of the content of N, that is shorter in bulk
4. The Effect: R recognizes S as a shorter restatement of N
Locus of the effect: N and S
 
- relation name: Sequence
1. Constraints on the Nucleus: multi-nuclear
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: a succession relationship between the situations is presented in the nuclei
4. The Effect: R recognizes the succession relationships among the nuclei
Locus of the effect: multiple nuclei
 
- relation name: Contrast
1. Constraints on the Nucleus: multi-nuclear
2. Constraints on the Satellite: none
3. Constraints on the combination of Nucleus and Satellite: no more than two nuclei; the situations presented in these two nuclei are (a) comprehended as the same in many respects (b) comprehended as differing in a few respects and (c) compared with respect to one or more of these differences
4. The Effect: R recognizes the comparability and the difference(s) yielded by the comparison is being made
Locus of the effect: multiple nuclei
 
Canonical Orders of Spans for Some Relations
Satellite Before Nucleus
Antithesis
Background
Concessive
Conditional
Justify
Solutionhood
Nucleus Before Satellite
Elaboration
Enablement
Evidence
Purpose
Restatement
 
 
TEXT TO ANALYZE:
{text}
 
STRICT RULES:
- Each clause must be grammatically complete
- "nucleus" and "satellite" must be exactly "clause1" or "clause2"
- Return each unique clause pair only once; do not duplicate the same pair or object
- Output ONLY a valid JSON array, no extra text
 
OUTPUT (JSON array only):
{{"clause1": "...", "clause2": "...", "relation": "RelationName", "nucleus": "clause1", "satellite": "clause2"}}
"""

# ── valid relations ───────────────────────────────────────────────────────────
RELATIONS_23 = {
    "circumstance", "solutionhood", "elaboration", "background",
    "enablement", "motivation", "evidence", "justify",
    "volitional-cause", "non-volitional-cause",
    "volitional-result", "non-volitional-result",
    "purpose", "antithesis", "concession", "condition", "otherwise",
    "interpretation", "evaluation", "restatement", "summary",
    "sequence", "contrast",
}

# ── tokenize ALL case/spacing variants to build logit_bias ───────────────────
def with_space_variants(relations: set[str]):
    """Yield each relation with and without leading/trailing space."""
    for r in relations:
        yield r
        yield f" {r}"
        yield f"{r} "


def tokenize(text: str) -> list[int]:
    """Return the token IDs for *text* from the vLLM tokenize endpoint."""
    response = httpx.post(
        TOKENIZE_URL,
        json={
            "model": MODEL,
            "prompt": text,
            "add_special_tokens": False,   # don't add BOS/EOS for a mid-text fragment
            "return_token_strs": False,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["tokens"]


# Build logit_bias: boost every token that appears in any relation variant.
# We preserve original casing so the bias applies to what the model generates.
logit_bias: dict[int, float] = {}
token_to_idx: dict[str, list[int]] = {}

# for variant in with_space_variants(RELATIONS_23):
#     tokens = tokenize(variant)
#     token_to_idx[variant] = tokens
#     for tok_id in tokens:
#         logit_bias[tok_id] = 10  # strong positive bias

# ── inference ─────────────────────────────────────────────────────────────────
# max_tokens must be large enough for the longest multi-token label.
# "Non-Volitional-Cause" tokenises to ~5 tokens; 10 is a safe ceiling.
MAX_LABEL_TOKENS = 10

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(
        text="Probably the most extreme case of Visitors Fever I have ever witnessed was a few summers ago. when I visited relatives in the Midwest.",
        )}],
    temperature=0,
    max_tokens=128,
    # frequency_penalty=0.4,
    # logprobs=True,
    # top_logprobs=5,
    # logit_bias=logit_bias,
)

raw_output = response.choices[0].message.content.strip()

try:
    parsed_output = json.loads(raw_output)
    if isinstance(parsed_output, list):
        deduped_output = []
        seen_items = set()
        for item in parsed_output:
            if isinstance(item, dict):
                item_key = (
                    item.get("clause1"),
                    item.get("clause2"),
                    item.get("relation"),
                    item.get("nucleus"),
                    item.get("satellite"),
                )
                if item_key in seen_items:
                    continue
                seen_items.add(item_key)
            deduped_output.append(item)
        print(f"Normalized model output : {json.dumps(deduped_output, ensure_ascii=False)}")
    else:
        print(f"Raw model output : {raw_output!r}")
except json.JSONDecodeError:
    print(f"Raw model output : {raw_output!r}")

# # ── token-level probabilities ─────────────────────────────────────────────────
# print("\nToken-level log-probabilities:")
# logs = response.choices[0].logprobs.content
# for token_logprob in logs:
#     prob = math.exp(float(token_logprob.logprob))
#     print(f"  token={token_logprob.token!r:25s}  logprob={token_logprob.logprob:.4f}  prob={prob:.4f}")

#     if token_logprob.top_logprobs:
#         print("  top alternatives:")
#         for alt in token_logprob.top_logprobs:
#             alt_prob = math.exp(float(alt.logprob))
#             print(f"    token={alt.token!r:25s}  logprob={alt.logprob:.4f}  prob={alt_prob:.4f}")
