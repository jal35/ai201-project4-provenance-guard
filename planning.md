## Detection Signals
**Signal 1**
A Groq LLM that looks at the overall flow, word choices, and vibe of the text to see if it reads like an AI wrote it
**Signal 2**
Python calculations that measure sentence length variance and punctuation density
**Output**
Both signals output a decimal score between 0.0 (Human) and 1.0 (AI)
**How they combine**
A weighted average that leans slightly heavier on the LLM's stylistic judgment
Final Confidence Score = (Groq Score x 0.6) + (Heuristic Score x 0.4)

## Uncertainty Representation
**Confidence Score of 0.6**
It represents genuine system uncertainty where the two detection signals are in conflict. Because a false positive (falsely accusing a human) is worse than a false negative on a creative platform, a score of 0.6 serves as a safety buffer. It ensures the system defaults to a neutral "Uncertain" label rather than issuing an incorrect AI penalty
**How will you map raw signal outputs to a calibrated score?**
Both the Groq LLM and the Python heuristics will be coded to return a raw value between 0.0 (completely human) and 1.0 (completely AI). We will map these to a single calibrated score using a weighted average that favors the holistic semantic judgment of the LLM
Calibrated \ Score = (Groq\ Score \times 0.6) + (Heuristic\ Score \times 0.4)
**What threshold separates "likely AI" from "uncertain" from "likely human"?**
0.00 to 0.35 (Likely Human): Highly variable sentence structures and natural formatting. 
0.36 to 0.75 (Uncertain): Conflicting or borderline metrics; safely padded up to 0.75 to protect human writers from false flags.  
0.76 to 1.00 (Likely AI): High structural uniformity and standard machine-generated patterns 
## Transparency label design
**High-Confidence Human**
This content heavily exhibits structural and stylistic markers unique to human writing.

**High-Confidence AI** 
Automated Content Flag: Our system detected strong semantic patterns and uniform structures typical of AI-generated text

**Uncertain** 
Unverified Attribution: Signals are highly conflicting. This writing contains elements matching both human variance and automated consistency

## Appeals Workflow
**Who can submit an appeal?**
Any registered platform creator whose text submission was processed by the system and who wants to contest the final classification result
**What information do they provide?**
They must provide the unique content_id generated during their original submission.  They must provide a text string explaining their defense (creator_reasoning), such as detailing their personal writing style or background context
**What does the system do when an appeal is received?**
Status Changes: The system updates the submission's status field from "classified" to "under review". Logging: The system appends the appeal timestamp and the verbatim creator_reasoning string directly into the structured audit log, locking it next to the original detection metrics
**What would a human reviewer see when they open the appeal queue?**
A consolidated dashboard or JSON array showing all submissions marked as "under review". For each entry, the reviewer will see the content_id, the original attribution verdict, the final confidence score, the individual signal breakdowns (Groq vs. Heuristics), and the raw creator_reasoning text to make a manual override decision

## Anticipated edge cases
**Non-Native English Speakers Writing Formal Essays**
Writers who are non-native English speakers usually lean on highly structured, gramatically perfect, and standard sentence frameworks to ensure clarity. Because their writing can lack natural, chaotic human idioms and displays uniform setence lengths, the pure Python stylometric heuristics will misinterpret this structural consistency as a robotic, artificially inflating the AI confidence score
**Technical Documentation**
Content like code explanations, step-by-step installation guides, or API documentation naturally relies on repetitive technical phrases, predictable vocabulary, and rigid sentence structure patterns. Both the Groq LLM and the stylometric calculations will struggle because informational technical writing inherently mimics the clean, uniform, and low-variance style that LLMs use by default, likely triggering a false positive

## Architecture

=== FLOW 1: CONTENT SUBMISSION ===
[User Text] ──► [POST /submit] ──► [Groq LLM (Signal 1)] ───────► [0.0 - 1.0] ──┐
                         │                                                      ▼
                         └────────► [Python Math Heuristics (Sig 2)] ──► [0.0 - 1.0] ──┼─► [Scoring Engine]
                                                                                            │
                                                                                            ▼
[API JSON Response] ◄── [Transparency Label] ◄── [Audit Log Entry Written] ◄── [Combined Score 0.0-1.0]

=== FLOW 2: APPEALS WORKFLOW ===
[creator_reasoning] ──► [POST /appeal] ──► [Database/Log] ──► Updates status to "under review"
**Narrative**
When text is submitted to POST /submit, it is processed concurrently by a Groq LLM semantic signal and a pure Python stylometric signal. These individual scores are aggregated into a weighted confidence metric, mapped to a non-technical transparency label, logged to a structured database, and returned as JSON. If a user hits POST /appeal, the backend instantly appends their reasoning and flips the content's system log status to "under review"

## AI tool plan
**M3**
Feed the AI our ## Detection Signals and ASCII diagram. Ask for a Flask skeleton with a blank POST /submit route, a GET /log endpoint, and a function handling the Groq API call. Test the Groq function standalone with basic text strings in the terminal to verify it spits out a raw 0.0–1.0 score
**M4**
Feed the AI our ## Uncertainty Representation rules. Ask for a pure Python function calculating sentence length variance and punctuation density, plus the weighted average formula to blend the signals. Test it against the 4 project sample texts to make sure scores scale dynamically across the whole 0–1 range
**M5**
Feed the AI our labels and appeals specs. Ask for the conditional block routing scores to our 3 exact label texts, the POST /appeal logger updating status to "under review", and Flask-Limiter setup. Test by hitting all 3 score tiers, submitting a mock appeal, and spamming the endpoint to force a 429 error