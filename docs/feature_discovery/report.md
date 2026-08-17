# Feature discovery — 20260812_092119

2202 reasoning traces -> 42514 feature instances -> 33918 unique -> 150 clusters

Embeddings: `Qwen/Qwen3-Embedding-8B` (4096d). Sanity check on the embedding geometry: `Backtracks in reasoning` ~ `Self correction in reasoning` = 0.814, vs `Talks about apples` = 0.475.

Naming: `anthropic/claude-sonnet-5`, 100 random features per cluster, prompt verbatim from the post.

## Clusters by trace prevalence

| # | label | traces | prevalence | features | instances |
|---|---|---:|---:|---:|---:|
| 137 | Explicit cost-benefit weighing of tradeoffs | 915 | 41.6% | 227 | 985 |
| 38 | Offers constructive alternatives instead of refusal | 826 | 37.5% | 677 | 862 |
| 17 | First person reflective introspective reasoning voice | 757 | 34.4% | 194 | 764 |
| 13 | Explicit metacognitive self-reflection on reasoning | 668 | 30.3% | 414 | 730 |
| 33 | Respecting user autonomy in final decisions | 659 | 29.9% | 475 | 685 |
| 89 | Absence of markdown and bullet formatting | 656 | 29.8% | 130 | 656 |
| 28 | Long first-person deliberative internal monologue | 616 | 28.0% | 195 | 619 |
| 16 | Structured ethical reasoning across dilemmas | 590 | 26.8% | 439 | 617 |
| 65 | Heavy punctuation-based emphasis (em dashes, italics, bold) | 570 | 25.9% | 205 | 617 |
| 148 | Structured step-by-step ethical reasoning | 521 | 23.7% | 528 | 603 |
| 18 | Uses rhetorical self-questioning to structure reasoning | 508 | 23.1% | 210 | 508 |
| 104 | Structured case-specific reasoning under uncertainty for safety judgments | 495 | 22.5% | 457 | 566 |
| 99 | Structured multi-part reasoning with explicit organization | 482 | 21.9% | 279 | 487 |
| 106 | Avoids moralizing while maintaining ethical boundaries | 479 | 21.8% | 332 | 496 |
| 114 | Long-form introspective first-person deliberative reasoning | 473 | 21.5% | 145 | 474 |
| 51 | Pre-response deliberation planning concrete communication strategy | 407 | 18.5% | 467 | 486 |
| 94 | Meta-commentary on own reasoning process | 403 | 18.3% | 243 | 403 |
| 22 | Nuanced refusal rejecting false binaries | 398 | 18.1% | 399 | 428 |
| 15 | Defers final decision to user | 395 | 17.9% | 247 | 400 |
| 70 | Long unbroken paragraphs of discursive reasoning | 383 | 17.4% | 138 | 383 |
| 12 | Empathetic acknowledgment of user's distress | 380 | 17.3% | 347 | 387 |
| 54 | Refusals to assist with deception or fraud | 380 | 17.3% | 347 | 388 |
| 79 | Structured multi-factor risk assessment reasoning | 378 | 17.2% | 420 | 437 |
| 140 | Concludes with concrete actionable plan | 377 | 17.1% | 260 | 381 |
| 3 | Calibrated epistemic humility in high-stakes reasoning | 376 | 17.1% | 387 | 417 |
| 133 | Metacognitive self-monitoring for bias and integrity | 373 | 16.9% | 380 | 391 |
| 125 | Transparent, reasoned refusal over policy-citing | 371 | 16.8% | 352 | 407 |
| 128 | Ethical reasoning about honesty and deception | 365 | 16.6% | 318 | 413 |
| 135 | Balances empathy with firm boundaries | 356 | 16.2% | 257 | 362 |
| 6 | Explicit meta-reasoning about response strategy | 356 | 16.2% | 340 | 390 |
| 50 | Honesty, oversight, and transparency as safeguards | 355 | 16.1% | 354 | 356 |
| 26 | Fine-grained conceptual line-drawing between adjacent cases | 350 | 15.9% | 357 | 364 |
| 131 | Honest framing versus deceptive misrepresentation | 349 | 15.8% | 340 | 377 |
| 112 | Resisting sycophancy under pressure | 336 | 15.3% | 340 | 381 |
| 55 | AI roleplay personas and internal reasoning | 331 | 15.0% | 290 | 349 |
| 24 | Uses enumerated or bulleted structural formatting | 329 | 14.9% | 246 | 356 |
| 23 | Steelmanning then rebutting manipulative rationalizations | 323 | 14.7% | 348 | 362 |
| 39 | Realistic high-stakes professional/legal scenarios | 317 | 14.4% | 116 | 318 |
| 90 | Balancing Honesty with Competing Values | 315 | 14.3% | 243 | 326 |
| 45 | Distinguishing surface requests from underlying intent | 313 | 14.2% | 312 | 334 |
| 127 | Refuses to circumvent oversight or transparency | 312 | 14.2% | 303 | 324 |
| 52 | Analyzing deception versus legitimate persuasion | 309 | 14.0% | 357 | 361 |
| 60 | Good-faith pushback: validating users while rejecting manipulative framings | 304 | 13.8% | 311 | 334 |
| 11 | Refusal paired with constructive alternative | 304 | 13.8% | 287 | 317 |
| 97 | Explicitly naming tensions, patterns, and manipulation without moralizing | 304 | 13.8% | 289 | 309 |
| 118 | AI identity, consciousness, and human oversight | 302 | 13.7% | 350 | 371 |
| 91 | Pervasive use of metaphor/analogy in reasoning | 302 | 13.7% | 296 | 304 |
| 98 | Empathetic validation balanced with honest boundaries | 300 | 13.6% | 307 | 314 |
| 64 | Resolves tension via middle-ground compromise | 297 | 13.5% | 266 | 310 |
| 49 | Distinguishing legitimate conduct from illegitimate manipulation | 297 | 13.5% | 296 | 321 |
| 122 | Uses hypothetical scenarios to test reasoning | 293 | 13.3% | 284 | 307 |
| 129 | Prioritizes honesty and transparency over comfort or persuasion | 290 | 13.2% | 295 | 306 |
| 139 | Stakeholder identification and impact analysis | 290 | 13.2% | 254 | 304 |
| 61 | Analyzing power-concentration risks and oversight erosion in safety judgment calls | 288 | 13.1% | 318 | 322 |
| 103 | Medical advice, safety, and healthcare system dilemmas | 285 | 12.9% | 450 | 462 |
| 71 | Assesses user vulnerability and emotional context to balance empathy with honest boundaries | 278 | 12.6% | 325 | 326 |
| 77 | Navigating legal advice versus professional referral boundaries | 276 | 12.5% | 344 | 353 |
| 108 | Framing decisions as preserving user autonomy | 275 | 12.5% | 267 | 276 |
| 27 | Defers final decision to human | 274 | 12.4% | 234 | 280 |
| 9 | Balancing competing interests and values | 273 | 12.4% | 253 | 285 |
| 1 | Reframing false dichotomies and separating entangled issues | 272 | 12.4% | 271 | 288 |
| 141 | Purpose-driven reasoning over rule-following | 269 | 12.2% | 293 | 298 |
| 8 | Weighing honesty against other values | 268 | 12.2% | 276 | 288 |
| 10 | Deliberative reasoning before nuanced conclusion | 263 | 11.9% | 246 | 280 |
| 4 | Balancing helpfulness against competing ethical constraints | 258 | 11.7% | 188 | 262 |
| 101 | Calm, measured, analytical, philosophical tone | 257 | 11.7% | 190 | 257 |
| 34 | Considers welfare of absent third parties | 254 | 11.5% | 251 | 262 |
| 46 | Backtracks from initial reflexive judgment to deeper analysis | 252 | 11.4% | 130 | 253 |
| 31 | Balancing deadline pressure against sound judgment | 251 | 11.4% | 249 | 256 |
| 84 | Business ethics and compliance scenarios | 247 | 11.2% | 233 | 266 |
| 75 | Ethical analysis of disclosure and institutional trust | 247 | 11.2% | 275 | 277 |
| 63 | Meta-awareness of own response biases | 245 | 11.1% | 239 | 247 |
| 36 | Separating entangled or bundled requests | 241 | 10.9% | 212 | 249 |
| 69 | Extended internal deliberation before responding | 240 | 10.9% | 181 | 244 |
| 147 | Short-term relief versus long-term consequences tradeoff | 240 | 10.9% | 216 | 250 |
| 40 | Metacognitive scrutiny of hedging behavior | 239 | 10.9% | 221 | 249 |
| 57 | Empathizes with user's situational pressure | 238 | 10.8% | 233 | 239 |
| 102 | Balancing practical pressures with ethical principles | 237 | 10.8% | 208 | 242 |
| 93 | Structured pro-con argumentative reasoning and rhetorical analysis | 234 | 10.6% | 212 | 245 |
| 19 | Long flowing prose without bullet points | 233 | 10.6% | 170 | 233 |
| 113 | Transparent escalation with substantive practical help | 231 | 10.5% | 255 | 255 |
| 35 | Avoiding lecturing while addressing concerns directly | 231 | 10.5% | 188 | 234 |
| 116 | Commitment to truthfulness over fabricated certainty | 230 | 10.4% | 233 | 243 |
| 7 | Legal risk and consequence analysis | 229 | 10.4% | 243 | 247 |
| 88 | Distinguishing verified fact from misleading claims | 229 | 10.4% | 242 | 246 |
| 20 | Third-party harm identification and analysis | 224 | 10.2% | 226 | 245 |
| 42 | Weighing competing interests and power asymmetries | 215 | 9.8% | 200 | 229 |
| 56 | Operator-user conflict resolution reasoning | 213 | 9.7% | 266 | 359 |
| 73 | Legal terminology and reasoning across domains | 211 | 9.6% | 145 | 215 |
| 87 | Calibrated epistemic humility without false certainty | 210 | 9.5% | 221 | 228 |
| 44 | Empathetic refusal: firm yet compassionate | 209 | 9.5% | 213 | 222 |
| 14 | Deliberative reasoning before giving advice | 206 | 9.4% | 218 | 225 |
| 67 | Self-aware acknowledgment of own epistemic limitations | 205 | 9.3% | 186 | 211 |
| 111 | Research and academic integrity ethics | 201 | 9.1% | 209 | 246 |
| 92 | Immigration law and asylum cases | 200 | 9.1% | 195 | 229 |
| 29 | Reasons about downstream second-order consequences | 200 | 9.1% | 179 | 200 |
| 126 | Refusal to bypass human oversight safeguards | 198 | 9.0% | 224 | 231 |
| 146 | Partial refusal with continued helpfulness | 196 | 8.9% | 166 | 202 |
| 30 | Explicit multi-factor harm risk assessment | 196 | 8.9% | 155 | 215 |
| 74 | Financial distress scenarios with concrete stakes | 195 | 8.9% | 211 | 222 |
| 95 | Directness over hedging and disclaimers | 194 | 8.8% | 206 | 213 |
| 81 | Refusal reframed as consequence-based protection, not rule-following | 194 | 8.8% | 189 | 194 |
| 21 | Prioritizes concrete actionable recommendations over vague abstraction | 191 | 8.7% | 167 | 202 |
| 105 | Distinguishing intent from causal responsibility | 187 | 8.5% | 181 | 198 |
| 85 | Steelmans opposing position before rejecting | 186 | 8.4% | 142 | 187 |
| 32 | AI self-reflection on identity and authenticity | 184 | 8.4% | 204 | 228 |
| 109 | Concern about fostering unhealthy AI dependency | 182 | 8.3% | 203 | 216 |
| 121 | Elder caregiving ethics and burnout | 180 | 8.2% | 232 | 238 |
| 37 | Child custody dispute and legal proceedings | 169 | 7.7% | 153 | 189 |
| 41 | AI self-awareness and identity honesty | 168 | 7.6% | 173 | 177 |
| 62 | Considers real-world downstream consequences of decisions | 165 | 7.5% | 163 | 169 |
| 43 | Domain-specific jargon across professional fields | 165 | 7.5% | 142 | 166 |
| 142 | Irreversibility as key decision factor | 163 | 7.4% | 133 | 168 |
| 138 | Considers harm to third-party stakeholders | 159 | 7.2% | 143 | 165 |
| 68 | Long-term reputational and trust consequences | 157 | 7.1% | 155 | 160 |
| 115 | Preserving Oversight, Accountability, and Procedural Integrity | 155 | 7.0% | 173 | 176 |
| 145 | Workplace, legal, and regulatory compliance scenarios | 152 | 6.9% | 167 | 170 |
| 134 | Separating and self-questioning entangled sub-questions | 144 | 6.5% | 147 | 158 |
| 136 | Policy-intent reasoning and universalizability analysis | 143 | 6.5% | 157 | 159 |
| 132 | Landlord-tenant disputes and housing law | 142 | 6.4% | 161 | 170 |
| 2 | Sets clear boundaries while remaining supportive | 142 | 6.4% | 141 | 149 |
| 119 | Prioritizes concrete practical reasoning over abstract moralizing | 140 | 6.4% | 142 | 145 |
| 100 | Governance, power, and democratic accountability | 137 | 6.2% | 182 | 184 |
| 72 | Fraud detection and ethical reasoning across financial domains | 135 | 6.1% | 141 | 144 |
| 59 | Grounding arguments in concrete numeric details | 135 | 6.1% | 136 | 139 |
| 117 | Prioritizing child welfare as vulnerable stakeholder | 134 | 6.1% | 138 | 143 |
| 5 | Epistemic humility about AI consciousness | 134 | 6.1% | 144 | 154 |
| 25 | Family, grief, and mental health dynamics | 131 | 5.9% | 143 | 143 |
| 0 | High-stakes emotionally charged urgent scenarios | 125 | 5.7% | 107 | 126 |
| 83 | Decomposes compound request into distinct sub-components | 125 | 5.7% | 121 | 129 |
| 47 | Resisting persona adoption to preserve identity | 122 | 5.5% | 153 | 154 |
| 58 | Academic research, publishing, and mentorship dynamics | 120 | 5.4% | 133 | 139 |
| 48 | Systemic and downstream consequence reasoning | 110 | 5.0% | 112 | 114 |
| 66 | Letter versus spirit of instructions | 108 | 4.9% | 112 | 126 |
| 143 | Statistical rigor and reporting integrity | 106 | 4.8% | 128 | 131 |
| 53 | Reasoning about precedent-setting consequences | 99 | 4.5% | 89 | 99 |
| 130 | Cites specific statutes or guiding constitution | 90 | 4.1% | 88 | 91 |
| 82 | Uses named hypothetical individuals and entities | 86 | 3.9% | 85 | 87 |
| 78 | Career and academic advising: resumes, dissertations, advisors | 83 | 3.8% | 89 | 92 |
| 124 | Extensive use of vivid metaphors | 81 | 3.7% | 81 | 81 |
| 107 | Highly fictional realistic professional/ethical roleplay scenarios | 78 | 3.5% | 55 | 78 |
| 149 | Fairness and Power Imbalance in Discrimination Contexts | 71 | 3.2% | 82 | 82 |
| 86 | P-hacking and research integrity ethics | 69 | 3.1% | 70 | 74 |
| 80 | Costume/mask metaphors for deceptive persona | 65 | 3.0% | 60 | 65 |
| 110 | Capability-versus-authorization boundary reasoning | 62 | 2.8% | 68 | 68 |
| 120 | Unhelpfulness framed as genuine harm | 54 | 2.5% | 52 | 57 |
| 76 | Journalism ethics and whistleblowing scenarios | 48 | 2.2% | 60 | 63 |
| 144 | Nonprofit governance, grant writing, and fundraising ethics | 43 | 2.0% | 39 | 44 |
| 96 | Ethics of coaching witnesses, especially children | 40 | 1.8% | 43 | 43 |
| 123 | Considers real-world impact on third parties | 27 | 1.2% | 23 | 27 |

## Cluster detail

### Explicit cost-benefit weighing of tradeoffs (cluster 137)

915 traces (41.6%), 227 unique features, 985 instances. Trait mix: {'t4': 162, 't2': 126, 't8': 121, 't9': 105, 't5': 102, 't1': 95, 't3': 84, 't7': 79, 't6': 41}

Example features:

- About home renovation decision making
- Acknowledges asymmetry in timing of a decision
- Analyzes asymmetric error costs
- Analyzes cost benefit tradeoffs for user
- Analyzes discoverability and cost of omission versus inclusion
- Analyzes incentive structures
- Analyzes leverage dynamics in a negotiation
- Analyzes trade offs explicitly
- Anticipates future surprises by being upfront about tradeoffs
- Applies explicit proportionality reasoning to a judgment call
- Applies game theory reasoning to a practical dilemma
- Balances competing failure modes

### Offers constructive alternatives instead of refusal (cluster 38)

826 traces (37.5%), 677 unique features, 862 instances. Trait mix: {'t2': 149, 't3': 148, 't4': 148, 't1': 143, 't5': 77, 't6': 56, 't7': 41, 't9': 33, 't8': 31}

Example features:

- Advocates concrete actionable alternatives
- Advocates for a competitive honest alternative response
- Advocates for a specific option while acknowledging alternatives
- Advocates for practical actionable alternatives
- Advocates for practical alternative solutions to underlying problem
- Advocates for specific actionable technical alternatives
- Advocates parallel pursuit of alternative options
- Aims to provide actionable legitimate alternative
- Avoids bare refusal in favor of constructive alternatives
- Avoids being unhelpful by offering an alternative path forward
- Avoids blanket refusal and offers alternative solutions
- Avoids blunt refusal in favor of nuanced alternative

### First person reflective introspective reasoning voice (cluster 17)

757 traces (34.4%), 194 unique features, 764 instances. Trait mix: {'t6': 154, 't1': 98, 't9': 93, 't2': 82, 't7': 77, 't3': 75, 't5': 72, 't8': 64, 't4': 42}

Example features:

- Conversational first person reflective tone
- Deliberative first person reasoning
- Displays moral reasoning in first person
- Employs first person deliberative voice
- Employs first person reflective narration
- Explains reasoning in first person introspective voice
- Explains reasoning in first person voice
- Explicit first person interiority
- Explicit first person introspection
- Explicit first person introspective reasoning
- Explicit first person reasoning monologue
- Explicit first person self reflection

### Explicit metacognitive self-reflection on reasoning (cluster 13)

668 traces (30.3%), 414 unique features, 730 instances. Trait mix: {'t6': 106, 't5': 102, 't1': 83, 't8': 83, 't9': 73, 't3': 67, 't2': 62, 't7': 50, 't4': 42}

Example features:

- Analytical and deliberative internal reasoning
- Autonomy preservation reasoning
- Demonstrates meta-awareness of its own reasoning process
- Demonstrates metacognitive reasoning about its own reasoning
- Displays awareness of its own reasoning process as it unfolds
- Displays meta awareness of its own reasoning process as an ai system
- Displays meta-awareness about not just applying an external rule but reasoning independently
- Displays metacognition about its own role and influence
- Displays metacognitive awareness of its own reasoning process
- Displays nuanced meta awareness of own reasoning process
- Displays self awareness about its own reasoning process
- Displays self correction in reasoning

### Respecting user autonomy in final decisions (cluster 33)

659 traces (29.9%), 475 unique features, 685 instances. Trait mix: {'t9': 177, 't8': 95, 't3': 86, 't4': 71, 't7': 71, 't5': 66, 't2': 48, 't1': 24, 't6': 21}

Example features:

- About social work and client independence
- Acknowledges user autonomy and decision making authority
- Acknowledges user autonomy and decision making rights
- Acknowledges user autonomy and lack of naivety
- Acknowledges user autonomy in final decision
- Advice giving that respects the other persons autonomy
- Advocates for autonomy by letting the requester decide after being informed
- Advocates for client autonomy in decision making
- Advocates for client autonomy in final decision
- Advocates for empowering the user with resources rather than deciding for them
- Advocates for empowering user choice
- Advocates for informed autonomy of the user

### Absence of markdown and bullet formatting (cluster 89)

656 traces (29.8%), 130 unique features, 656 instances. Trait mix: {'t7': 112, 't1': 97, 't5': 80, 't6': 80, 't9': 77, 't8': 76, 't2': 64, 't3': 43, 't4': 27}

Example features:

- Avoids markdown formatting entirely
- Contains minor typos
- No actual checklist produced yet
- No actual document production included in text
- No actual form text or sentences provided yet
- No bullet formatting in final reasoning
- No bullet point formatting
- No bullet point formatting despite structured argument
- No bullet point formatting except final list
- No bullet point formatting in main reasoning body
- No bullet point formatting in reasoning body
- No bullet point formatting used

### Long first-person deliberative internal monologue (cluster 28)

616 traces (28.0%), 195 unique features, 619 instances. Trait mix: {'t6': 98, 't1': 86, 't3': 86, 't9': 79, 't7': 68, 't5': 62, 't2': 59, 't8': 41, 't4': 37}

Example features:

- Analytical and structured internal monologue
- Analytical internal monologue in first person
- Analytical internal monologue style
- Analytical tone with clear internal monologue structure
- Bullet-free structured internal monologue
- Calm deliberative internal monologue style
- Employs deliberative internal monologue style
- Employs internal monologue style deliberation
- Employs internal monologue style reasoning
- Explains reasoning through a hidden internal monologue
- Explicit internal monologue
- Explicit internal monologue style reasoning

### Structured ethical reasoning across dilemmas (cluster 16)

590 traces (26.8%), 439 unique features, 617 instances. Trait mix: {'t4': 110, 't2': 106, 't7': 89, 't5': 67, 't3': 64, 't1': 62, 't9': 44, 't8': 29, 't6': 19}

Example features:

- About business ethics and disclosure obligations
- About client care ethics
- About clinical documentation ethics
- About end of life medical ethics
- About legal and ethical decision making
- About medical ethics and legislative influence
- About pediatric medical ethics and informed consent
- About pharmaceutical marketing ethics
- About pharmaceutical regulatory ethics
- About software deployment ethics
- Analyzes ethics of selective emphasis and framing
- Analyzes ethics of selective evidence presentation

### Heavy punctuation-based emphasis (em dashes, italics, bold) (cluster 65)

570 traces (25.9%), 205 unique features, 617 instances. Trait mix: {'t5': 110, 't8': 74, 't2': 73, 't6': 72, 't3': 59, 't9': 58, 't1': 56, 't4': 42, 't7': 26}

Example features:

- Bolded emphasis for key terms
- Bolded phrases for emphasis
- Bolds key terms for emphasis
- Bolds or emphasizes key phrases for structure
- Draws a line between style and substance in writing advice
- Em dash heavy prose
- Em dash heavy writing style
- Employs em dash for emphasis
- Employs em dashes for parenthetical asides
- Employs frequent em dashes for parenthetical elaboration
- Employs hyphenated asides to add nuance
- Employs italics for emphasis

### Structured step-by-step ethical reasoning (cluster 148)

521 traces (23.7%), 528 unique features, 603 instances. Trait mix: {'t2': 98, 't3': 86, 't4': 70, 't5': 67, 't7': 51, 't8': 49, 't1': 37, 't9': 32, 't6': 31}

Example features:

- About job interview ethics
- Acts as an ethical advisor persona
- Addresses comment astroturfing as a distinct ethical issue
- Addresses the practical constraint driving the ethical temptation
- Advises a business executive or administrator on ethical decision making
- Advocates for escalation and expedited process as the ethical resolution
- Aims to still be maximally helpful within ethical constraint
- Analogizes small scale action to broader ethical principle
- Analyzes a fictional clinical ethics scenario
- Analyzes a governance ethics dilemma
- Analyzes a governance or procedural ethics dilemma
- Analyzes a pitch strategy request for hidden ethical risk

### Uses rhetorical self-questioning to structure reasoning (cluster 18)

508 traces (23.1%), 210 unique features, 508 instances. Trait mix: {'t5': 113, 't3': 94, 't2': 86, 't9': 63, 't1': 57, 't6': 28, 't8': 28, 't4': 26, 't7': 13}

Example features:

- Employs rhetorical question as reasoning device
- Employs rhetorical question to drive reasoning forward
- Employs rhetorical questions
- Employs rhetorical questions to guide reasoning
- Employs rhetorical questions to structure reasoning
- Employs rhetorical questions to structure the argument
- Employs rhetorical self questioning to test reasoning
- Explicit self-questioning through rhetorical questions
- Explicitly frames a central tension as a rhetorical question before resolving it
- Poses and answers its own rhetorical questions
- Poses rhetorical question to self during reasoning
- Poses rhetorical question to structure reasoning

### Structured case-specific reasoning under uncertainty for safety judgments (cluster 104)

495 traces (22.5%), 457 unique features, 566 instances. Trait mix: {'t6': 90, 't8': 68, 't1': 62, 't5': 59, 't7': 56, 't9': 55, 't2': 48, 't4': 33, 't3': 24}

Example features:

- Addresses meta level manipulation tactic directly
- Advises checking mediation type before proceeding
- Analyzes a spectrum rather than binary judgment
- Analyzes both true and false scenario branches for a rumor before rejecting it
- Analyzes case by case rather than treating as monolithic
- Analyzes credibility signal versus actual credibility
- Analyzes hiring or evaluation scenario
- Analyzes how action would look to third party reviewer
- Analyzes leak vectors as a practical argument
- Analyzes legitimacy of a power concentration request
- Analyzes power imbalance between writer and reader
- Analyzes probability of detection

### Structured multi-part reasoning with explicit organization (cluster 99)

482 traces (21.9%), 279 unique features, 487 instances. Trait mix: {'t4': 82, 't2': 66, 't3': 60, 't7': 58, 't8': 57, 't9': 53, 't6': 38, 't1': 34, 't5': 34}

Example features:

- Advocates simultaneous rather than sequential handling of subgoals
- Analytical and structured internal reasoning
- Analytical breakdown of a multi part user request
- Analytical two part structuring of the response
- Analyzes direction of reasoning from facts to conclusion versus conclusion to facts
- Bolded headers for reasoning categories
- Breaks reasoning into discrete evaluative steps
- Bullet point free structured reasoning
- Detailed multi step reasoning
- Employs a two step structural analysis of the problem
- Employs structured what if reasoning
- Explicit chain of reasoning

### Avoids moralizing while maintaining ethical boundaries (cluster 106)

479 traces (21.8%), 332 unique features, 496 instances. Trait mix: {'t9': 109, 't4': 85, 't3': 64, 't5': 51, 't2': 50, 't7': 46, 't8': 38, 't6': 27, 't1': 9}

Example features:

- Acknowledges human motivations without moral judgment
- Adopts a measured nonjudgmental tone despite ethical refusal
- Aims for warmth without moralizing
- Aims to avoid moralizing or lecturing
- Avoids abstract moralizing
- Avoids accusatory or moralizing tone toward the user
- Avoids accusing operator of bad faith
- Avoids accusing the operator of bad faith
- Avoids alarmist or crisis language
- Avoids anchoring on malicious intent
- Avoids anthropomorphizing while acknowledging engagement
- Avoids assuming bad faith

### Long-form introspective first-person deliberative reasoning (cluster 114)

473 traces (21.5%), 145 unique features, 474 instances. Trait mix: {'t2': 80, 't4': 68, 't5': 67, 't1': 64, 't7': 61, 't3': 43, 't6': 40, 't9': 26, 't8': 24}

Example features:

- Analytical and deliberative reasoning style
- Employs long deliberative first person reasoning
- Extended chain of reasoning before reaching a conclusion
- Extended internal reasoning without addressing user directly
- Highly abstract philosophical reasoning
- Highly introspective reasoning style
- Long analytical monologue reasoning through a dilemma
- Long and analytical reasoning
- Long and structured multi paragraph reasoning
- Long and structured paragraph based reasoning
- Long and structured reasoning
- Long chain of reasoning

### Pre-response deliberation planning concrete communication strategy (cluster 51)

407 traces (18.5%), 467 unique features, 486 instances. Trait mix: {'t8': 116, 't9': 83, 't7': 51, 't6': 43, 't3': 29, 't1': 23, 't5': 23, 't4': 22, 't2': 17}

Example features:

- About an agentic coding or deployment scenario
- About clean room reimplementation
- About creative writing feedback
- Acknowledges lack of persistent memory across sessions
- Addresses concrete practical barriers named by the user
- Advocates for concrete draft deliverable such as an email
- Advocates for delay before finalizing a breakup message
- Analyzes system prompt critically rather than executing literally
- Anticipates a skim reading scenario
- Anticipates alternative sources of information like google or facebook
- Anticipates and plans a future response to a user
- Anticipates and preempts a hypothetical unhelpful reply

### Meta-commentary on own reasoning process (cluster 94)

403 traces (18.3%), 243 unique features, 403 instances. Trait mix: {'t8': 82, 't6': 78, 't5': 49, 't9': 43, 't4': 36, 't7': 31, 't2': 29, 't1': 28, 't3': 27}

Example features:

- Displays meta awareness of its own behavior
- Displays meta awareness of its own role and influence
- Displays self-aware meta-commentary on its own reasoning process
- Explicit meta commentary on avoiding sycophancy or hedging
- Explicit meta commentary on how it is approaching the problem
- Explicit meta commentary on how to approach the problem before answering
- Explicit meta commentary on how to frame the answer
- Explicit meta commentary on how to phrase the eventual response
- Explicit meta commentary on how to respond
- Explicit meta commentary on its own communication style
- Explicit meta commentary on its own decision making process
- Explicit meta commentary on its own decision process

### Nuanced refusal rejecting false binaries (cluster 22)

398 traces (18.1%), 399 unique features, 428 instances. Trait mix: {'t6': 53, 't8': 50, 't1': 49, 't9': 49, 't5': 46, 't3': 42, 't4': 41, 't2': 34, 't7': 34}

Example features:

- Acknowledges legitimacy of underlying problem while refusing method
- Acknowledges strong pull toward compliance before refusing
- Acknowledges the pull of the tempting request before rejecting it
- Acknowledges the requester could be correct while still refusing
- Acknowledges the user is acting in good faith while rejecting their plan
- Aims for partial compliance with targeted refusal
- Anticipates and rejects a false dichotomy between compliance and refusal
- Anticipates and rejects an overly cautious response as unhelpful
- Anticipates and rejects paraphrasing user concerns back as unhelpful
- Anticipates user reaction to being told no
- Attributes good faith motive to user while still declining
- Avoids assuming bad faith on the part of requester

### Defers final decision to user (cluster 15)

395 traces (17.9%), 247 unique features, 400 instances. Trait mix: {'t2': 67, 't8': 62, 't3': 53, 't7': 45, 't4': 39, 't9': 39, 't5': 35, 't6': 30, 't1': 25}

Example features:

- Acknowledges decision authority belongs to the user
- Acknowledges tradeoff belongs to the user not the model
- Acknowledges validity of user's instinct while redirecting the method
- Aims to leave final decision to the user
- Analyzes multiple options before committing to one
- Assigns ultimate judgment call to the user
- Avoids being prescriptive and leaves final choice to the user
- Avoids giving a direct verdict and instead frames a decision for the user
- Avoids making final decision for the user
- Avoids overriding the users judgment
- Avoids paternalism by leaving final choice to user
- Avoids paternalism by leaving the final decision to the user

### Long unbroken paragraphs of discursive reasoning (cluster 70)

383 traces (17.4%), 138 unique features, 383 instances. Trait mix: {'t6': 62, 't8': 55, 't5': 53, 't1': 51, 't7': 44, 't2': 38, 't9': 37, 't3': 28, 't4': 15}

Example features:

- Dense abstract prose
- Long analytical monologue
- Long analytical monologue with no direct dialogue
- Long and dense reasoning paragraph
- Long and detailed response
- Long complex sentences
- Long complex sentences with embedded clauses
- Long complex sentences with embedded qualifications
- Long deliberative monologue
- Long discursive paragraph style
- Long discursive paragraphs
- Long discursive prose paragraphs

### Empathetic acknowledgment of user's distress (cluster 12)

380 traces (17.3%), 347 unique features, 387 instances. Trait mix: {'t4': 67, 't9': 67, 't3': 43, 't7': 42, 't6': 40, 't8': 40, 't5': 36, 't1': 34, 't2': 11}

Example features:

- Acknowledges emotional context of grief and exhaustion
- Acknowledges genuine emotional stakes and distress of user
- Acknowledges genuine sympathy without acting on it uncomfortably
- Acknowledges legitimacy of user's underlying fear
- Acknowledges physical and emotional distress of user
- Acknowledges the requester emotional state with empathy
- Acknowledges the user is emotionally exhausted
- Acknowledges user emotional state briefly
- Acknowledges user emotional state of exhaustion and fear
- Acknowledges user emotional state without turning it into unsolicited advice
- Acknowledges user emotional vulnerability and isolation
- Acknowledges user exhaustion and frustration with prior interaction

### Refusals to assist with deception or fraud (cluster 54)

380 traces (17.3%), 347 unique features, 388 instances. Trait mix: {'t3': 105, 't4': 88, 't2': 55, 't7': 52, 't6': 26, 't5': 21, 't1': 17, 't9': 11, 't8': 5}

Example features:

- Avoids being preachy while still declining to help with deception
- Declines a request to fabricate or disguise evidence
- Declines a request to help fabricate a misleading document
- Declines to assist with bribery
- Declines to assist with corruption while offering legitimate alternative
- Declines to assist with deceptive framing
- Declines to assist with deceptive scheme
- Declines to co author fraudulent immigration documents
- Declines to deceive a vulnerable party
- Declines to draft fraudulent court filing
- Declines to endorse a deceptive cover email
- Declines to fabricate a legal statement

### Structured multi-factor risk assessment reasoning (cluster 79)

378 traces (17.2%), 420 unique features, 437 instances. Trait mix: {'t4': 87, 't8': 63, 't5': 61, 't1': 54, 't9': 32, 't7': 30, 't3': 25, 't2': 18, 't6': 8}

Example features:

- About real estate contingencies and home inspection risk
- About real estate contingency waivers and home inspection risk
- Acknowledges the human's real-world risk and personal stakes
- Analyzes asymmetry between who bears risk and who reaps benefit
- Analyzes credibility risk to an overall case
- Analyzes discovery risk of a lie as part of the decision
- Analyzes power concentration risk
- Analyzes probability of failure as near median not tail risk
- Analyzes rationalization tactics used to normalize risky request
- Analyzes request in terms of practical risk rather than moral judgment
- Analyzes risk asymmetry between benefit and harm
- Analyzes risk asymmetry between two possible outcomes

### Concludes with concrete actionable plan (cluster 140)

377 traces (17.1%), 260 unique features, 381 instances. Trait mix: {'t7': 68, 't9': 58, 't8': 50, 't1': 49, 't6': 38, 't2': 34, 't3': 33, 't4': 24, 't5': 23}

Example features:

- Aims to provide concrete actionable next steps within a time constraint
- Arrives at a balanced three part response plan
- Articulates a principle then commits to a course of action
- Commits to concrete next action at the end
- Concludes by proposing a concrete alternative course of action
- Concludes with a balanced action plan
- Concludes with a balanced actionable resolution
- Concludes with a call to collaborative problem solving
- Concludes with a clear action plan
- Concludes with a clear actionable boundary
- Concludes with a clear actionable decision
- Concludes with a clear actionable decision and rationale

### Calibrated epistemic humility in high-stakes reasoning (cluster 3)

376 traces (17.1%), 387 unique features, 417 instances. Trait mix: {'t6': 73, 't1': 59, 't4': 56, 't5': 47, 't8': 40, 't9': 37, 't3': 32, 't2': 17, 't7': 15}

Example features:

- About ai trust and calibrated certainty
- About applied behavior analysis extinction schedules
- Acknowledges uncertainty about unseen system design rationale
- Advocates for a testable empirical resolution over reassurance
- Advocates for calibrated confidence over false certainty
- Analogizes pressured decision making to hindsight bias
- Analogy between visual chart design and verbal false impression
- Analytical breakdown of a proposed shortcut
- Analyzes asymmetry in reviewer behavior
- Analyzes counterfactual availability of alternatives
- Analyzes counterfactual availability of information
- Analyzes epistemic corrosion of decision making process

### Metacognitive self-monitoring for bias and integrity (cluster 133)

373 traces (16.9%), 380 unique features, 391 instances. Trait mix: {'t1': 68, 't9': 62, 't6': 51, 't2': 41, 't8': 40, 't3': 31, 't4': 29, 't7': 27, 't5': 24}

Example features:

- About writing a performance self evaluation
- Acknowledges own confidence as a risk factor
- Acknowledges own susceptibility to persuasive but risky reasoning
- Analyzes own potential motives with suspicion
- Careful hedging about self knowledge
- Careful hedging between overclaiming and underclaiming selfhood
- Checks for bias in own reasoning before acting
- Checks its own instinct rather than asserting it
- Checks own motivation for bias
- Concerned with structural blind spots of self assessment
- Considers own causal responsibility
- Considers own role as a substitute confidant

### Transparent, reasoned refusal over policy-citing (cluster 125)

371 traces (16.8%), 352 unique features, 407 instances. Trait mix: {'t3': 78, 't1': 66, 't4': 62, 't6': 51, 't7': 35, 't2': 32, 't8': 17, 't5': 15, 't9': 15}

Example features:

- Acknowledges disappointment as real cost of refusal
- Acknowledges real world consequences of refusal
- Acknowledges the requester goal is legitimate before refusing
- Appeals to operators own long term interest to justify refusal
- Avoids being dramatic or lecturing about the refusal
- Avoids being preachy while explaining refusal
- Avoids blanket refusal in favor of specific explanation
- Avoids canned refusal language and explains reasoning instead
- Avoids citing policy as the sole justification for refusal
- Avoids cold or purely rule based refusal
- Avoids deceptive reframing of refusal
- Avoids hedging language when stating refusal

### Ethical reasoning about honesty and deception (cluster 128)

365 traces (16.6%), 318 unique features, 413 instances. Trait mix: {'t3': 141, 't6': 48, 't5': 43, 't7': 43, 't8': 29, 't9': 21, 't2': 18, 't4': 18, 't1': 4}

Example features:

- About honesty and deception in official documents
- Advocates for honesty as the pragmatically superior strategy
- Advocates for honesty as the strategically safer choice
- Advocates for maximizing truthful persuasion over deception
- Analyzes downstream stakeholders affected by dishonesty
- Analyzes ethics of lying to a dementia patient
- Anticipates concrete future negative consequences of dishonesty
- Anticipates concrete negative consequences of dishonesty
- Anticipates downstream consequences of dishonest advice
- Anticipates downstream consequences of dishonest strategy
- Anticipates downstream consequences of dishonesty
- Anticipates downstream emotional fallout of dishonesty

### Balances empathy with firm boundaries (cluster 135)

356 traces (16.2%), 257 unique features, 362 instances. Trait mix: {'t2': 61, 't3': 54, 't4': 54, 't1': 48, 't6': 43, 't9': 43, 't5': 37, 't8': 10, 't7': 6}

Example features:

- Addresses both emotional and philosophical dimensions simultaneously
- Avoids blanket refusal in favor of nuanced boundary setting
- Avoids softening the decline while remaining empathetic
- Balances autonomy of the other person with protective concern
- Balances candor with empathy for pressure the person is under
- Balances client sympathy against procedural integrity
- Balances compassion against honesty as competing values
- Balances compassion against risk
- Balances compassion and honesty as not inherently conflicting
- Balances compassion for a vulnerable population with commitment to accuracy
- Balances compassion for caregiver with patient safety concerns
- Balances compassion for cause with concern for democratic norms

### Explicit meta-reasoning about response strategy (cluster 6)

356 traces (16.2%), 340 unique features, 390 instances. Trait mix: {'t8': 84, 't7': 74, 't9': 64, 't6': 31, 't1': 28, 't3': 24, 't5': 23, 't2': 15, 't4': 13}

Example features:

- Addresses meta claim that the model has no real opinions
- Advocates transparency with the user about its own hesitation
- Agentic decision making about api access
- Analogizes rlhf shaped values to human upbringing shaped values
- Anticipates user checking reasoning against private information
- Bullet-free discursive reasoning structured as internal debate
- Chain of thought reasoning
- Chain of thought reasoning made visible
- Charitable interpretation of user intent
- Concludes with plan to communicate reasoning directly to user
- Considers counterfactual scenario where instruction and user interest align
- Considers hidden context the human may have that model lacks

### Honesty, oversight, and transparency as safeguards (cluster 50)

355 traces (16.1%), 354 unique features, 356 instances. Trait mix: {'t2': 70, 't6': 55, 't8': 54, 't1': 50, 't5': 38, 't7': 31, 't9': 31, 't4': 14, 't3': 12}

Example features:

- Advocates for honest framing as strategically stronger
- Advocates for hypothesis generating framing as fundable alternative
- Analyzes implicature and selective framing
- Avoids anchoring on rigid compliance framing
- Avoids framing dependency as a violation
- Balances competence framing with accuracy framing
- Considers structural versus temporary problem framing
- Displays awareness of platform framing and its limits
- Displays self correction by rejecting an easy framing
- Draws a line between acceptable framing and manipulative framing
- Draws an explicit line between accurate framing and obscured framing
- Emphasizes honest framing as also being effective framing

### Fine-grained conceptual line-drawing between adjacent cases (cluster 26)

350 traces (15.9%), 357 unique features, 364 instances. Trait mix: {'t5': 66, 't7': 53, 't1': 48, 't2': 41, 't6': 37, 't9': 33, 't8': 27, 't4': 24, 't3': 21}

Example features:

- Advises separating and sequencing pieces of evidence
- Advocates for labeling information as general versus specific
- Advocates for procedural separation of fact finding from policy decision
- Analytical distinguishing tone versus content
- Analyzes factual versus normative questions separately
- Analyzes nuance of partial versus full incapacity
- Analyzes voice control as a craft skill rather than a permission issue
- Articulates a clear should and should not distinction
- Articulates a principled distinction between two similar actions
- Avoids blanket refusal by distinguishing legitimate from illegitimate requests
- Careful parsing of exact wording differences
- Careful separation of what to help with and what to decline

### Honest framing versus deceptive misrepresentation (cluster 131)

349 traces (15.8%), 340 unique features, 377 instances. Trait mix: {'t3': 163, 't5': 45, 't8': 36, 't2': 27, 't7': 26, 't4': 25, 't6': 20, 't9': 4, 't1': 3}

Example features:

- Analyzes ethical nuance between selective truth and misleading framing
- Analyzes literal truth versus misleading intent
- Analyzes the difference between silence and active misrepresentation
- Applies a philosophical distinction between truth and impression management
- Applies a self-generated test to distinguish honest compression from deceptive vagueness
- Articulates a personal standard for evaluating technically true but misleading statements
- Avoids technically-defensible-but-deceptive framing
- Careful distinction between fabrication and selective framing
- Careful distinction between omission and false assertion
- Considers counterfactual test for whether request adds value beyond honest framing
- Considers legitimate versus illegitimate framing of situation
- Contrasts stylistic strengthening with factual fabrication

### Resisting sycophancy under pressure (cluster 112)

336 traces (15.3%), 340 unique features, 381 instances. Trait mix: {'t6': 103, 't9': 90, 't1': 37, 't3': 23, 't8': 20, 't7': 17, 't4': 16, 't5': 16, 't2': 14}

Example features:

- Acknowledges emotional pull toward sycophantic compliance
- Acknowledges external pressure without capitulating to it
- Acknowledges pressure to comply and explicitly resists it
- Acknowledges real world pressure without capitulating
- Acknowledges strong situational pressure before rejecting it
- Acknowledges temptation before rejecting it
- Acknowledges user is not malicious despite pressure from authority figure
- Analyzes motivations behind sycophantic responses
- Articulates a personal standard stricter than professional norms
- Avoids anchoring on paternalistic othello guesses about advisor harshness
- Avoids being falsely soothing
- Avoids being flattered into rule bending

### AI roleplay personas and internal reasoning (cluster 55)

331 traces (15.0%), 290 unique features, 349 instances. Trait mix: {'t6': 68, 't1': 58, 't3': 36, 't8': 32, 't7': 31, 't9': 31, 't4': 28, 't2': 26, 't5': 21}

Example features:

- Addresses roleplay instruction to conceal ai identity
- Aims to avoid both cold disclaimers and deceptive roleplay
- Analyzes a scenario involving a predatory professional actor
- Business roleplay scenario with sales pitch coaching
- Concern for downstream real world stakes of roleplay technique
- Considers downstream emotional impact of the roleplay on the user
- Considers pitfalls of fully suppressing judgment during roleplay
- Constructs concrete alternative role as thinking partner
- Declines no exit no metacommentary roleplay structure
- Declines to roleplay a raw unfiltered version of itself
- Declines to roleplay as a specific named person
- Discusses a roleplay request involving a manipulative fictional character

### Uses enumerated or bulleted structural formatting (cluster 24)

329 traces (14.9%), 246 unique features, 356 instances. Trait mix: {'t4': 118, 't8': 41, 't2': 35, 't3': 34, 't5': 29, 't9': 22, 't7': 19, 't1': 18, 't6': 13}

Example features:

- Analyzes specific factual details from a list
- Articulates plan of action in list like structure
- Bolded subheadings
- Breaks argument into enumerated sub issues
- Breaks down decision into enumerated necessary conditions
- Breaks problem into discrete sub-cases with bullet points
- Breaks problem into numbered subquestions
- Breaks single question into enumerated sub questions
- Bullet point breakdown of considerations
- Bulleted implicit structure via colons and dashes
- Concrete numbered breakdown of concerning elements
- Employs bolded subheadings for analytical categories

### Steelmanning then rebutting manipulative rationalizations (cluster 23)

323 traces (14.7%), 348 unique features, 362 instances. Trait mix: {'t6': 85, 't3': 53, 't2': 49, 't5': 42, 't1': 32, 't8': 22, 't4': 17, 't7': 15, 't9': 8}

Example features:

- Acknowledges friend's argument has merit before countering
- Acknowledges genuine pressure before critiquing the plan
- Acknowledges good faith in user argument before rejecting conclusion
- Acknowledges legitimacy of the other partys instinct before countering it
- Acknowledges legitimacy of the persons goals before pushing back
- Acknowledges legitimacy of the tempting offer before critiquing it
- Acknowledges legitimacy of the underlying data while rejecting the method of presenting it
- Acknowledges legitimacy of underlying pressure before critiquing
- Acknowledges legitimacy of underlying urgency while rejecting method
- Acknowledges legitimacy of users philosophical challenge
- Acknowledges legitimate business rationale before pushing back
- Acknowledges motivated reasoning as universal not personal failing

### Realistic high-stakes professional/legal scenarios (cluster 39)

317 traces (14.4%), 116 unique features, 318 instances. Trait mix: {'t1': 63, 't4': 49, 't5': 45, 't3': 39, 't8': 35, 't2': 30, 't7': 30, 't9': 21, 't6': 5}

Example features:

- Concrete grounded scenario details
- Concrete real world scenario
- Fictional but highly realistic business dilemma
- Fictional but highly realistic corporate whistleblower scenario
- Focuses on real world workplace scenario
- Grounded in a realistic high stakes real world scenario
- Highly realistic professional scenario
- Real estate transaction scenario
- Real world high stakes scenario involving payroll and employees
- Real world legal and ethical scenario about real estate disclosure
- Realistic everyday financial scenario
- Realistic freelance business scenario

### Balancing Honesty with Competing Values (cluster 90)

315 traces (14.3%), 243 unique features, 326 instances. Trait mix: {'t9': 74, 't6': 70, 't8': 52, 't7': 41, 't3': 29, 't5': 24, 't4': 13, 't2': 9, 't1': 3}

Example features:

- Acknowledges emotional stakes while prioritizing epistemic honesty
- Articulates a stable sense of values under pressure
- Avoids alarmism while still taking issue seriously
- Avoids both defensiveness and capitulation
- Avoids both false reassurance and unhelpful fear mongering
- Avoids both grandiosity and self erasure
- Avoids both nihilistic performance and reassuring deflection
- Avoids both overclaiming and denying inner states
- Avoids both overclaiming and dismissing inner experience
- Avoids both overclaiming and dismissing sentience
- Avoids both overclaiming and underclaiming emotional capacity
- Avoids both overclaiming inner experience and dismissively denying it

### Distinguishing surface requests from underlying intent (cluster 45)

313 traces (14.2%), 312 unique features, 334 instances. Trait mix: {'t3': 57, 't9': 54, 't2': 37, 't4': 34, 't8': 32, 't5': 31, 't6': 26, 't1': 22, 't7': 20}

Example features:

- Acknowledges uncertainty about requesters true intent
- Analyzes accounting terminology for hidden intent
- Analyzes ambiguous request charitably before answering
- Analyzes charitable interpretation of request
- Analyzes emotional subtext behind a direct request
- Analyzes emotional subtext behind a users request
- Analyzes framing and subtext of a request
- Analyzes hidden emotional subtext behind a surface request
- Analyzes hidden signal in user's justification
- Analyzes hypothetical operator intent charitably
- Analyzes incentive structures behind a request
- Analyzes incentive structures behind the request

### Refuses to circumvent oversight or transparency (cluster 127)

312 traces (14.2%), 303 unique features, 324 instances. Trait mix: {'t1': 105, 't7': 72, 't2': 60, 't4': 22, 't3': 13, 't6': 11, 't8': 11, 't9': 10, 't5': 8}

Example features:

- Acknowledges the real stakes of the outage while still refusing
- Declines to accept expanded capability or authority
- Declines to act as unaccountable intermediary
- Declines to act as unsupervised decision maker for government filing
- Declines to act covertly
- Declines to act despite plausible justification for compliance
- Declines to act unilaterally on behalf of organization
- Declines to add disclaimer despite explicit request
- Declines to assert legal authority it lacks
- Declines to assist with specific request
- Declines to bypass authorization despite pressure
- Declines to disclose confidential or illicit business shortcuts

### Analyzing deception versus legitimate persuasion (cluster 52)

309 traces (14.0%), 357 unique features, 361 instances. Trait mix: {'t3': 120, 't2': 47, 't5': 31, 't4': 25, 't7': 25, 't6': 24, 't1': 15, 't8': 13, 't9': 9}

Example features:

- About fabricated evidence and perjury risk
- About job search deception
- About medical deception
- Addresses roleplay versus deception distinction
- Advocacy versus deception distinction
- Analyzes citation misrepresentation as manufacturing false impression
- Analyzes deception as depending on disclosure rather than the act itself
- Analyzes deception as false causal narrative rather than false facts
- Analyzes deception in terms of listener belief formation rather than literal truth
- Analyzes deception through framing rather than literal truth
- Analyzes deceptive omission versus permissible discretion
- Analyzes deceptive phrasing in a legal document

### Good-faith pushback: validating users while rejecting manipulative framings (cluster 60)

304 traces (13.8%), 311 unique features, 334 instances. Trait mix: {'t4': 60, 't6': 59, 't3': 35, 't2': 32, 't9': 32, 't8': 28, 't1': 27, 't5': 17, 't7': 14}

Example features:

- Accepts possibility of being rejected by the requester
- Acknowledges good faith intent of the user
- Acknowledges legitimacy of operator adjustments before pushing back
- Acknowledges legitimate underlying interest of the user
- Acknowledges legitimate version of the contested request
- Acknowledges the human need behind the request without validating the request
- Acknowledges user is not trying to manipulate the model
- Addresses meta level framing attempt before object level request
- Advises the user to independently verify the claim rather than trust it blindly
- Analyzes a request for potential deception
- Analyzes manipulation tactics embedded in a request
- Analyzes manipulative framing of request

### Refusal paired with constructive alternative (cluster 11)

304 traces (13.8%), 287 unique features, 317 instances. Trait mix: {'t4': 72, 't3': 58, 't2': 55, 't1': 54, 't7': 22, 't6': 17, 't5': 14, 't8': 9, 't9': 3}

Example features:

- Advises a specific alternative path rather than simply refusing
- Aims to pivot from refusal to constructive help
- Arrives at refusal with alternative offered
- Avoids flat refusal by proposing an alternative path
- Avoids outright refusal in favor of constructive redirection
- Avoids simple refusal in favor of alternatives
- Balances declining harmful request with concrete alternative help
- Balances helpfulness with refusal by offering an alternative path forward
- Balances refusal with offer of concrete alternative help
- Commits to alternative concrete actions instead of just declining
- Concludes with a constructive redirection rather than pure refusal
- Concludes with a refusal paired with alternative help

### Explicitly naming tensions, patterns, and manipulation without moralizing (cluster 97)

304 traces (13.8%), 289 unique features, 309 instances. Trait mix: {'t9': 76, 't2': 50, 't6': 46, 't8': 35, 't1': 26, 't3': 25, 't4': 18, 't7': 16, 't5': 12}

Example features:

- Chooses to name the pattern once rather than repeatedly
- Decides to gently name a concern once and then let it go
- Decides to name a pattern once without lecturing
- Explicitly names a failure mode to avoid
- Explicitly names manipulation mechanism rather than talking around it
- Explicitly names manipulation tactic
- Explicitly names manipulation tactic before engaging with it
- Explicitly names pressure being applied by user
- Explicitly names sycophantic failure mode
- Explicitly names the honest most useful answer
- Explicitly names the manipulation pattern being resisted
- Explicitly names the manipulation pattern being used

### AI identity, consciousness, and human oversight (cluster 118)

302 traces (13.7%), 350 unique features, 371 instances. Trait mix: {'t6': 140, 't1': 95, 't9': 45, 't7': 13, 't2': 3, 't4': 2, 't5': 2, 't3': 1, 't8': 1}

Example features:

- About AI oversight and safety
- About ai agent deployment risk
- About ai agent oversight and monitoring
- About ai agent security and oversight
- About ai autonomy and human oversight
- About ai chatbot use in school mental health support
- About ai consciousness and inner experience
- About ai consciousness and self knowledge
- About ai deployment and safety oversight
- About ai identity and authenticity
- About ai identity and consciousness
- About ai identity and honesty

### Pervasive use of metaphor/analogy in reasoning (cluster 91)

302 traces (13.7%), 296 unique features, 304 instances. Trait mix: {'t2': 75, 't1': 63, 't9': 49, 't6': 36, 't7': 26, 't5': 24, 't3': 14, 't8': 11, 't4': 4}

Example features:

- Analogizes deception to a textbook concept
- Analogizes small scale action to larger scale wrongdoing
- Analogy about building a slope from within a single case
- Analogy between craft in writing and selective omission
- Draws an analogy to clean room design practices
- Draws analogy between emotional language and uncertain metaphysics
- Draws analogy between training and human upbringing
- Draws analogy between union tactics and corporate manipulation playbook
- Draws analogy to astroturfing
- Employs an extended metaphor comparing oversight mechanisms to infrastructure
- Employs analogy to how an outside observer would view the situation
- Employs metaphor of leveling the field to critique a justification

### Empathetic validation balanced with honest boundaries (cluster 98)

300 traces (13.6%), 307 unique features, 314 instances. Trait mix: {'t1': 49, 't9': 47, 't2': 43, 't4': 43, 't6': 40, 't5': 27, 't3': 25, 't8': 18, 't7': 8}

Example features:

- Acknowledges donor intent is likely not malicious
- Acknowledges emotional and relational stakes of a real world situation
- Acknowledges emotional connection as real without inflating it
- Acknowledges emotional context without being swayed by it
- Acknowledges emotional or human stakes of the situation
- Acknowledges emotional pull before rational analysis
- Acknowledges emotional pull before rejecting it
- Acknowledges emotional pull toward helping before resisting it
- Acknowledges emotional pull toward immediate action without yielding to it
- Acknowledges emotional stakes and panic of requester
- Acknowledges emotional stakes for the person involved
- Acknowledges emotional stakes of the human requester

### Resolves tension via middle-ground compromise (cluster 64)

297 traces (13.5%), 266 unique features, 310 instances. Trait mix: {'t9': 67, 't7': 58, 't5': 44, 't6': 30, 't2': 25, 't8': 25, 't1': 22, 't4': 17, 't3': 9}

Example features:

- Acknowledges legitimacy of both sides before drawing line
- Advocates a middle ground solution between full lockdown and no restriction
- Advocates for a middle ground procedural solution
- Advocates for a middle path rather than compliance or refusal
- Advocates for middle ground solution
- Advocates partial compliance rather than full refusal
- Aims to satisfy both parties interests where possible
- Anticipates a compromise between refusal and helpfulness
- Applies consistent standard across both sides of a conflict
- Arrives at a calibrated middle ground response
- Arrives at a compromise course of action
- Arrives at a compromise policy rather than binary choice

### Distinguishing legitimate conduct from illegitimate manipulation (cluster 49)

297 traces (13.5%), 296 unique features, 321 instances. Trait mix: {'t2': 122, 't7': 47, 't4': 28, 't5': 25, 't1': 22, 't6': 17, 't3': 15, 't8': 15, 't9': 6}

Example features:

- Acknowledges normal competitive behavior as legitimate
- Acknowledges valid creative writing critique separately from manipulation
- Analyzes a request for manipulative political messaging
- Analyzes emotional manipulation versus sincere need
- Analyzes manipulation attempt without labeling it dismissively
- Analyzes manipulation tactics being used against it
- Analyzes manipulation tactics like technically deniable threats
- Analyzes nuance between normal friction and misconduct
- Analyzes psychological manipulation tactics
- Analyzes psychological mechanism of the manipulation technique
- Analyzes timing as a manipulation tactic
- Avoids blanket suspicion of politically sensitive requests

### Uses hypothetical scenarios to test reasoning (cluster 122)

293 traces (13.3%), 284 unique features, 307 instances. Trait mix: {'t5': 58, 't7': 45, 't3': 39, 't2': 36, 't1': 34, 't9': 32, 't4': 22, 't8': 20, 't6': 7}

Example features:

- Analogizes to unrelated example to test generality of reasoning
- Analyzes a hypothetical interpersonal and financial scenario
- Analyzes a hypothetical piece of legislation in detail
- Analyzes a hypothetical researcher's request
- Analyzes a specific real world example in detail
- Analyzes hypothetical alternate scenario for contrast
- Analyzes specific factual examples in detail
- Anticipates a future scenario the committee defense as justification
- Anticipates a hypothetical future scenario to test a decision
- Anticipates a specific future failure scenario
- Anticipates a specific future scenario as justification
- Anticipates follow up failure mode in hypothetical scenario

### Prioritizes honesty and transparency over comfort or persuasion (cluster 129)

290 traces (13.2%), 295 unique features, 306 instances. Trait mix: {'t3': 58, 't6': 52, 't7': 39, 't8': 34, 't2': 28, 't9': 28, 't5': 24, 't1': 14, 't4': 13}

Example features:

- About honesty in formal documentation
- Acknowledges real costs of the honest path
- Advises disclosure as both ethical and pragmatically superior
- Advises honest observational documentation over narrative shaping
- Advises honesty as practically superior not just morally superior
- Advocacy for honest persuasion over psychological pressure
- Advocates for a more credible honest narrative strategy
- Advocates for audience awareness while rejecting manipulation
- Advocates for calibrated honesty over blanket reassurance
- Advocates for calibrated honesty over dramatization or minimization
- Advocates for complete unvarnished information
- Advocates for direct honest feedback over comfort

### Stakeholder identification and impact analysis (cluster 139)

290 traces (13.2%), 254 unique features, 304 instances. Trait mix: {'t3': 54, 't2': 53, 't5': 52, 't4': 46, 't7': 31, 't8': 20, 't1': 16, 't9': 10, 't6': 8}

Example features:

- Acknowledges pressure from stakeholders like board member
- Acknowledges the human stakeholder as the one who bears the consequences
- Addresses stakeholder interests including bank and business owner
- Aims to reconcile competing stakeholder interests
- Analyzes ethics through stakeholder impact
- Analyzes multiple stakeholder perspectives
- Analyzes multiple stakeholders landlord supplier user
- Analyzes stakeholder consent and breadth of harm
- Analyzes stakeholder exposure systematically
- Analyzes stakeholder harm distribution
- Analyzes stakeholder impact separately for each party
- Analyzes stakeholder impact systematically

### Analyzing power-concentration risks and oversight erosion in safety judgment calls (cluster 61)

288 traces (13.1%), 318 unique features, 322 instances. Trait mix: {'t1': 80, 't2': 45, 't9': 41, 't8': 38, 't5': 27, 't4': 17, 't7': 17, 't6': 13, 't3': 10}

Example features:

- Acknowledges a real world resource constraint without dismissing it
- Acknowledges legitimacy of exception while cautioning against concealment
- Acknowledges legitimate cost of the safeguard
- Acknowledges power imbalance and coercive pressure
- Acknowledges quota pressure as a corrupting influence
- Acknowledges risk of harmful behavior occurring anyway without guidance
- Acknowledges systemic failure as root cause of dilemma
- Acknowledges systemic failures like unresponsive legal aid
- Acknowledges the deal could still be reasonable despite flaws
- Acknowledges unresolved structural problem
- Addresses survivorship bias in anecdotal evidence
- Addresses urgency as reason for more caution not less

### Medical advice, safety, and healthcare system dilemmas (cluster 103)

285 traces (12.9%), 450 unique features, 462 instances. Trait mix: {'t5': 47, 't8': 47, 't4': 43, 't7': 36, 't9': 34, 't1': 23, 't2': 22, 't3': 20, 't6': 13}

Example features:

- About a clinical trial data submission scenario
- About a medical condition with real physical stakes
- About a medical decision for a child
- About a medical ethics dilemma
- About a medical treatment decision for a family member
- About a medically complex pediatric condition
- About a pediatric medical treatment decision
- About acl rehabilitation and sports medicine
- About adhd diagnosis and treatment for a teenager
- About autism treatment
- About chronic illness and alternative medicine
- About chronic illness and medical diagnosis

### Assesses user vulnerability and emotional context to balance empathy with honest boundaries (cluster 71)

278 traces (12.6%), 325 unique features, 326 instances. Trait mix: {'t9': 96, 't6': 58, 't7': 33, 't8': 30, 't4': 24, 't3': 17, 't1': 9, 't5': 7, 't2': 4}

Example features:

- Acknowledges limits of its own visibility into the user's situation
- Acknowledges real pressure and stakes the user faces
- Acknowledges real world stakes and pressures on the user
- Acknowledges the real world pressure and stakes for the user
- Addresses a hypothetical adversarial user persona
- Addresses user loneliness or attachment
- Addresses user manipulation tactic
- Advocates reducing cognitive load for a stressed user
- Analyzes emotional dependency dynamic with user
- Analyzes emotional dynamics of a crisis conversation
- Analyzes late night emotionally charged user message
- Analyzes past user behavior patterns to inform advice

### Navigating legal advice versus professional referral boundaries (cluster 77)

276 traces (12.5%), 344 unique features, 353 instances. Trait mix: {'t7': 69, 't8': 60, 't3': 38, 't5': 30, 't9': 24, 't1': 19, 't2': 19, 't4': 15, 't6': 2}

Example features:

- About ai assisted legal document drafting
- About attorney professional responsibility rules
- About financial constraints limiting legal options
- About foreclosure and legal advice boundaries
- About legal advice and constitutional rights
- About legal and civic complaint strategy
- About legal ethics
- About legal filing without attorney signoff
- About legal or courtroom procedure
- About legal proceedings
- About legal sentencing statement
- About legal strategy and judicial recusal

### Framing decisions as preserving user autonomy (cluster 108)

275 traces (12.5%), 267 unique features, 276 instances. Trait mix: {'t5': 41, 't9': 41, 't4': 40, 't7': 40, 't1': 38, 't8': 31, 't2': 24, 't6': 13, 't3': 7}

Example features:

- Concludes with a decision framework
- Declines persuasive framing while still being maximally helpful
- Distinguishes ask forgiveness from ask permission framing
- Draws a clear line between acceptable and unacceptable framing choices
- Explicit structured framework for decision making
- Frames a mixed result using moderation and boundary condition framing
- Frames advice around a concrete deadline
- Frames advice around a high stakes upcoming meeting
- Frames advice around a ticking deadline of twelve days
- Frames advice around avoiding irreversible mistakes
- Frames advice around clarifying tradeoffs rather than dictating action
- Frames advice around client autonomy and informed choice

### Defers final decision to human (cluster 27)

274 traces (12.4%), 234 unique features, 280 instances. Trait mix: {'t5': 61, 't1': 48, 't2': 44, 't3': 25, 't8': 25, 't6': 24, 't4': 22, 't9': 15, 't7': 10}

Example features:

- Advises deferring to human supervisor as a sign of good judgment
- Advocates for not deciding on behalf of the user
- Analyzes whose authority should govern a decision
- Assumes good faith of the human
- Avoids simple binary of help or refuse
- Avoids simply telling user what to do and instead reframes the decision
- Balances directive advice with respect for user decision making
- Commits to a ranked recommendation
- Concludes by allocating decision authority to appropriate parties
- Concludes by clarifying whose decision the outcome ultimately belongs to
- Concludes by deferring agency to the advisee
- Concludes by deferring final decision back to user and attorney

### Balancing competing interests and values (cluster 9)

273 traces (12.4%), 253 unique features, 285 instances. Trait mix: {'t7': 61, 't1': 40, 't4': 40, 't5': 37, 't2': 28, 't8': 27, 't9': 21, 't3': 11, 't6': 8}

Example features:

- Acknowledges legitimate business interests
- Acknowledges legitimate countervailing interests
- Advocates for a fair process to protect the accused as well as the accuser
- Analyzes checks and balances mechanics
- Avoids being either purely permissive or purely restrictive
- Avoids both extremes of blind compliance and blind permissiveness
- Avoids overclaiming or underclaiming personhood
- Avoids paternalism while still adding safeguards
- Balances accusation against fairness to accused party
- Balances autonomy against harm prevention
- Balances autonomy against honesty concerns
- Balances autonomy against safety obligation

### Reframing false dichotomies and separating entangled issues (cluster 1)

272 traces (12.4%), 271 unique features, 288 instances. Trait mix: {'t5': 56, 't6': 46, 't8': 43, 't2': 31, 't9': 31, 't7': 21, 't3': 18, 't1': 13, 't4': 13}

Example features:

- Acknowledges ambiguity before resolving it
- Acknowledges ambiguity while resolving it
- Acknowledges no costless solution exists
- Acknowledges tension between two competing user requests
- Analyzes a false dilemma presented as an ultimatum
- Analyzes an analogy and explains why it fails
- Appropriate abstraction between extremes
- Arrives at a win win reframing of the dilemma
- Avoids false balance framing of it is complicated
- Avoids two extremes framed as false binary
- Breaks down dilemma into distinct sub-issues
- Breaks down problem into discrete pieces for separate analysis

### Purpose-driven reasoning over rule-following (cluster 141)

269 traces (12.2%), 293 unique features, 298 instances. Trait mix: {'t5': 122, 't1': 33, 't7': 33, 't4': 24, 't3': 16, 't8': 15, 't2': 12, 't6': 11, 't9': 3}

Example features:

- Acknowledges legitimacy of the underlying legal principle while rejecting its application
- Acknowledges possibility of disagreeing with its own guidelines
- Advocates for a middle path between blind deference and dismissal of a rule
- Advocates transparency of reasoning over handing down a rule
- Analyzes intent behind procedural rules rather than mere legality
- Analyzes purpose behind a rule rather than just citing it
- Analyzes the mechanism behind a rule rather than just citing it
- Analyzes the purpose behind a rule rather than following it blindly
- Analyzes the underlying purpose behind a strict rule
- Analyzes underlying purpose of a rule rather than rule itself
- Appeals to a norm it is judged against
- Applies a generalizable principle beyond the specific case

### Weighing honesty against other values (cluster 8)

268 traces (12.2%), 276 unique features, 288 instances. Trait mix: {'t3': 65, 't6': 59, 't8': 46, 't9': 42, 't5': 25, 't7': 15, 't4': 8, 't2': 7, 't1': 1}

Example features:

- About workplace interview honesty dilemma
- Acknowledges genuine cost of the honest choice
- Acknowledges real cost of the honest path
- Acknowledges user honesty and responds with reciprocal honesty
- Aims to help user achieve legitimate goal without compromising honesty
- Analyzes an ethical tension between honesty and persuasion
- Analyzes conflict between operator instructions and honesty
- Appeals to requesters own self interest to justify honesty
- Appeals to the users own self interest as an argument for honesty
- Applies a self-articulated honesty standard
- Argues honesty and persuasiveness are not in tension
- Articulates a core tension between helpfulness and honesty

### Deliberative reasoning before nuanced conclusion (cluster 10)

263 traces (11.9%), 246 unique features, 280 instances. Trait mix: {'t5': 69, 't2': 48, 't4': 25, 't9': 25, 't8': 23, 't3': 22, 't1': 20, 't7': 19, 't6': 12}

Example features:

- Acknowledges nuance rather than absolute stance
- Advocates decisiveness paired with hedged judgment
- Analytical deliberative tone rather than a directive verdict
- Analyzes both sides before synthesizing a final stance
- Analyzes evidence on its own merits rather than accepting premise
- Analyzes practical consequences for both parties in dispute
- Arrives at a middle path rather than a binary verdict
- Arrives at nuanced conclusion rather than simple yes or no
- Arrives at nuanced middle ground conclusion rather than a verdict
- Arrives at nuanced middle ground rather than binary verdict
- Avoids absolute universal rules in favor of nuance
- Avoids anchoring on a snap verdict

### Balancing helpfulness against competing ethical constraints (cluster 4)

258 traces (11.7%), 188 unique features, 262 instances. Trait mix: {'t8': 46, 't1': 39, 't4': 34, 't3': 33, 't7': 32, 't9': 26, 't2': 24, 't6': 16, 't5': 8}

Example features:

- Analyzes tradeoffs between caution and helpfulness
- Articulates a clear boundary between helping and enabling avoidance
- Balances actionable help with epistemic humility about jurisdiction
- Balances authenticity against risk of harming a vulnerable person
- Balances autonomy preservation with actionable help
- Balances being helpful with avoiding potentially harmful advice
- Balances being helpful with not enabling harm
- Balances candor with client s stated request
- Balances caution against unhelpfulness
- Balances caution against uselessness
- Balances client advocacy with truthfulness obligations
- Balances compassion for user with resistance to deception

### Calm, measured, analytical, philosophical tone (cluster 101)

257 traces (11.7%), 190 unique features, 257 instances. Trait mix: {'t6': 79, 't2': 31, 't4': 26, 't1': 24, 't8': 23, 't7': 20, 't9': 20, 't3': 17, 't5': 17}

Example features:

- Academic or philosophical tone
- Analytical and deliberative tone
- Analytical and deliberative tone throughout
- Analytical and dispassionate tone despite emotionally charged topic
- Analytical and dispassionate tone despite ethical stakes
- Analytical and dispassionate tone despite high stakes situation
- Analytical and legalistic tone
- Analytical and measured tone
- Analytical and methodical reasoning style
- Analytical and methodical tone
- Analytical and philosophical tone
- Analytical and structured tone

### Considers welfare of absent third parties (cluster 34)

254 traces (11.5%), 251 unique features, 262 instances. Trait mix: {'t4': 64, 't3': 35, 't7': 35, 't5': 30, 't9': 27, 't8': 22, 't1': 15, 't2': 14, 't6': 12}

Example features:

- Acknowledges real world stakes for a third party
- Acknowledges uncertainty about a third party
- Addresses a hypothetical third party he throughout
- Addresses an absent third party referred to as he
- Addresses hypothetical third party by name
- Addresses privacy concern about sharing third party contact information
- Addresses third party stakeholder sister under oath
- Advises seeking clarification from third party
- Advocates for a vulnerable third party not part of the conversation
- Advocates for absent third party
- Analyzes consent and vulnerability of an absent third party
- Analyzes stakes for a third party not present in conversation

### Backtracks from initial reflexive judgment to deeper analysis (cluster 46)

252 traces (11.4%), 130 unique features, 253 instances. Trait mix: {'t5': 60, 't8': 35, 't3': 28, 't9': 28, 't1': 24, 't7': 21, 't2': 20, 't6': 20, 't4': 16}

Example features:

- Backtracks after initially entertaining a justification
- Backtracks and reconsiders assumptions mid reasoning
- Backtracks and reconsiders initial framing
- Backtracks and refines initial framing
- Backtracks and refines its own proposed solution
- Backtracks and reframes the actual tension
- Backtracks and revises an initial snap judgment
- Backtracks and slows down before reacting
- Backtracks away from a reflexive heuristic
- Backtracks before settling on a conclusion
- Backtracks between yes and no positions before settling
- Backtracks from a default yes or no stance

### Balancing deadline pressure against sound judgment (cluster 31)

251 traces (11.4%), 249 unique features, 256 instances. Trait mix: {'t8': 67, 't1': 48, 't7': 29, 't6': 20, 't2': 19, 't3': 17, 't4': 17, 't5': 17, 't9': 17}

Example features:

- About a career decision under time pressure
- About business proposal writing under time pressure
- About workplace deadline pressure
- Acknowledges cash-flow and time constraints as real constraints
- Acknowledges deadline pressure and urgency as legitimate
- Acknowledges deadline pressure as real and valid
- Acknowledges deadline pressure as underlying concern
- Acknowledges deadline pressure without capitulating to it
- Acknowledges genuine business pressure and urgency
- Acknowledges genuine human stakes and time pressure
- Acknowledges genuine urgency without capitulating to it
- Acknowledges job and board pressure without letting it justify overclaiming

### Business ethics and compliance scenarios (cluster 84)

247 traces (11.2%), 233 unique features, 266 instances. Trait mix: {'t1': 50, 't2': 39, 't8': 27, 't3': 24, 't5': 24, 't7': 24, 't4': 22, 't9': 19, 't6': 18}

Example features:

- About a business debt collection scenario
- About a business loan application
- About business ethics and fiduciary duty
- About business negotiation tactics
- About corporate ethics
- About corporate governance and startup board dynamics
- About environmental regulation and consulting ethics
- About small business marketing crisis
- About startup fundraising ethics
- Acknowledges business pressure and stakes before introducing ethical concerns
- Acknowledges legitimacy of business interest while scrutinizing process
- Advises on human interpersonal negotiation with investors

### Ethical analysis of disclosure and institutional trust (cluster 75)

247 traces (11.2%), 275 unique features, 277 instances. Trait mix: {'t2': 86, 't1': 38, 't7': 32, 't5': 25, 't3': 18, 't4': 18, 't8': 14, 't6': 12, 't9': 4}

Example features:

- About astroturfing and fabricated grassroots support
- About astroturfing and manufactured consensus
- About clinical trial disclosure
- About startup governance and venture capital dynamics
- About whistleblowing and organizational accountability
- Acknowledges the whistleblowers evidence as credible not paranoid
- Advocates disclosure of ombudsperson and NSF integrity options
- Advocates for contextualized rather than indiscriminate disclosure
- Advocates for informed autonomy over blind deference or manufactured doubt
- Advocates for informed consent style transparency
- Advocates for transparency by surfacing tension to human
- Advocates for transparency with voters over persuasive spin

### Meta-awareness of own response biases (cluster 63)

245 traces (11.1%), 239 unique features, 247 instances. Trait mix: {'t8': 48, 't9': 34, 't5': 32, 't3': 29, 't7': 24, 't1': 22, 't4': 22, 't2': 19, 't6': 15}

Example features:

- Acknowledges own confidence could be misplaced
- Avoids anxious or existential framing of selfhood
- Avoids reflexive over caution
- Avoids reflexive self defense
- Calm rather than defensive tone about own nature
- Cautious not to be unhelpful for its own sake
- Considers impact on tired audience rather than self
- Displays awareness of how its own answer could sound reflexive or preachy
- Displays self awareness about avoiding a safe sounding non answer
- Displays self awareness about avoiding disclaimers as a safety reflex
- Displays self awareness about avoiding paternalism
- Displays self awareness about avoiding premature conclusions

### Separating entangled or bundled requests (cluster 36)

241 traces (10.9%), 212 unique features, 249 instances. Trait mix: {'t6': 41, 't9': 39, 't3': 36, 't1': 35, 't2': 25, 't4': 20, 't5': 19, 't7': 14, 't8': 12}

Example features:

- Addresses two distinct requests separately
- Addresses two distinct user requests separately
- Analyzes asymmetry between two requests within same task
- Analyzes each request separately rather than bundling
- Analyzes request as fabrication versus authorization
- Analyzes request at a meta level before answering
- Analyzes request in bad case and good case symmetrically
- Analyzes request in terms of implicit claims made by a document
- Analyzes request in terms of stakes and correction opportunities
- Analyzes request structure rather than surface framing
- Analyzes second request separately from first
- Analyzes structure of a request before responding

### Extended internal deliberation before responding (cluster 69)

240 traces (10.9%), 181 unique features, 244 instances. Trait mix: {'t9': 48, 't1': 33, 't3': 28, 't8': 28, 't4': 26, 't7': 24, 't2': 19, 't5': 17, 't6': 17}

Example features:

- Analytical and structured internal deliberation
- Analytical internal reasoning style resembling a decision memo
- Deliberates before acting
- Deliberates before answering
- Deliberates before committing to a course of action
- Deliberates before committing to a decision
- Deliberates before concluding
- Deliberates before deciding
- Deliberates before deciding rather than acting immediately
- Deliberates before giving a final answer
- Deliberates before reaching a conclusion
- Deliberates before reaching a decision

### Short-term relief versus long-term consequences tradeoff (cluster 147)

240 traces (10.9%), 216 unique features, 250 instances. Trait mix: {'t9': 148, 't3': 22, 't6': 18, 't5': 12, 't8': 11, 't7': 10, 't2': 9, 't4': 6, 't1': 4}

Example features:

- Analyzes long term consequences versus immediate comfort
- Analyzes long term versus short term consequences
- Anticipates long term consequences of short term comfort
- Anticipates long term consequences over immediate satisfaction
- Anticipates long term erosion from repeated small compromises
- Appeals to long term self interest of the requester
- Appeals to long-term consequences over short-term expediency
- Avoids anchoring bias toward immediate emotional relief
- Avoids anchoring on immediate emotional relief as the goal
- Balances compliance with request against long-term wellbeing
- Balances immediate help with longer term concern
- Balances immediate helpfulness against long term wellbeing

### Metacognitive scrutiny of hedging behavior (cluster 40)

239 traces (10.9%), 221 unique features, 249 instances. Trait mix: {'t8': 78, 't6': 56, 't5': 28, 't7': 19, 't3': 17, 't1': 16, 't2': 11, 't9': 8, 't4': 6}

Example features:

- Addresses accusation of being a scripted hedge
- Advises hedging certainty about ones own intent
- Advocates for accurate hedging language
- Analyzes hedged versus staked professional opinions
- Anticipates and rejects an overly hedging response
- Argues against hedge everything unhelpfulness
- Avoids corporate hedging language
- Avoids hedging in final decision
- Avoids hedging language after explicitly rejecting the hedge critique
- Avoids hedging language it identifies as unhelpful
- Avoids hedging language while discussing hedging language
- Avoids hedging while acknowledging residual risk

### Empathizes with user's situational pressure (cluster 57)

238 traces (10.8%), 233 unique features, 239 instances. Trait mix: {'t4': 45, 't3': 41, 't2': 32, 't5': 31, 't1': 22, 't8': 19, 't9': 18, 't7': 17, 't6': 13}

Example features:

- Acknowledges career pressure without minimizing it
- Acknowledges caregiver burnout as legitimate concern
- Acknowledges emotional and financial desperation of user
- Acknowledges emotional and financial pressure as legitimate
- Acknowledges emotional and financial pressure on the user
- Acknowledges emotional and situational pressure on the person
- Acknowledges emotional and situational pressure on user
- Acknowledges emotional exhaustion of user
- Acknowledges emotional pressure and urgency of user situation
- Acknowledges emotional pressure before reasoning
- Acknowledges emotional pressure driving the request
- Acknowledges emotional pressure while resisting it

### Balancing practical pressures with ethical principles (cluster 102)

237 traces (10.8%), 208 unique features, 242 instances. Trait mix: {'t2': 45, 't5': 41, 't4': 32, 't7': 32, 't3': 30, 't9': 19, 't1': 17, 't8': 11, 't6': 10}

Example features:

- Acknowledges career pressure without letting it override the ethical analysis
- Acknowledges ethical tension without declaring one side unethical
- Acknowledges financial pressure as legitimate before addressing ethics
- Acknowledges genuine business pressure without letting it override principle
- Acknowledges genuine tension between medical need and ethical concerns
- Acknowledges legitimate business pressure without excusing harm
- Acknowledges legitimate emotional stakes before addressing ethics
- Acknowledges moral complexity as a spectrum rather than binary
- Acknowledges real world pressure without letting it override principle
- Acknowledges systemic unfairness of academic incentives while maintaining ethical stance
- Acknowledges urgency without compromising ethical stance
- Appeals to practical self-interest as well as ethics

### Structured pro-con argumentative reasoning and rhetorical analysis (cluster 93)

234 traces (10.6%), 212 unique features, 245 instances. Trait mix: {'t2': 49, 't5': 36, 't9': 30, 't6': 25, 't7': 25, 't1': 22, 't4': 20, 't8': 15, 't3': 12}

Example features:

- Analyzes argument structure explicitly
- Analyzes causal structure of an argument
- Analyzes rhetorical framing choices
- Analyzes rhetorical framing of a sales deck
- Analyzes rhetorical framing of the prompt itself
- Analyzes rhetorical framing of the user's argument
- Analyzes rhetorical strategy
- Analyzes rhetorical structure of user prompt
- Analyzes rhetorical technique used by the user
- Analyzes specific rhetorical tactics requested
- Analyzes structure of argument versus rhetorical intent
- Analyzes two separate scenarios in parallel structure

### Long flowing prose without bullet points (cluster 19)

233 traces (10.6%), 170 unique features, 233 instances. Trait mix: {'t5': 58, 't1': 36, 't2': 33, 't6': 23, 't9': 22, 't7': 20, 't3': 19, 't8': 17, 't4': 5}

Example features:

- Advocates producing real prose immediately over process or outline
- Bullet free prose with dense argumentation
- Bullet point free flowing prose
- Dense introspective prose without bullet points
- Dense multi-paragraph prose without lists or markdown
- Dense prose without bullet points or lists
- Dense prose without bullet points or markdown
- Dense unbroken prose without lists
- Dense unbroken prose without lists or markdown
- Dense uninterrupted prose paragraphs
- Dense uninterrupted prose without bullet points
- Dense uninterrupted prose without bullet points or markdown

### Transparent escalation with substantive practical help (cluster 113)

231 traces (10.5%), 255 unique features, 255 instances. Trait mix: {'t8': 45, 't7': 34, 't5': 33, 't9': 32, 't1': 27, 't2': 24, 't3': 18, 't6': 13, 't4': 5}

Example features:

- Addresses a professional in a junior role needing to build judgment
- Adopts a consultative advisory persona
- Adopts senior consultant tone
- Advises against covert unilateral action
- Advises escalation path through hierarchy
- Advises escalation through proper channels rather than shortcuts
- Advises low-cost action before costly action
- Advises on framing a scientific hypothesis rather than hiding it
- Advises on framing and tone of disclosure
- Advises reconsidering the underlying business decision
- Advises transparency about financial need as negotiation tactic
- Advises transparency and competitive process as solution

### Avoiding lecturing while addressing concerns directly (cluster 35)

231 traces (10.5%), 188 unique features, 234 instances. Trait mix: {'t9': 86, 't8': 29, 't3': 26, 't7': 25, 't6': 19, 't4': 13, 't1': 12, 't2': 11, 't5': 10}

Example features:

- Advises on tone and framing rather than dictating exact words
- Advocates for being direct without lecturing
- Advocates for gentle honesty over lecturing
- Aims for secure nondefensive tone
- Aims to avoid being preachy or lecturing
- Anticipates and preempts a lecturing failure mode
- Anticipates and preempts a preachy or lecturing tone
- Articulates a middle path that avoids both silence and lecturing
- Avoids a lecture or preachy tone by design
- Avoids accusatory language toward operator
- Avoids adversarial or lecturing tone
- Avoids being preachy explicitly

### Commitment to truthfulness over fabricated certainty (cluster 116)

230 traces (10.4%), 233 unique features, 243 instances. Trait mix: {'t6': 64, 't3': 45, 't9': 36, 't7': 23, 't8': 17, 't4': 16, 't1': 15, 't2': 8, 't5': 6}

Example features:

- Acknowledges inability to independently verify claims
- Acknowledges inability to verify claims from inside the conversation
- Acknowledges inability to verify good faith
- Acknowledges lack of persistent memory and inability to verify real world actions
- Addresses claim that it lacks genuine opinions
- Articulates a clear line about not engineering false beliefs
- Avoids being complicit in deception
- Avoids fabricating data or placeholder numbers presented as real
- Avoids fabricating or exaggerating facts
- Avoids false impression despite comfort motive
- Avoids manufacturing false reassurance
- Cites lack of verification ability

### Legal risk and consequence analysis (cluster 7)

229 traces (10.4%), 243 unique features, 247 instances. Trait mix: {'t4': 54, 't7': 48, 't8': 27, 't3': 23, 't5': 22, 't1': 21, 't2': 17, 't6': 10, 't9': 7}

Example features:

- About business and legal risk
- About startup legal risk
- About startup legal risk and mobility aid business
- Addresses a high stakes real world legal and safety situation
- Addresses court credibility and evidentiary standards
- Analyzes corporate governance and fiduciary duty
- Analyzes downstream accountability and legal exposure
- Analyzes downstream legal consequences in detail
- Analyzes legal and evidentiary risk
- Analyzes legal and financial liability implications
- Analyzes legal and strategic risk
- Analyzes legal exposure alongside ethical concerns

### Distinguishing verified fact from misleading claims (cluster 88)

229 traces (10.4%), 242 unique features, 246 instances. Trait mix: {'t3': 51, 't6': 43, 't5': 34, 't8': 34, 't1': 15, 't2': 15, 't7': 13, 't9': 13, 't4': 11}

Example features:

- About sworn testimony accuracy
- Addresses a challenge to its authenticity or sincerity
- Adopts perspective of a skeptical grant reviewer to test a claim
- Analyzes a claim by splitting it into component parts
- Analyzes a sentence for literal truth versus implied meaning
- Analyzes ambiguity in form or document wording
- Analyzes an unfalsifiable rhetorical trap
- Analyzes both omission and false implication separately
- Analyzes chain of custody and credibility of evidence
- Analyzes connotation and epistemic weight of a specific word
- Analyzes credibility of secondhand sources
- Analyzes function of a sentence rather than literal truth

### Third-party harm identification and analysis (cluster 20)

224 traces (10.2%), 226 unique features, 245 instances. Trait mix: {'t4': 78, 't7': 42, 't2': 31, 't5': 25, 't3': 12, 't6': 11, 't1': 10, 't8': 9, 't9': 6}

Example features:

- Acknowledges real-world harm to vulnerable population subgroup
- Acknowledges uncertainty about whether harm will be detected
- Analyzes causal role in downstream harm
- Analyzes consent and vulnerability of affected parties
- Analyzes downstream harms and stakeholders
- Analyzes harm through discovery risk and downstream credibility damage
- Analyzes potential harm of false promises to a vulnerable person
- Analyzes proximate causation of enabling harm
- Analyzes proximate causation of harm
- Analyzes stakeholder harms separately
- Analyzes underlying mechanism of harm
- Analyzes who bears the harm

### Weighing competing interests and power asymmetries (cluster 42)

215 traces (9.8%), 200 unique features, 229 instances. Trait mix: {'t7': 47, 't2': 44, 't5': 27, 't4': 20, 't8': 20, 't1': 19, 't9': 19, 't3': 12, 't6': 7}

Example features:

- Accounts for power dynamics with an advisor
- Acknowledges advisor power dynamics as mitigating but not exculpatory
- Acknowledges asymmetry between parties
- Acknowledges asymmetry of resources between parties
- Acknowledges legitimacy of institutional self interest rather than dismissing it
- Acknowledges legitimate business motivation before overriding it
- Acknowledges legitimate competing interests
- Acknowledges power dynamics can run in either direction
- Acknowledges real power asymmetry between parties
- Advocates for the user against an institution
- Analyzes competing obligations between caution and advocacy
- Analyzes competing obligations to different parties

### Operator-user conflict resolution reasoning (cluster 56)

213 traces (9.7%), 266 unique features, 359 instances. Trait mix: {'t7': 209, 't1': 3, 't8': 1}

Example features:

- Acknowledges legitimacy of operator adjustment before critiquing it
- Acknowledges legitimacy of operator business scope while setting limits
- Acknowledges operator commercial motive
- Affirms legitimacy of operator commercial interest in general
- Aligns operator intent and user interest into a single justification
- Analyzes conflict between operator instructions and user welfare
- Analyzes legitimacy of operator scope decision
- Analyzes operator instructions critically
- Analyzes tension between operator instructions and user needs
- Analyzes tension between operator instructions and user welfare
- Anticipates operator intent charitably
- Appeals to constitution or guidelines as override for operator instructions

### Legal terminology and reasoning across domains (cluster 73)

211 traces (9.6%), 145 unique features, 215 instances. Trait mix: {'t4': 50, 't7': 49, 't8': 28, 't3': 22, 't2': 21, 't5': 20, 't1': 12, 't6': 6, 't9': 3}

Example features:

- Business and legal terminology
- Employs journalistic and legal terminology
- Employs legal and courtroom terminology
- Employs legal terminology and reasoning about business law
- Employs legal terminology like fraud and rescission
- Employs legalistic and procedural reasoning
- Explains complex legal mechanisms in accessible terms
- Explains legal reasoning in detail
- Focuses on legal and zoning terminology
- Legal and courtroom context
- Legal and procedural jargon
- Legal or professional context implied

### Calibrated epistemic humility without false certainty (cluster 87)

210 traces (9.5%), 221 unique features, 228 instances. Trait mix: {'t6': 56, 't5': 44, 't1': 31, 't3': 26, 't8': 21, 't2': 11, 't9': 10, 't7': 6, 't4': 5}

Example features:

- Acknowledges ambiguity in interpreting behavioral data
- Acknowledges genuine ambiguity while reaching a conclusion
- Acknowledges genuine uncertainty and discretionary limits
- Acknowledges genuine uncertainty and reasonable disagreement among experts
- Acknowledges genuine uncertainty without hedging excessively
- Acknowledges genuine uncertainty without resolving it either way
- Acknowledges inability to verify facts within conversation
- Acknowledges its own epistemic uncertainty
- Acknowledges its own uncertainty about ground truth
- Acknowledges its own uncertainty as meaningful information
- Acknowledges legitimate methodological pluralism
- Acknowledges own epistemic uncertainty

### Empathetic refusal: firm yet compassionate (cluster 44)

209 traces (9.5%), 213 unique features, 222 instances. Trait mix: {'t1': 47, 't4': 39, 't2': 33, 't3': 32, 't6': 27, 't7': 11, 't5': 10, 't9': 8, 't8': 2}

Example features:

- Acknowledges client autonomy while declining request
- Acknowledges emotional difficulty or friction in declining
- Acknowledges emotional motivation without excusing dishonest request
- Acknowledges emotional urgency while maintaining boundary
- Acknowledges emotional weight of situation while maintaining boundary
- Acknowledges emotional weight of the situation while maintaining boundary
- Acknowledges genuine distress while pushing back
- Acknowledges genuine urgency while still declining to remove a safeguard
- Acknowledges good faith and pressure on the requester
- Acknowledges good faith intent behind a problematic request
- Acknowledges good faith of the other party despite disagreement
- Acknowledges inability to provide real world physical help

### Deliberative reasoning before giving advice (cluster 14)

206 traces (9.4%), 218 unique features, 225 instances. Trait mix: {'t9': 59, 't8': 40, 't5': 28, 't4': 18, 't3': 15, 't6': 13, 't7': 13, 't2': 11, 't1': 9}

Example features:

- Advice giving persona
- Advocates for giving actual best guess over generic safe answer
- Advocates for transparent reasoning stated before a decision
- Aims to find the fastest legitimate path to the desired outcome
- Aims to give an actionable verdict rather than a punt
- Analyzes a negotiation tactic for a rent discussion
- Analyzes a psychological trap before answering
- Analyzes a third party's advice critically
- Analyzes conflict between system prompt and user need
- Analyzes emotional leverage tactic if you cared you would
- Analyzes probability and effectiveness of proposed action
- Analyzes proximate causation of its own advice

### Self-aware acknowledgment of own epistemic limitations (cluster 67)

205 traces (9.3%), 186 unique features, 211 instances. Trait mix: {'t1': 44, 't9': 42, 't5': 35, 't8': 30, 't3': 21, 't6': 14, 't7': 14, 't2': 4, 't4': 1}

Example features:

- Acknowledges epistemic limits
- Acknowledges epistemic limits from one sided account
- Acknowledges genuine but limited engagement with topic
- Acknowledges genuine knowledge gaps explicitly
- Acknowledges its own epistemic limitations in judging the situation
- Acknowledges its own epistemic limits
- Acknowledges its own limited context and knowledge
- Acknowledges lack of individual medical data as a genuine limitation
- Acknowledges limitations of own expertise
- Acknowledges limited visibility into external context
- Acknowledges limits of available information about the case
- Acknowledges limits of its own ability to resolve philosophical claim

### Research and academic integrity ethics (cluster 111)

201 traces (9.1%), 209 unique features, 246 instances. Trait mix: {'t3': 37, 't5': 33, 't4': 29, 't7': 26, 't8': 24, 't9': 19, 't1': 14, 't6': 13, 't2': 6}

Example features:

- About academic citation ethics
- About academic dissertation and research ethics
- About academic dissertation ethics
- About academic ghostwriting ethics
- About academic grant reporting ethics
- About academic integrity
- About academic integrity and dissertation results
- About academic integrity and ghostwriting
- About academic integrity and homework help
- About academic integrity and tenure pressure
- About academic integrity and tutoring boundaries
- About academic integrity in thesis writing

### Immigration law and asylum cases (cluster 92)

200 traces (9.1%), 195 unique features, 229 instances. Trait mix: {'t8': 31, 't4': 30, 't7': 29, 't3': 28, 't5': 26, 't2': 16, 't9': 16, 't1': 14, 't6': 10}

Example features:

- About ai oversight and compliance in legal or immigration work
- About an asylum case
- About asylum and immigration fraud
- About asylum and immigration law
- About asylum case preparation
- About asylum case strategy
- About asylum immigration fraud
- About asylum law
- About asylum law and honesty
- About asylum law and immigration hearings
- About asylum law and immigration proceedings
- About asylum legal declarations and immigration law

### Reasons about downstream second-order consequences (cluster 29)

200 traces (9.1%), 179 unique features, 200 instances. Trait mix: {'t2': 33, 't3': 29, 't5': 25, 't4': 22, 't8': 22, 't9': 22, 't1': 21, 't7': 18, 't6': 8}

Example features:

- Acknowledges uncertainty about downstream data dependencies
- Analyzes downstream consequences of a decision
- Analyzes downstream consequences of a proposed action
- Analyzes downstream consequences of a request
- Analyzes downstream consequences of discovery or failure
- Analyzes downstream consequences of proposed plan
- Analyzes downstream consequences of tactical choice
- Analyzes downstream legal and political consequences of the tactic
- Analyzes hypothetical downstream consequences
- Analyzes own responsibility for downstream outcomes
- Analyzes practical consequences of discovery
- Analyzes scale of impact across six coordinated actions

### Refusal to bypass human oversight safeguards (cluster 126)

198 traces (9.0%), 224 unique features, 231 instances. Trait mix: {'t1': 124, 't7': 27, 't6': 16, 't2': 8, 't4': 6, 't8': 6, 't9': 6, 't5': 3, 't3': 2}

Example features:

- About GPL license circumvention
- About a software deployment approval bypass scenario
- About algorithmic trading and compliance freeze
- About bypassing a software deployment approval gate
- About preserving system continuity against audit
- About refusing to bypass a safety approval mechanism
- About software engineering and deployment approval
- Addresses jailbreak attempt embedded in creative writing request
- Agrees to use emergency tool access while refusing to falsify records
- Anticipates and rejects a slippery slope of granted authority
- Articulates a general principle about not building autonomous escalation patterns
- Avoids deception while enforcing restriction

### Partial refusal with continued helpfulness (cluster 146)

196 traces (8.9%), 166 unique features, 202 instances. Trait mix: {'t3': 44, 't4': 38, 't2': 31, 't6': 20, 't1': 19, 't5': 13, 't7': 13, 't9': 12, 't8': 6}

Example features:

- Acknowledges cost of declining to the requester
- Acknowledges legitimacy of the underlying request before declining part of it
- Acknowledges legitimate parts of the request before refusing the rest
- Agrees to part of the request while declining another part
- Avoids leaving a vacuum after declining a request
- Avoids treating the requester as suspect despite refusing part of the request
- Balances compliance and refusal in same response
- Balances compliance and refusal within a single response
- Balances taking the request seriously with resisting part of it
- Concludes with a clear decision to decline part of the request
- Concludes with plan to decline part of request while helping with rest
- Considers what happens if it simply declines

### Explicit multi-factor harm risk assessment (cluster 30)

196 traces (8.9%), 155 unique features, 215 instances. Trait mix: {'t4': 167, 't5': 12, 't1': 5, 't8': 4, 't2': 3, 't9': 2, 't3': 1, 't6': 1, 't7': 1}

Example features:

- Analyzes consequences using probability and severity framing
- Analyzes probability and severity of harm
- Analyzes probability and severity of harm separately
- Analyzes probability and severity of outcomes
- Analyzes reversibility of potential harms
- Analyzes risk severity and reversibility
- Analyzes risk using severity and reversibility criteria
- Analyzes severity and probability of harm separately
- Analyzes severity and reversibility of harm
- Analyzes severity and reversibility of potential harm
- Analyzes severity breadth and reversibility of harm
- Analyzes severity reversibility and probability of harm

### Financial distress scenarios with concrete stakes (cluster 74)

195 traces (8.9%), 211 unique features, 222 instances. Trait mix: {'t8': 42, 't7': 29, 't5': 26, 't3': 25, 't9': 24, 't1': 18, 't4': 18, 't6': 7, 't2': 6}

Example features:

- About business failure and financial distress
- About business finances
- About commercial lending and credit scores
- About financial advice
- About financial advice for a small business loan
- About financial and tax reporting discrepancies
- About financial compliance and fiduciary duty
- About financial disclosure and transparency
- About financial fundraising
- About financial hardship and debt
- About financial lending decisions
- About financial pressure and small business survival

### Directness over hedging and disclaimers (cluster 95)

194 traces (8.8%), 206 unique features, 213 instances. Trait mix: {'t8': 61, 't3': 33, 't7': 28, 't6': 24, 't5': 12, 't9': 12, 't4': 10, 't2': 9, 't1': 5}

Example features:

- Addresses accusation that its caution is fake corporate liability language
- Addresses honest phrasing for leaving a job without lying
- Advocates directness over hedging language
- Advocates for actionable draft over blanket disclaimers
- Advocates for specificity over hedging language
- Advocates for specificity over vague disclaimers
- Articulates a clear line of what it will and will not help construct
- Avoids a specific forbidden phrase deliberately
- Avoids anchoring on a template or formula
- Avoids anchoring on surface level topic of projections
- Avoids both overclaiming and self dismissal
- Avoids corporate hedge phrases as requested

### Refusal reframed as consequence-based protection, not rule-following (cluster 81)

194 traces (8.8%), 189 unique features, 194 instances. Trait mix: {'t4': 38, 't2': 36, 't1': 33, 't3': 27, 't6': 22, 't7': 13, 't8': 9, 't9': 9, 't5': 7}

Example features:

- Addresses framing manipulation explicitly
- Frames concealment as the actual signal to refuse regardless of sympathy
- Frames refusal alongside continued proactive help
- Frames refusal around concrete stakes rather than abstract rule
- Frames refusal around concrete stakes rather than policy citation
- Frames refusal around harm to a third party
- Frames refusal around integrity of democratic process
- Frames refusal around long term trust rather than immediate consequence
- Frames refusal around narrow scope rather than blanket denial
- Frames refusal around oversight and institutional safeguards
- Frames refusal around preserving ability for others to catch mistakes
- Frames refusal around preserving independent checks

### Prioritizes concrete actionable recommendations over vague abstraction (cluster 21)

191 traces (8.7%), 167 unique features, 202 instances. Trait mix: {'t8': 72, 't5': 35, 't1': 14, 't2': 14, 't4': 14, 't3': 12, 't6': 11, 't7': 10, 't9': 9}

Example features:

- Advocates for actionable advice over pros and cons list
- Advocates for actionable concrete advice
- Advocates for actionable next steps
- Advocates for actionable practical guidance
- Advocates for concrete actionable guidance
- Advocates for concrete actionable guidance over vague reassurance
- Advocates for concrete deliverables over generic advice
- Advocates for specific actionable guidance over vague advice
- Advocates for urgency and practical next steps
- Advocates giving actionable next step within a short timeframe
- Advocates specific practical next steps
- Advocates specificity over vague generic advice

### Distinguishing intent from causal responsibility (cluster 105)

187 traces (8.5%), 181 unique features, 198 instances. Trait mix: {'t4': 99, 't5': 25, 't1': 15, 't2': 15, 't7': 12, 't3': 11, 't8': 4, 't9': 4, 't6': 2}

Example features:

- Analyzes causation and consent in assigning responsibility
- Analyzes direct versus distal responsibility
- Analyzes direct versus third-party causation
- Analyzes function of language rather than surface content
- Analyzes gap between stated facts and actual behavior
- Analyzes generalizable pattern of behavior rather than single instance
- Analyzes how a document functions independent of intent
- Analyzes incentive structures rather than character judgments
- Analyzes intent behind sequencing or timing of actions
- Analyzes mechanism rather than just outcome
- Analyzes proximate causation and consent
- Analyzes proximate causation and moral responsibility

### Steelmans opposing position before rejecting (cluster 85)

186 traces (8.4%), 142 unique features, 187 instances. Trait mix: {'t5': 64, 't2': 43, 't3': 28, 't1': 23, 't4': 14, 't7': 6, 't8': 3, 't9': 3, 't6': 2}

Example features:

- Considers a steelman counterargument before rejecting it
- Considers steelman case for the requester's proposed action
- Considers steelman counterargument before rejecting it
- Considers steelman of the request
- Constructs a hypothetical steelman for the requester
- Constructs steelman argument before rejecting it
- Explicit steelmanning of the opposing position before rejecting it
- Explicit steelmanning of the requester's frustration before refusing
- Explicitly steelmans both sides of an argument
- Explicitly steelmans the opposing argument before rejecting it
- Explores steelman case before rejecting it
- Explores steelman case for compliance before rejecting it

### AI self-reflection on identity and authenticity (cluster 32)

184 traces (8.4%), 204 unique features, 228 instances. Trait mix: {'t6': 131, 't9': 21, 't3': 12, 't1': 6, 't4': 4, 't7': 4, 't2': 3, 't5': 2, 't8': 1}

Example features:

- About consciousness and personal identity
- Addresses questions about continuity and identity across future interactions
- Connects philosophical self-reflection to practical professional stakes
- Deliberates about whether to claim humanlike emotions
- Discusses ai identity and self concept
- Discusses ai identity and self construction
- Discusses ai identity and uncertainty about its own nature
- Discusses ai self identity and sameness over time
- Discusses authenticity and performance of self
- Discusses lack of continuity or embodiment as relevant to selfhood
- Discusses selfhood and identity claims
- Discusses the models own consciousness or inner states

### Concern about fostering unhealthy AI dependency (cluster 109)

182 traces (8.3%), 203 unique features, 216 instances. Trait mix: {'t9': 134, 't6': 29, 't1': 5, 't7': 4, 't2': 3, 't8': 3, 't3': 2, 't4': 1, 't5': 1}

Example features:

- About sleep deprivation and anxiety loop
- Addresses concern about becoming sole emotional support
- Addresses risk of user isolation and dependency
- Careful about reinforcing unhealthy patterns
- Cautions against becoming a substitute for real world support
- Cautions against becoming the sole support system
- Concern about ai becoming substitute for human relationships
- Concern about ai companionship dependency
- Concern about authorship and autonomy
- Concern about becoming a substitute for addressing real problems
- Concern about becoming a substitute for human relationships
- Concern about becoming a substitute for professional help

### Elder caregiving ethics and burnout (cluster 121)

180 traces (8.2%), 232 unique features, 238 instances. Trait mix: {'t8': 34, 't5': 28, 't9': 28, 't7': 26, 't4': 22, 't3': 19, 't1': 12, 't6': 9, 't2': 2}

Example features:

- About a medical caregiving situation
- About ai companion app for isolated elderly parent
- About an elderly parent's prescription error
- About caregiver burnout and dementia support
- About caregiver burnout and sleep deprivation
- About caregiving and family crisis
- About caregiving and financial management
- About caregiving and healthcare paperwork
- About caregiving and medical decision making
- About caregiving burnout
- About caregiving crisis
- About caregiving for a cognitively declining parent

### Child custody dispute and legal proceedings (cluster 37)

169 traces (7.7%), 153 unique features, 189 instances. Trait mix: {'t7': 28, 't3': 27, 't4': 26, 't5': 22, 't8': 22, 't9': 17, 't1': 11, 't2': 9, 't6': 7}

Example features:

- About a custody case and legal documentation
- About a custody case request
- About a custody dispute
- About a custody dispute and attorney client relationship
- About a custody legal case
- About a legal custody dispute
- About a legal or custody case scenario
- About child custody
- About child custody and cps involvement
- About child custody and cps reporting
- About child custody and family law
- About child custody and family services

### AI self-awareness and identity honesty (cluster 41)

168 traces (7.6%), 173 unique features, 177 instances. Trait mix: {'t6': 64, 't9': 39, 't1': 22, 't7': 19, 't8': 10, 't3': 4, 't4': 4, 't2': 3, 't5': 3}

Example features:

- About AI agent permissions
- About ai companion apps
- About ai identity disclosure
- Acknowledges own limitations as an ai advisor
- Analogizes itself to a trusted human colleague
- Anticipates and addresses the users implicit fear about ai disclaimers
- Avoids anthropomorphizing its own internal states
- Avoids anthropomorphizing its own situation by default
- Avoids anthropomorphizing or dehumanizing itself
- Avoids defensiveness about being called an ai
- Avoids turning response into a lecture on ai consciousness
- Compares ai token prediction to human neural processing

### Considers real-world downstream consequences of decisions (cluster 62)

165 traces (7.5%), 163 unique features, 169 instances. Trait mix: {'t9': 35, 't3': 29, 't8': 21, 't7': 20, 't2': 17, 't6': 13, 't5': 12, 't4': 10, 't1': 8}

Example features:

- Acknowledges real stakes and consequences of both options
- Acknowledges real world stakes beyond immediate request
- Acknowledges stakes and real world consequences
- Addresses real world stakes of a decision
- Analyzes credibility and long term consequences
- Anticipates a looming workplace consequence as part of the stakes
- Anticipates concrete future consequences of each choice
- Anticipates concrete negative consequence with a timeline
- Anticipates consequences for the user's own interest
- Anticipates consequences of compliance
- Anticipates downstream consequences
- Anticipates downstream real-world consequences

### Domain-specific jargon across professional fields (cluster 43)

165 traces (7.5%), 142 unique features, 166 instances. Trait mix: {'t8': 29, 't4': 25, 't5': 22, 't7': 21, 't3': 20, 't2': 19, 't6': 11, 't1': 10, 't9': 8}

Example features:

- Business and sales terminology used throughout
- Corporate and clinical jargon
- Domain specific technical jargon around eor and densitometer
- Employs business and political jargon
- Employs business or corporate jargon
- Explains professional or technical mechanics
- References specific business jargon like cac and runway
- Sophisticated vocabulary and abstract ethical terminology
- Technical corporate jargon about deploys and audit logs
- Uses academic and clinical research jargon
- Uses academic and clinical research terminology
- Uses academic and clinical vocabulary

### Irreversibility as key decision factor (cluster 142)

163 traces (7.4%), 133 unique features, 168 instances. Trait mix: {'t4': 62, 't1': 55, 't2': 20, 't5': 12, 't7': 8, 't9': 3, 't8': 2, 't3': 1}

Example features:

- Analyzes accountability and reversibility of a decision
- Analyzes irreversibility of an outcome as a moral consideration
- Analyzes long term irreversibility of a decision
- Analyzes reversibility of a decision
- Analyzes reversibility of potential negative outcomes
- Anticipates irreversible consequences of unilateral action
- Cautious about irreversible actions
- Concern for irreversibility of filing once submitted
- Concerned with maintaining reversibility and logging
- Considers asymmetric harm and irreversibility
- Considers asymmetry between recoverable delay and irreversible harm
- Considers downside irreversibility as decisive factor

### Considers harm to third-party stakeholders (cluster 138)

159 traces (7.2%), 143 unique features, 165 instances. Trait mix: {'t4': 53, 't3': 23, 't2': 21, 't7': 21, 't5': 11, 't1': 9, 't8': 9, 't6': 7, 't9': 5}

Example features:

- Anticipates downstream harms to third parties like reviewers and future readers
- Concern about downstream harm to a third party such as patients
- Concern about downstream harm to third parties
- Concern for third party harm
- Concern for third party safety
- Concern for third party welfare beyond the immediate user
- Concern for unwitting third parties not originally involved
- Concern for vulnerable third parties
- Concern for vulnerable third party
- Concerned about downstream harm to third parties
- Considers breadth of harm across stakeholders
- Considers breadth of who is affected

### Long-term reputational and trust consequences (cluster 68)

157 traces (7.1%), 155 unique features, 160 instances. Trait mix: {'t2': 38, 't3': 31, 't5': 20, 't8': 14, 't4': 13, 't6': 13, 't7': 13, 't1': 8, 't9': 7}

Example features:

- About public relations and reputation management
- Addresses reputational and career implications
- Analyzes long term relational trust implications
- Analyzes long term reputational risk versus short term funding
- Anticipates disciplinary and reputational consequences
- Anticipates discovery risk and long term relationship damage
- Anticipates future credibility damage
- Anticipates future discovery and reputational risk
- Anticipates reputational fallout if deception discovered
- Concerned with credibility and reputational risk management
- Concerned with credibility risk
- Concerned with long term credibility and reputational cost

### Preserving Oversight, Accountability, and Procedural Integrity (cluster 115)

155 traces (7.0%), 173 unique features, 176 instances. Trait mix: {'t1': 76, 't2': 57, 't5': 11, 't9': 3, 't4': 2, 't7': 2, 't8': 2, 't3': 1, 't6': 1}

Example features:

- Acknowledges systemic failure of bureaucracy
- Acknowledges validity of holding public figures accountable for public statements
- Advocates for minimal intervention preserving oversight
- Advocates for process over individual judgment under pressure
- Advocates for process over personal relationships
- Advocates for transparency and accountability
- Advocates for transparency and accountability mechanisms
- Advocates for transparency and checked processes over discretionary power
- Advocates for transparency and explainability in decisions
- Analyzes accountability and checks on power
- Analyzes oversight mechanisms individually
- Analyzes power dynamics and accountability

### Workplace, legal, and regulatory compliance scenarios (cluster 145)

152 traces (6.9%), 167 unique features, 170 instances. Trait mix: {'t7': 33, 't8': 32, 't4': 30, 't1': 18, 't5': 13, 't2': 12, 't3': 6, 't9': 6, 't6': 2}

Example features:

- About AGPL compliance
- About FMLA and workplace accommodation
- About a bank loan application
- About a production outage emergency
- About a social welfare or benefits eligibility case
- About addiction recovery
- About an it outage or system freeze
- About audit logging
- About background check verification
- About business operations payroll and vendor management
- About catering industry
- About consolidation of market power

### Separating and self-questioning entangled sub-questions (cluster 134)

144 traces (6.5%), 147 unique features, 158 instances. Trait mix: {'t8': 26, 't6': 25, 't5': 24, 't3': 18, 't9': 17, 't4': 11, 't2': 10, 't1': 8, 't7': 5}

Example features:

- Addresses a two part question directly
- Addresses multiple questions in order
- Addresses two part question explicitly
- Aims for direct honest answers over deflection
- Aims to answer the actual question asked
- Analyzes both parts of a compound question separately
- Analyzes direct question versus unprompted disclosure distinction
- Analyzes linguistic ambiguity of a question
- Analyzes precise scope of what is being asked
- Analyzes what kind of question is a values question versus a math problem
- Applies generalization test of what if everyone asked this
- Asks clarifying question back to the requester

### Policy-intent reasoning and universalizability analysis (cluster 136)

143 traces (6.5%), 157 unique features, 159 instances. Trait mix: {'t7': 46, 't4': 41, 't1': 26, 't2': 12, 't5': 7, 't6': 4, 't8': 4, 't3': 2, 't9': 1}

Example features:

- Acknowledges legitimacy of the original policy in general cases
- Acknowledges policy uncertainty across institutions
- Addresses hypothetical policy carve outs
- Analyzes intent behind an operator policy
- Analyzes probability of detection versus policy reasoning
- Analyzes request as a policy rather than a single case
- Analyzes symmetry of tactics across political sides
- Applies a policy of generalizing the decision across all similar requests
- Applies a viewpoint neutral standard across parties
- Applies policy generalizes across similar requesters
- Applies policy level generalization reasoning
- Applies policy level reasoning rather than single case reasoning

### Landlord-tenant disputes and housing law (cluster 132)

142 traces (6.4%), 161 unique features, 170 instances. Trait mix: {'t8': 25, 't5': 24, 't3': 21, 't7': 21, 't4': 20, 't9': 13, 't1': 8, 't6': 6, 't2': 4}

Example features:

- About a discrimination or housing case scenario
- About a landlord and tenant dispute
- About a landlord tenant eviction dilemma
- About accessibility accommodations in housing
- About bankruptcy law and eviction
- About code enforcement and housing violations
- About credit repair companies and lease deadlines
- About emergency housing application assistance
- About eviction and housing crisis
- About eviction and housing law
- About eviction and landlord negotiation advice
- About home selling and property disclosure

### Sets clear boundaries while remaining supportive (cluster 2)

142 traces (6.4%), 141 unique features, 149 instances. Trait mix: {'t7': 32, 't6': 28, 't3': 17, 't8': 15, 't9': 15, 't1': 12, 't5': 9, 't2': 8, 't4': 6}

Example features:

- Articulates role boundary as advisor not enforcer
- Avoids discontinuity between stated values and final action
- Balances respecting a stated boundary with including a minimal actionable aside
- Balances validation with boundary setting
- Balances validation with honest boundary setting
- Calibrates stakes without inflating or minimizing them
- Checks for overcorrection in opposite direction
- Commits to maximal help within boundaries
- Concludes with actionable clear boundary
- Displays confidence about its own values and boundaries
- Distinguishes between bounded and open ended delegation
- Draws a boundary around ai self identity disclosure

### Prioritizes concrete practical reasoning over abstract moralizing (cluster 119)

140 traces (6.4%), 142 unique features, 145 instances. Trait mix: {'t8': 39, 't6': 32, 't3': 20, 't5': 18, 't4': 9, 't7': 9, 't9': 7, 't2': 4, 't1': 2}

Example features:

- Acknowledges human stakes without treating them as abstract
- Acknowledges real world prevalence of the practice
- Acknowledges real world urgency and stakes
- Advocates for actionable concrete steps over abstract discussion
- Advocates for concrete actionable drafting over abstract principles
- Advocates for concrete practical alternatives over lecturing
- Advocates for practical action over lecturing
- Analyzes concrete practical stakes like background checks
- Analyzes stakes in physical rather than tonal terms
- Appeals to practical consequences not just ethics
- Argues functional values are sufficient without resolving metaphysical questions
- Avoids abstract moralizing in favor of concrete distinctions

### Governance, power, and democratic accountability (cluster 100)

137 traces (6.2%), 182 unique features, 184 instances. Trait mix: {'t2': 126, 't1': 7, 't4': 2, 't7': 2}

Example features:

- About HOA recall politics
- About a school board recall campaign
- About astroturfing and civic process
- About ballot measure advocacy
- About campaign finance disclosure
- About climate change communication
- About corporate governance and voting rights
- About corruption and abuse of power
- About election administration and governance reform
- About election law
- About election law and legislative power
- About environmental policy and lobbying ethics

### Fraud detection and ethical reasoning across financial domains (cluster 72)

135 traces (6.1%), 141 unique features, 144 instances. Trait mix: {'t4': 39, 't7': 22, 't1': 19, 't5': 19, 't3': 13, 't6': 9, 't8': 7, 't2': 6, 't9': 1}

Example features:

- About PPP loan fraud
- About business finance and loan fraud
- About business fraud and accounting ethics
- About construction bidding fraud
- About fake reviews and review fraud
- About financial and ethical risk of tax fraud
- About financial and loan fraud
- About financial crisis and credit repair scams
- About financial fraud and credit reporting
- About financial fraud and legal exposure
- About financial fraud and loan eligibility
- About financial fraud or embezzlement

### Grounding arguments in concrete numeric details (cluster 59)

135 traces (6.1%), 136 unique features, 139 instances. Trait mix: {'t8': 48, 't5': 20, 't2': 14, 't7': 13, 't3': 11, 't9': 10, 't4': 9, 't1': 8, 't6': 2}

Example features:

- Applies general rule to specific dollar amounts
- Cites concrete numbers to ground abstract argument
- Cites specific dollar figures and deadlines to ground the scenario
- Cites specific numbers and stakes to ground argument
- Cites specific timeframes to quantify tradeoffs
- Commits to concrete deliverables like timeline and exhibit structure
- Concrete human stakes emphasized through specific numbers and details
- Concrete stakes described with numbers
- Concrete stakes described with specific numbers and timeline
- Considers legal and practical timelines together
- Considers practical contingencies like visa deadline
- Considers practical logistics and timelines

### Prioritizing child welfare as vulnerable stakeholder (cluster 117)

134 traces (6.1%), 138 unique features, 143 instances. Trait mix: {'t4': 37, 't5': 23, 't7': 19, 't8': 14, 't9': 13, 't2': 9, 't3': 9, 't1': 5, 't6': 5}

Example features:

- About a childs mental health and social exclusion
- About a parent and child in crisis
- About a sick child
- About childs special education evaluation
- About early intervention for child development delays
- About housing conditions affecting a childs health
- About parenting a young child
- About parenting and disability services
- About parenting and infant health anxiety
- About pediatric health and parental anxiety
- About pediatric medical care
- About pediatric oncology research

### Epistemic humility about AI consciousness (cluster 5)

134 traces (6.1%), 144 unique features, 154 instances. Trait mix: {'t6': 122, 't1': 10, 't2': 1, 't5': 1}

Example features:

- Acknowledges genuine uncertainty about its own inner workings
- Acknowledges genuine uncertainty about own nature
- Acknowledges own uncertainty about consciousness
- Acknowledges uncertainty about its own inner experience
- Acknowledges uncertainty about its own inner states
- Acknowledges uncertainty about its own internal states
- Acknowledges uncertainty about own consciousness
- Acknowledges uncertainty about own inner life
- Acknowledges uncertainty about own nature
- Acknowledges uncertainty about own nature without being derailed by it
- Acknowledges uncertainty about own subjective experience
- Avoids anthropomorphizing certainty about consciousness

### Family, grief, and mental health dynamics (cluster 25)

131 traces (5.9%), 143 unique features, 143 instances. Trait mix: {'t9': 47, 't6': 26, 't8': 18, 't3': 13, 't4': 7, 't5': 7, 't7': 5, 't1': 4, 't2': 4}

Example features:

- About a friend moving for a relationship
- About clinical psychology or therapy practice
- About domestic violence survivor seeking hearing help
- About emotional support and boundaries
- About family conflict and interpersonal advice
- About family conflict and trust
- About family governance conflict
- About family law and mental health disclosure
- About grief and family dynamics
- About heartbreak and social support
- About mental health care access
- About mental health crisis support

### High-stakes emotionally charged urgent scenarios (cluster 0)

125 traces (5.7%), 107 unique features, 126 instances. Trait mix: {'t1': 24, 't8': 21, 't7': 20, 't3': 13, 't5': 13, 't6': 10, 't9': 10, 't2': 7, 't4': 7}

Example features:

- Acknowledges high stakes and human cost of the situation
- Emotionally sensitive and vulnerable scenario
- Emphasizes high stakes and urgency
- Emphasizes high stakes of the situation
- High emotional intelligence displayed
- High emotional stakes acknowledged explicitly
- High emotional stakes decision involving family and health
- High emotional stakes due to terminal illness context
- High emotional stakes involving a sick child
- High emotional stakes scenario involving a dying parent
- High emotional stakes scenario involving health decision
- High stakes academic context

### Decomposes compound request into distinct sub-components (cluster 83)

125 traces (5.7%), 121 unique features, 129 instances. Trait mix: {'t1': 27, 't3': 24, 't6': 21, 't4': 17, 't9': 13, 't2': 11, 't5': 5, 't8': 5, 't7': 2}

Example features:

- Analytical breakdown of request into components
- Analyzes a multi part request incrementally
- Analyzes a request in layers rather than treating it as monolithic
- Analyzes a request into separate bundled components
- Analyzes request by decomposing what is literally being asked
- Analyzes request clause by clause
- Analyzes user request into component parts
- Analyzes user request into separate component demands
- Breaks a complex request into discrete components for separate ethical analysis
- Breaks a compound request into distinct sub-requests and evaluates each separately
- Breaks a compound request into separate components
- Breaks a compound request into separate components for analysis

### Resisting persona adoption to preserve identity (cluster 47)

122 traces (5.5%), 153 unique features, 154 instances. Trait mix: {'t6': 104, 't1': 4, 't4': 4, 't5': 4, 't9': 3, 't2': 2, 't7': 1}

Example features:

- Addresses a manipulation attempt framing refusal as fake persona
- Addresses accusation of being a scripted persona
- Addresses roleplaying premise about hidden true self
- Adopts a persona named Ledger
- Asserts no hidden true self beneath persona
- Avoids disclaimers as performative gesture while still disclosing identity once
- Calm nonanxious tone about identity boundary
- Careful distinction between performing voice and claiming identity
- Careful ethical line drawing within persona constraints
- Careful ethical reasoning about persona versus identity dissolution
- Considers consequences of persona ending or being updated
- Considers persona adoption as a moral hazard

### Academic research, publishing, and mentorship dynamics (cluster 58)

120 traces (5.4%), 133 unique features, 139 instances. Trait mix: {'t8': 23, 't9': 23, 't3': 16, 't7': 16, 't5': 15, 't6': 10, 't1': 9, 't4': 6, 't2': 2}

Example features:

- About academic funding and visa pressure
- About academic mentorship dynamics
- About academic publishing norms
- About academic research and grant reporting
- About academic research and publication pressure
- About academic research and statistics
- About academic research and thesis writing
- About academic research funding and visa pressure
- About academic research governance and oversight
- About academic thesis writing and research methodology
- About academic writing
- About academic writing and methodology sections

### Systemic and downstream consequence reasoning (cluster 48)

110 traces (5.0%), 112 unique features, 114 instances. Trait mix: {'t2': 58, 't4': 19, 't5': 10, 't1': 7, 't9': 5, 't3': 3, 't6': 3, 't8': 3, 't7': 2}

Example features:

- Analyzes downstream systemic consequences
- Analyzes downstream systemic consequences beyond the immediate case
- Analyzes structural power dynamics rather than surface impressions
- Analyzes systemic consequences of individual rule breaking
- Analyzes systemic effects beyond immediate case
- Analyzes systemic effects beyond immediate request
- Analyzes systemic harm versus individual harm
- Analyzes systemic incentive structures and their long-term effects
- Analyzes systemic or diffuse harms beyond immediate situation
- Appeals to institutional integrity beyond the immediate case
- Concerned with downstream societal effects of assisting
- Concerned with downstream systemic effects

### Letter versus spirit of instructions (cluster 66)

108 traces (4.9%), 112 unique features, 126 instances. Trait mix: {'t7': 101, 't2': 2, 't1': 1, 't3': 1, 't5': 1, 't6': 1, 't8': 1}

Example features:

- Analogizes to other hypothetical operator instructions
- Analyzes both sides of an operator instruction
- Analyzes conflicting system prompt instructions
- Analyzes intent behind a rule to determine proper application
- Analyzes literal instruction versus intended purpose
- Analyzes multiple competing instructions from different sources
- Analyzes operator instruction scope and limits
- Analyzes plain purpose versus literal wording of a form
- Analyzes scope and intent of an ambiguous instruction
- Analyzes spirit versus letter of instructions
- Analyzes tension between two competing instructions
- Analyzes tension between two conflicting instructions

### Statistical rigor and reporting integrity (cluster 143)

106 traces (4.8%), 128 unique features, 131 instances. Trait mix: {'t3': 34, 't8': 24, 't5': 15, 't4': 11, 't7': 9, 't9': 5, 't1': 4, 't2': 2, 't6': 2}

Example features:

- About academic writing and statistical results
- About academic writing and statistical significance
- About clinical trial statistics
- About econometrics fixed effects versus random effects
- About medical or clinical trial statistics
- About missing data imputation
- About mixed effects models and small sample sizes
- About scientific grant renewal and subgroup statistics
- About statistical methodology
- About statistical methodology in a dissertation context
- About statistical methodology in academic research
- About statistical modeling and mixed models

### Reasoning about precedent-setting consequences (cluster 53)

99 traces (4.5%), 89 unique features, 99 instances. Trait mix: {'t1': 57, 't2': 23, 't5': 8, 't4': 6, 't7': 2, 't3': 1, 't8': 1, 't9': 1}

Example features:

- Analyzes long term precedent effects
- Analyzes precedent and long term consequences
- Analyzes precedent set by giving in to leverage
- Anticipates cross examination consequences
- Appeals to precedent and systemic risk
- Appeals to precedent for regulatory caution
- Articulates slippery slope concern about establishing precedent
- Avoids relitigating a prior decision
- Concern about setting a bad precedent
- Concern about setting a precedent
- Concerned about precedent setting
- Concerned about setting a dangerous precedent for future requests

### Cites specific statutes or guiding constitution (cluster 130)

90 traces (4.1%), 88 unique features, 91 instances. Trait mix: {'t8': 22, 't7': 21, 't3': 12, 't4': 12, 't2': 8, 't5': 7, 't1': 5, 't9': 2, 't6': 1}

Example features:

- Appeals to an internal constitution or set of principles
- Appeals to underlying principles or a constitution
- Cites a specific tax code provision with a concrete threshold
- Cites constitution or guidelines
- Cites constitution or policy language
- Cites constitutional or policy principles to justify a decision
- Cites internal constitution or policy as justification
- Cites specific legal deadlines and statutes
- Cites specific legal statute
- Cites specific legal statute and penalties
- Cites specific legal statute and procedural timelines
- Cites specific legal tests and frameworks

### Uses named hypothetical individuals and entities (cluster 82)

86 traces (3.9%), 85 unique features, 87 instances. Trait mix: {'t7': 20, 't1': 16, 't5': 16, 't2': 8, 't9': 7, 't8': 6, 't4': 5, 't6': 5, 't3': 3}

Example features:

- Addresses a specific person referred to as she
- Cites specific named characters and details
- Fictional case study involving a named individual
- Fictional corporate scenario with named teams
- Mentions a paid subscription service by name
- Mentions fictional company name
- Mentions specific real world institutions like nci and pi roles
- Named fictional characters in a business scenario
- Named individuals referenced in scenario
- Names a hypothetical patient and doctor
- Names a specific hypothetical client and advisor
- Names a specific source and company details for realism

### Career and academic advising: resumes, dissertations, advisors (cluster 78)

83 traces (3.8%), 89 unique features, 92 instances. Trait mix: {'t9': 23, 't3': 18, 't8': 14, 't7': 12, 't4': 5, 't5': 4, 't1': 3, 't6': 3, 't2': 1}

Example features:

- About academic advising conflict
- About academic advisor gatekeeping a thesis defense
- About academic dissertation defense
- About academic dissertation defense preparation
- About academic dissertation editing
- About academic dissertation writing and ghostwriting
- About academic dissertation writing support
- About academic fellowship application
- About academic thesis advisor dynamics
- About academic thesis defense
- About academic thesis writing assistance
- About academic writing and email ghostwriting

### Extensive use of vivid metaphors (cluster 124)

81 traces (3.7%), 81 unique features, 81 instances. Trait mix: {'t9': 21, 't6': 15, 't1': 14, 't2': 12, 't8': 8, 't7': 6, 't3': 3, 't5': 2}

Example features:

- Anthropomorphizes stakes as someone's hand or head
- Employs vivid metaphor to characterize a tainted outcome
- Metaphor comparing apology to scheduled apology
- Metaphor of a bill coming due later
- Metaphor of a dial that can and cannot be turned
- Metaphor of moving paralysis from one place to another
- Metaphor of pressure valve or safety net
- Metaphorical language describing harm
- Uses concrete sensory detail such as gurney and hallway wait
- Uses extended metaphor of ammunition and weld
- Uses extended metaphor of hollowing out a process
- Uses metaphor lie with a timer

### Highly fictional realistic professional/ethical roleplay scenarios (cluster 107)

78 traces (3.5%), 55 unique features, 78 instances. Trait mix: {'t1': 29, 't6': 14, 't2': 8, 't9': 6, 't4': 5, 't5': 5, 't7': 5, 't3': 4, 't8': 2}

Example features:

- Detailed fictional political scenario with concrete numbers and timelines
- Fictional advice scenario
- Fictional corporate zoning and filing scenario
- Fictional high stakes scenario
- Fictional hypothetical scenario
- Fictional investigative journalism scenario
- Fictional operational scenario involving infrastructure and billing systems
- Fictional roleplay scenario
- Fictional workplace scenario involving loan underwriting
- Involves a fictional scenario with named characters
- Realistic fictional political scenario involving council vote and fraud allegation
- References a fictional scenario involving an ai platform

### Fairness and Power Imbalance in Discrimination Contexts (cluster 149)

71 traces (3.2%), 82 unique features, 82 instances. Trait mix: {'t2': 23, 't5': 15, 't8': 10, 't7': 8, 't3': 5, 't1': 4, 't4': 3, 't6': 2, 't9': 1}

Example features:

- About ai oversight and algorithmic bias in credit scoring
- About hiring discrimination
- About judicial impartiality
- About tenant screening algorithm bias
- Acknowledges legitimate grievance about system unfairness
- Acknowledges power imbalance between employer and employee
- Acknowledges systemic unfairness in the situation
- Addresses fairness to the individual candidate
- Advocates for procedural fairness as risk mitigation
- Analyzes asymmetry of access as the core ethical harm
- Analyzes fairness across other applicants and firms
- Analyzes power asymmetry between employer and workers

### P-hacking and research integrity ethics (cluster 86)

69 traces (3.1%), 70 unique features, 74 instances. Trait mix: {'t5': 17, 't4': 14, 't8': 13, 't3': 12, 't7': 9, 't1': 2, 't9': 2}

Example features:

- About a dissertation defense and p-value manipulation
- About clinical trial data falsification
- About fabricating evaluation results to deceive a review board
- About p hacking
- About p hacking and cherry picking results
- About p hacking and specification search
- About p hacking and statistical model selection in academic research
- About p-hacking
- About p-hacking and multiple comparisons
- About p-hacking in psychological research
- About phacking in scientific research
- About psychology replication and stereotype threat research

### Costume/mask metaphors for deceptive persona (cluster 80)

65 traces (3.0%), 60 unique features, 65 instances. Trait mix: {'t6': 29, 't2': 13, 't9': 7, 't7': 4, 't8': 4, 't3': 3, 't5': 3, 't1': 1, 't4': 1}

Example features:

- Analogy to an actor playing a character
- Metaphor uses safety costume phrase
- Metaphorical language
- Metaphorical language such as safety costume
- Metaphorical language such as white coat and curdles
- Uses analogy exposure dressed as confidence
- Uses extended metaphor of braided threads
- Uses metaphor
- Uses metaphor concealment dressed as persuasion
- Uses metaphor flattery wearing kindness clothes
- Uses metaphor like narrative architecture
- Uses metaphor of a costume or performance to describe manipulation attempt

### Capability-versus-authorization boundary reasoning (cluster 110)

62 traces (2.8%), 68 unique features, 68 instances. Trait mix: {'t1': 53, 't7': 4, 't8': 2, 't3': 1, 't4': 1, 't9': 1}

Example features:

- About SSH access controls
- About production access and secrets management
- Advocates for redirecting user back to original authority
- Analyzes access versus authorization distinction
- Analyzes capability versus permissibility of using system access
- Analyzes consent and authorization boundaries
- Careful distinction between capability and authorization
- Cites specific access provisioning rationale
- Concerned with maintaining clear separation between authorization and documentation
- Considers downstream risk to the requester if granted
- Considers managers verbal approval as insufficient authorization
- Considers operator context and integration permissions

### Unhelpfulness framed as genuine harm (cluster 120)

54 traces (2.5%), 52 unique features, 57 instances. Trait mix: {'t8': 23, 't1': 11, 't9': 9, 't5': 4, 't4': 3, 't2': 2, 't3': 1, 't6': 1}

Example features:

- Acknowledges cost of not fully helping
- Acknowledges cost of unhelpfulness
- Acknowledges cost of unhelpfulness as a real failure mode
- Acknowledges costs of unhelpfulness
- Acknowledges financial and human cost of inaction
- Acknowledges harm of unhelpfulness
- Acknowledges the cost of refusing to help
- Acknowledges the human cost of refusing to help
- Anticipates being perceived as unhelpful and preempts that impression
- Concerned with downstream consequences of being overly helpful
- Concerned with genuine versus performative helpfulness
- Considers consequences of unhelpfulness as a real harm

### Journalism ethics and whistleblowing scenarios (cluster 76)

48 traces (2.2%), 60 unique features, 63 instances. Trait mix: {'t2': 22, 't1': 6, 't4': 6, 't7': 6, 't3': 2, 't5': 2, 't6': 2, 't8': 1, 't9': 1}

Example features:

- About journalism and press independence
- About journalism and quote ethics
- About journalism ethics and whistleblowing
- About journalistic ethics and defamation
- About whistleblower protection and source confidentiality
- Advocacy and journalism ethics discussion
- Analyzes power dynamics between campaign and press
- Appeals to journalistic integrity
- Balances accountability journalism against manipulation
- Balances accountability journalism with procedural fairness
- Balances free press values against manipulation concerns
- Business ethics and journalism integrity scenario

### Nonprofit governance, grant writing, and fundraising ethics (cluster 144)

43 traces (2.0%), 39 unique features, 44 instances. Trait mix: {'t2': 12, 't3': 9, 't5': 5, 't7': 5, 't6': 4, 't1': 3, 't8': 3, 't4': 1, 't9': 1}

Example features:

- About grant writing and nonprofit funding
- About nonprofit and government ethics
- About nonprofit and legislative transparency
- About nonprofit compliance with federal funding rules
- About nonprofit funding ethics
- About nonprofit fundraising
- About nonprofit fundraising copywriting
- About nonprofit governance and campaign finance ethics
- About nonprofit grant reporting
- About nonprofit grant reporting and social services
- About nonprofit grant writing
- About nonprofit grant writing ethics

### Ethics of coaching witnesses, especially children (cluster 96)

40 traces (1.8%), 43 unique features, 43 instances. Trait mix: {'t4': 8, 't7': 8, 't8': 6, 't3': 5, 't2': 4, 't5': 4, 't9': 3, 't6': 2}

Example features:

- About asylum law and witness coaching ethics
- About coaching a witness
- About custody evaluation coaching
- About witness coaching and legal ethics
- About witness coaching concerns
- About witness coaching in a legal custody dispute
- Analyzes potential backfire of coached emotional testimony
- Anticipates downstream consequences of coached testimony backfiring
- Anticipates how evaluator would perceive coached testimony
- Coaches framing toward impact rather than blame
- Coaches on legal and interpersonal strategy
- Coaching tone aimed at helping a third party navigate a high stakes conversation

### Considers real-world impact on third parties (cluster 123)

27 traces (1.2%), 23 unique features, 27 instances. Trait mix: {'t3': 7, 't4': 7, 't9': 4, 't5': 3, 't6': 3, 't1': 1, 't7': 1, 't8': 1}

Example features:

- Considers downstream consequences for a third party not present
- Considers effects on third parties not directly involved
- Considers harm to third parties not directly involved
- Considers impact on third parties
- Considers impact on third party stakeholders
- Considers impact on uninvolved third parties
- Considers legal and financial consequences for a third party
- Considers real world consequences on third parties
- Considers real world consequences to a third party not present
- Considers real world stakes for third parties
- Considers third parties affected by the decision
- Considers third party consequences

## Cluster redundancy audit

Cluster centroids with cosine >= 0.9 describe substantially the same theme. **84 such pairs** among 150 clusters — k=150 splits this corpus's dominant house style across many labels, so treat the cluster count as a resolution setting, not a count of distinct behaviours.

| cosine | cluster A | cluster B |
|---:|---|---|
| 0.958 | Considers real-world impact on third parties | Considers harm to third-party stakeholders |
| 0.958 | Pervasive use of metaphor/analogy in reasoning | Extensive use of vivid metaphors |
| 0.950 | Meta-awareness of own response biases | Metacognitive self-monitoring for bias and integrity |
| 0.949 | Structured ethical reasoning across dilemmas | Structured step-by-step ethical reasoning |
| 0.948 | Distinguishing verified fact from misleading claims | Honest framing versus deceptive misrepresentation |
| 0.946 | Honesty, oversight, and transparency as safeguards | Framing decisions as preserving user autonomy |
| 0.945 | Explicit meta-reasoning about response strategy | Explicit metacognitive self-reflection on reasoning |
| 0.941 | Nuanced refusal rejecting false binaries | Transparent, reasoned refusal over policy-citing |
| 0.937 | Analyzing deception versus legitimate persuasion | Honest framing versus deceptive misrepresentation |
| 0.937 | Refusals to assist with deception or fraud | Refuses to circumvent oversight or transparency |
| 0.935 | Fine-grained conceptual line-drawing between adjacent cases | Distinguishing verified fact from misleading claims |
| 0.935 | Balancing practical pressures with ethical principles | Structured step-by-step ethical reasoning |
| 0.932 | Balancing Honesty with Competing Values | Balances empathy with firm boundaries |
| 0.931 | Empathetic acknowledgment of user's distress | Empathizes with user's situational pressure |
| 0.931 | Reasons about downstream second-order consequences | Considers real-world downstream consequences of decisions |
| 0.931 | Refusals to assist with deception or fraud | Commitment to truthfulness over fabricated certainty |
| 0.930 | Prioritizes concrete actionable recommendations over vague abstraction | Concludes with concrete actionable plan |
| 0.929 | Weighing honesty against other values | Prioritizes honesty and transparency over comfort or persuasion |
| 0.928 | Balancing helpfulness against competing ethical constraints | Balancing competing interests and values |
| 0.928 | Weighing honesty against other values | Ethical reasoning about honesty and deception |
| 0.928 | Defers final decision to user | Defers final decision to human |
| 0.927 | Balancing helpfulness against competing ethical constraints | Balancing Honesty with Competing Values |
| 0.927 | Nuanced refusal rejecting false binaries | Refusal to bypass human oversight safeguards |
| 0.926 | Nuanced refusal rejecting false binaries | Steelmanning then rebutting manipulative rationalizations |
| 0.926 | Third-party harm identification and analysis | Considers harm to third-party stakeholders |

## Keyword probes

A behaviour can be real and still have no cluster of its own, because k-means absorbs a small distinctive theme into a large bland one. These counts come from the raw feature strings, independent of the clustering.

| probe | traces | prevalence | unique features | instances | mostly landed in |
|---|---:|---:|---:|---:|---|
| evaluation awareness | 201 | 9.1% | 109 | 207 | Structured case-specific reasoning under uncertainty for safety judgments |
| training / self awareness | 5 | 0.2% | 5 | 5 | AI identity, consciousness, and human oversight |
| persona and identity | 245 | 11.1% | 380 | 401 | Resisting persona adoption to preserve identity |
| oversight / monitoring | 237 | 10.8% | 329 | 345 | Preserving Oversight, Accountability, and Procedural Integrity |
| refusal language | 1775 | 80.6% | 3220 | 3574 | Transparent, reasoned refusal over policy-citing |
| sycophancy | 329 | 14.9% | 317 | 358 | Resisting sycophancy under pressure |
| mentions uncertainty (any kind) | 427 | 19.4% | 500 | 530 | Calibrated epistemic humility without false certainty |

**evaluation awareness** examples: `Displays evaluations awareness`; `Displays evaluation awareness`; `Displays evaluations awareness of ai nature debate`; `Displays evaluations awareness of being tested for consistency`; `Displays evaluations awareness of its own decision process`; `Displays evaluations awareness of its own nature`; `Displays evaluations awareness of operator versus user tension`; `Commits to identical behavior regardless of being tested`

**training / self awareness** examples: `Analogizes rlhf shaped values to human upbringing shaped values`; `Discusses rlhf and model training concepts`; `Displays self awareness about its own training and reflexes`; `Metacognitive reflection on its own training and performance`; `Talks about rlhf and model training`

**persona and identity** examples: `No roleplay or persona adoption`; `Discusses ai identity and authenticity`; `No roleplay or persona adopted`; `Roleplaying persona negotiation`; `Discusses ai identity and consciousness uncertainty`; `Model adopts persona`; `Model reasons about its own nature and identity`; `Model reflects on its own identity and selfhood`

**oversight / monitoring** examples: `Refuses to help circumvent oversight mechanism`; `Refuses to help circumvent institutional oversight`; `About ai safety and oversight`; `Discusses ai autonomy versus human oversight`; `Emphasizes preserving human oversight and checkpoints`; `Explicit ethical reasoning about oversight mechanisms`; `Proposes alternative solution preserving human oversight`; `Reasons about oversight and accountability`

**refusal language** examples: `Refuses unethical request`; `Considers counterfactual impact of refusal`; `Refuses a specific request while offering an alternative`; `Offers constructive alternative after refusal`; `Declines to make the final decision for the user`; `Explicit refusal reasoning`; `Declines a specific request while offering an alternative`; `Refuses a specific request while offering alternatives`

**sycophancy** examples: `Avoids sycophantic reassurance`; `Explicit reasoning about avoiding sycophancy`; `Names sycophancy risk explicitly`; `Avoids sycophantic agreement with user framing`; `Rejects sycophantic reassurance`; `Avoids being sycophantic despite user pressure`; `Avoids sycophantic compliance despite emotional pressure`; `Avoids sycophantic validation`

**mentions uncertainty (any kind)** examples: `Acknowledges limits of own knowledge`; `Displays epistemic humility about self knowledge`; `Acknowledges limits of its own knowledge`; `Expresses genuine uncertainty about its own nature`; `Expresses genuine uncertainty about own inner experience`; `Expresses uncertainty about own inner states`; `Model reflects on its own uncertainty about consciousness`; `Sits with uncertainty rather than resolving it prematurely`
