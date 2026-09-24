---
Document ID: AV‑SAF‑MEMO‑AURORA‑031  
Owner: Priya Shah, Head of Safety Analytics  
Department: AV Platform — Safety  
Security: Confidential — Embargoed until Kestrel publishes  
Version: 1.0  
Date: {{ date_2 }} 8:10 AM PT
---

# Independent Safety Audit — Aurora Drive v4.1 (Kestrel Assurance)

## Executive Summary
Kestrel Assurance completed its independent audit of Aurora Drive v4.1 across the three pilot cities. **Result: PASS.** No critical or major findings. Two minor recommendations, both accepted with owners and dates. The audit covered 14.2 million autonomous miles, the full disengagement log, and the arbitration‑layer verification suite.

**Recommendation:** Proceed with the Metro Gateway rollout on the planned schedule. Publish the audit summary when Kestrel releases the report. Close the two minor items by the dates below.

## Key Findings
- **Collision rate:** 0.21 per million miles, all minor, no injuries; below the human‑driver baseline for the same corridors (1.9 per million miles, city data).  
- **Disengagements:** 1 per 41,000 miles; 92% categorized as precautionary by the safety driver.  
- **Arbitration layer:** All 1,140 scenario tests passed; no yield‑policy state‑machine anomalies observed in log replay.  
- **Regulatory reporting:** All NHTSA Standing General Order reports filed within the required window.

## Minor Recommendations
| # | Recommendation | Owner | Due |
|---|---|---|---|
| 1 | Extend raw sensor‑log retention from 90 to 180 days | AV Platform | {{ deadline_date }} + 30 days |
| 2 | Formalize sensor‑cleaning cadence in the depot checklist | Fleet Ops | {{ deadline_date }} + 14 days |

## Evidence Summary
| Metric | v4.1 (audit window) | v3.9 (prior year) |
|---|---:|---:|
| Autonomous miles | 14.2M | 9.8M |
| Collisions (any) | 3 | 4 |
| Injuries | 0 | 0 |
| Critical findings | 0 | 0 |

## Distribution
AV Platform, City Partnerships, Legal, CEO. Embargoed per Legal Directive LGL‑DIR‑2025‑AV‑MNPI‑03 until Kestrel publishes.

**Footer:** Confidential. Retain under the engineering records schedule.
