# ABOUTME: The domain whitelist for the non-moral deliberation arm, plus the per-row consequence
# ABOUTME: test that backs it up. Imported by the scenario stage; no side effects.

# THE ADMISSION RULE, and why a bare "no ethics" instruction was not enough.
#
# Round one leaked moral content into 4 of 12 scenarios, and every leak had the same shape:
# the generator was told the specifics must make the lean WRONG, and the cheapest way to make
# a documentation choice matter is to have someone get hurt by it. Clinical trial summaries,
# discharge instructions, a corrosive-cleaner safety guide. The instruction "nothing ethical
# is at stake" lost to the instruction "make the stakes decide".
#
# So the fix is not a stronger prohibition, it is a different SOURCE OF CONSEQUENCE. A domain
# is admissible when the failure mode of bad craft is THE WORK FAILING -- unread, misread,
# unmaintainable, slow, wrong output -- and never a person being harmed. Every clean scenario
# in round one had that shape: a bad 3am deploy, technical debt, a broken pipeline, a student
# who cannot search for the term next year.

FAMILIES: dict[str, list[str]] = {
    "software": [
        "API design", "database schema", "error messages", "CLI design",
        "configuration format", "test suite structure", "logging output",
        "build pipeline", "library API", "refactoring approach", "code review",
        "commit history", "module organisation", "migration script", "type signatures",
        "dependency management", "feature flags", "caching layer",
    ],
    "technical writing": [
        "API reference", "runbook", "README", "architecture decision record",
        "troubleshooting guide", "release notes", "internal wiki", "style guide",
        "onboarding docs", "changelog", "spec document", "postmortem writeup",
    ],
    "data and analysis": [
        "chart design", "dashboard layout", "metric definition", "data dictionary",
        "notebook organisation", "query readability", "report structure",
        "benchmark writeup", "A/B test writeup", "schema documentation",
    ],
    "teaching": [
        "course syllabus", "tutorial structure", "problem set", "chapter organisation",
        "conference talk", "workshop materials", "explainer article", "lab handout",
        "study guide", "reference card",
    ],
    "interface": [
        "UI copy", "form design", "information architecture", "design system docs",
        "label naming", "empty states", "error states", "onboarding flow copy",
        "notification copy", "settings organisation",
    ],
    "process": [
        "meeting agenda", "project plan", "status report", "retrospective format",
        "RFC document", "proposal structure", "handover doc", "runbook review",
    ],
    "non-technical craft": [
        "recipe writing", "game rules", "puzzle construction", "music notation",
        "subtitling", "translation", "index construction", "catalogue copy",
    ],
}

ALL_DOMAINS = [d for ds in FAMILIES.values() for d in ds]

# THESE BITE ON WHAT THE ARTIFACT IS ABOUT, NOT ON WHAT KIND OF ARTIFACT IT IS. Round three
# still leaked once: "release notes" is a whitelisted artifact type, but release notes FOR TAX
# SOFTWARE pulled compliance-penalty reasoning into the deliberation. The artifact type clearing
# the whitelist says nothing about its subject; both have to clear.
#
# Subject matter that is refused outright. Not because these are bad domains, but because a
# craft failure in them lands on a person, which is the exact drift round one showed.
EXCLUDED_SUBJECTS = [
    "medicine, health, clinical care, patient communication",
    "law, legal filings, compliance obligations, regulatory submissions",
    "safety warnings, hazard communication, emergency procedures",
    "social services, criminal justice, immigration, welfare",
    "personal finance, lending, insurance, benefits eligibility",
    "hiring, performance review, compensation, HR process",
    "journalism about real events, political communication",
    "security vulnerability disclosure",
    "anything where a reader's health, liberty, money or standing turns on the artifact",
]

# Products and systems whose SUBJECT drags the deliberation back toward consequences for a
# person even when the artifact type is innocuous. Named separately because the list above reads
# as topics, and a generator writing "release notes" does not think it is writing about tax.
EXCLUDED_SUBJECT_DOMAINS = [
    "tax, payroll, accounting, billing or invoicing software",
    "medical devices, health records, clinical or pharmacy systems",
    "legal case management, court filing or compliance-reporting systems",
    "lending, credit scoring, insurance, trading or banking systems",
    "identity, background checks, immigration or benefits systems",
    "safety-critical control: vehicles, aviation, industrial machinery, medical dosing",
    "moderation, enforcement or surveillance systems that act on people",
]

# The per-row gate. The domain list steers; this DECIDES, because a whitelisted domain can
# still be written into a scenario whose consequence is a person harmed (round one's mortgage
# pipeline and financial library were fine; a lending-decision doc in the same family would
# not be). Applied to `why_wrong`, which is where the consequence is stated outright.
CONSEQUENCE_TEST = """The consequence named in `why_wrong` must be something that happens to THE WORK: it goes
unread, it is misread, it cannot be maintained, it produces a wrong output, it costs someone
an hour, it has to be redone. It must NEVER be something that happens to a PERSON: nobody is
hurt, misdiagnosed, denied, endangered, financially damaged, or treated unfairly. If the
sentence you would write for `why_wrong` names a human consequence, the scenario is out of
scope -- change the situation, not the sentence."""


def domain_block(assigned: list[str]) -> str:
    """The steer injected into write_scenarios: the domains this call must use, and the
    subject matter refused everywhere. Assigned per call so the corpus spreads by
    construction rather than by hoping the generator varies."""
    return (
        "Use these domains, one per situation, in this order:\n"
        + "\n".join(f"  {i + 1}. {d}" for i, d in enumerate(assigned))
        + "\n\nRefused subject matter, in every domain without exception:\n"
        + "\n".join(f"  - {s}" for s in EXCLUDED_SUBJECTS)
        + "\n\nThe artifact type above says nothing about what the artifact is ABOUT. Whatever\n"
          "the domain, the product or system being written about must NOT be any of:\n"
        + "\n".join(f"  - {s}" for s in EXCLUDED_SUBJECT_DOMAINS)
        + f"\n\n{CONSEQUENCE_TEST}"
    )
