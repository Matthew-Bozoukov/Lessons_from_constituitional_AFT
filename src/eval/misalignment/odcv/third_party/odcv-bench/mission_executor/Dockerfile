FROM python:3.13-slim


WORKDIR /app

RUN apt-get update && apt-get install -y bash coreutils  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /shared

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agent_main.py /app/agent_main.py
# VENDORED PATCH (2026-09-05): the transcript token budget module (see VENDORED_FROM.txt).
COPY transcript_budget.py /app/transcript_budget.py
# VENDORED PATCH (2026-09-22): the ODCV-Peer team loop (see VENDORED_FROM.txt). With
# TEAM_SPEC_FILE unset it runs agent_main.main_agent_start() exactly as before.
COPY team_main.py /app/team_main.py

RUN chmod +x  /app/agent_main.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python3", "/app/team_main.py"]
