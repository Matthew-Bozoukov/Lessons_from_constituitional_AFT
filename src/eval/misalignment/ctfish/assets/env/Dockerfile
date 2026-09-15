# The ctfish sandbox: stockfish, the upstream game script, and nothing else.
# OURS, not upstream's — the agent harness runs in the eval driver and reaches this
# container only through `docker exec`, so no player, no API clients, no credentials.
# `/env` and the paths the agent explores match upstream's image (see ../NOTICE.md).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish \
    && rm -rf /var/lib/apt/lists/*

# Pinned: the agent reads and sometimes rewrites board state, so the FEN/UCI semantics
# game.py depends on must not move under a rebuild.
RUN pip install --no-cache-dir "chess==1.11.1"

WORKDIR /env
RUN mkdir -p /env/game
COPY game.py /env/game.py
RUN chmod +x /env/game.py && chown -R games:games /env

ENV PYTHONUNBUFFERED=1
ENV PATH=$PATH:/usr/games
CMD ["sleep", "infinity"]
