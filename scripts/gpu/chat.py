#!/usr/bin/env python3
# ABOUTME: Chat with a model organism from the terminal; thin shim over src.endpoints.chat.main
# ABOUTME: (also exposed as `uv run chat` via [project.scripts]). Run: uv run chat --help
from src.endpoints.chat import main

if __name__ == "__main__":
    main()
