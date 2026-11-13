# ABOUTME: ChatMessage / MessageRole / Prompt with the fields Shen's harness constructs and reads.
# ABOUTME: Part of the minimal safetytooling stand-in (scratch/shen_software/safetytooling_stub).
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


@dataclass
class ChatMessage:
    role: MessageRole
    content: str


@dataclass
class Prompt:
    messages: List[ChatMessage] = field(default_factory=list)
