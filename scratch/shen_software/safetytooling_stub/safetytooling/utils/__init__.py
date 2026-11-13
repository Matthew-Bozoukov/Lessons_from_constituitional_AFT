# ABOUTME: utils.setup_environment() stand-in; the real one loads API keys, which this setup
# ABOUTME: does not need (every call goes to the local vLLM server).
from . import utils  # noqa: F401  (`from safetytooling.utils import utils`)
