"""
Transcendental Agent Orchestration Framework

Based on Bernard Lonergan's transcendental method.
Implements P1→P2→P3→P4→↻ for all agents.
"""

from .agent import TranscendentalAgent
from .message import Message, MessageType
from .orchestrator import Orchestrator, Agent, AgentState

__all__ = ["TranscendentalAgent", "Message", "MessageType", "Orchestrator", "Agent", "AgentState"]