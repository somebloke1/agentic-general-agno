"""
Message passing infrastructure for agent communication.

Supports [A]⇄[A] bidirectional communication patterns.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class MessageType(Enum):
    """Types of messages agents can exchange."""
    QUERY = "query"          # Question requiring answer
    REQUEST = "request"      # Action request
    RESPONSE = "response"    # Reply to query/request
    TASK = "task"           # Task assignment
    ERROR = "error"         # Error notification
    INFO = "info"           # Information sharing
    

@dataclass
class Message:
    """
    Message structure for agent communication.
    
    Embodies communication that preserves P1→P2→P3→P4→↻.
    """
    sender: str
    recipient: str
    content: Any
    type: MessageType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    in_reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "in_reply_to": self.in_reply_to,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            type=MessageType(data["type"]),
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            in_reply_to=data.get("in_reply_to"),
            metadata=data.get("metadata", {})
        )