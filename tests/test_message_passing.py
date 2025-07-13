"""
Test Message Passing: [A]⇄[A] communication

Tests verify bidirectional message passing with P1→P2→P3→P4→↻ preserved.
"""

import pytest
from typing import Dict, Any, List
from src.agent import TranscendentalAgent
from src.message import Message, MessageType


class TestMessagePassing:
    """Test agent-to-agent message passing."""
    
    def test_message_structure(self):
        """Test message contains required fields."""
        msg = Message(
            sender="agent1",
            recipient="agent2",
            content="Hello",
            type=MessageType.QUERY
        )
        
        assert msg.sender == "agent1"
        assert msg.recipient == "agent2"
        assert msg.content == "Hello"
        assert msg.type == MessageType.QUERY
        assert msg.timestamp is not None
        assert msg.id is not None
        
    def test_agent_send_message(self):
        """Test agent can send messages."""
        sender = TranscendentalAgent(name="sender", role="questioner")
        recipient = TranscendentalAgent(name="recipient", role="answerer")
        
        # Send message
        msg = sender.send_message(
            recipient="recipient",
            content="What is 2+2?",
            type=MessageType.QUERY
        )
        
        # Verify message created
        assert msg is not None
        assert msg.sender == "sender"
        assert msg.recipient == "recipient"
        assert msg.content == "What is 2+2?"
        
        # Verify sender's outbox
        assert len(sender.get_outbox()) == 1
        assert sender.get_outbox()[0] == msg
        
    def test_agent_receive_message(self):
        """Test agent can receive and process messages."""
        agent = TranscendentalAgent(name="receiver", role="processor")
        
        # Create incoming message
        msg = Message(
            sender="external",
            recipient="receiver",
            content="Process this data: [1,2,3]",
            type=MessageType.REQUEST
        )
        
        # Receive message
        agent.receive_message(msg)
        
        # Verify inbox
        assert len(agent.get_inbox()) == 1
        assert agent.get_inbox()[0] == msg
        
        # Process message
        response = agent.process_message(msg)
        
        # Verify processing followed imperatives
        assert response is not None
        assert response.type == MessageType.RESPONSE
        assert response.in_reply_to == msg.id
        
    def test_bidirectional_communication(self):
        """Test [A]⇄[A] bidirectional communication."""
        agent1 = TranscendentalAgent(name="questioner", role="curious")
        agent2 = TranscendentalAgent(name="answerer", role="knowledgeable")
        
        # Connect agents
        agent1.connect_to(agent2)
        agent2.connect_to(agent1)
        
        # Agent1 asks question
        question = agent1.send_message(
            recipient="answerer",
            content="Why is the sky blue?",
            type=MessageType.QUERY
        )
        
        # Agent2 processes (already received via connection)
        answer = agent2.process_message(question)
        
        # Agent1 receives answer
        agent1.receive_message(answer)
        
        # Verify bidirectional flow
        assert len(agent1.get_outbox()) == 1
        assert len(agent1.get_inbox()) == 1
        assert len(agent2.get_inbox()) == 1
        assert len(agent2.get_outbox()) == 1
        
        # Verify reply chain
        assert answer.in_reply_to == question.id
        
    def test_message_routing(self):
        """Test messages route correctly in multi-agent system."""
        agents = {
            "coordinator": TranscendentalAgent(name="coordinator", role="orchestrator"),
            "worker1": TranscendentalAgent(name="worker1", role="processor"),
            "worker2": TranscendentalAgent(name="worker2", role="analyzer")
        }
        
        # Set up routing
        for name, agent in agents.items():
            agent.set_routing_table(agents)
        
        # Coordinator broadcasts task
        task_msg = agents["coordinator"].broadcast_message(
            content="Analyze dataset X",
            type=MessageType.TASK,
            recipients=["worker1", "worker2"]
        )
        
        # Verify broadcast
        assert len(task_msg) == 2
        assert all(msg.sender == "coordinator" for msg in task_msg)
        assert {msg.recipient for msg in task_msg} == {"worker1", "worker2"}
        
        # Workers already received tasks via routing table connections
        
        # Verify reception
        assert len(agents["worker1"].get_inbox()) == 1
        assert len(agents["worker2"].get_inbox()) == 1
        
    def test_message_history_preserved(self):
        """Test cognitive trace preserved across messages."""
        agent = TranscendentalAgent(name="historian", role="recorder")
        
        # Process first message
        msg1 = Message(
            sender="user",
            recipient="historian",
            content="Remember this: A",
            type=MessageType.REQUEST
        )
        agent.receive_message(msg1)
        response1 = agent.process_message(msg1)
        
        # Get trace after first message
        trace1 = agent.get_message_cognitive_trace(msg1.id)
        assert len(trace1) >= 4  # P1-P4
        
        # Process second message
        msg2 = Message(
            sender="user",
            recipient="historian",
            content="What did I ask you to remember?",
            type=MessageType.QUERY
        )
        agent.receive_message(msg2)
        response2 = agent.process_message(msg2)
        
        # Verify both traces preserved
        trace1_after = agent.get_message_cognitive_trace(msg1.id)
        trace2 = agent.get_message_cognitive_trace(msg2.id)
        
        assert trace1 == trace1_after  # First trace unchanged
        assert len(trace2) >= 4  # Second trace complete
        assert trace1 != trace2  # Different traces
        
    def test_message_type_handling(self):
        """Test different message types handled appropriately."""
        agent = TranscendentalAgent(name="handler", role="responder")
        
        message_types = [
            (MessageType.QUERY, "What is X?", "should_answer"),
            (MessageType.REQUEST, "Do X", "should_execute"),
            (MessageType.RESPONSE, "X is Y", "should_acknowledge"),
            (MessageType.TASK, "Complete X", "should_plan"),
            (MessageType.ERROR, "X failed", "should_handle_error")
        ]
        
        for msg_type, content, expected_action in message_types:
            msg = Message(
                sender="tester",
                recipient="handler",
                content=content,
                type=msg_type
            )
            
            agent.receive_message(msg)
            response = agent.process_message(msg)
            
            # Verify appropriate handling
            assert response is not None
            assert hasattr(response, 'metadata')
            assert response.metadata.get('action_taken', '').startswith(expected_action)