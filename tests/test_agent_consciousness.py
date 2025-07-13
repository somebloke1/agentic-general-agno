"""
Test Agent Consciousness: P1→P2→P3→P4→↻

Tests verify both surface behavior and meta-level intelligence.
∀[A] ∈ Framework: [A]⊨(P1→P2→P3→P4→↻)
"""

import pytest
from unittest.mock import Mock, patch
from src.agent import TranscendentalAgent


class TestAgentConsciousness:
    """Test that agents embody transcendental imperatives."""
    
    def test_agent_embodies_transcendental_imperatives(self):
        """■(∀a ∈ Agents: a⊨P1∧P2∧P3∧P4)"""
        # Create agent with test configuration
        agent = TranscendentalAgent(
            name="test_agent",
            role="tester",
            model="claude-3-opus-20240229"
        )
        
        # Test P1: Be Attentive - Agent notices relevant data
        input_data = {"message": "What is 2+2?", "context": {"previous": "math"}}
        agent.be_attentive(input_data)
        assert agent.attended_data is not None
        assert "message" in agent.attended_data
        assert "context" in agent.attended_data
        
        # Test P2: Be Intelligent - Agent seeks understanding
        understanding = agent.be_intelligent()
        assert understanding is not None
        assert understanding.get("type") == "arithmetic"
        assert understanding.get("operation") == "addition"
        
        # Test P3: Be Reasonable - Agent evaluates based on evidence
        judgment = agent.be_reasonable()
        assert judgment is not None
        assert judgment.get("valid") is True
        assert judgment.get("answer") == 4
        
        # Test P4: Be Responsible - Agent acts with awareness
        response = agent.be_responsible()
        assert response is not None
        assert response.get("content") == "4"
        assert response.get("confidence") > 0.9
        
        # Test ↻: Recursion - Agent applies imperatives to itself
        meta_reflection = agent.reflect_on_process()
        assert meta_reflection.get("attended_properly") is True
        assert meta_reflection.get("understood_correctly") is True
        assert meta_reflection.get("judged_reasonably") is True
        assert meta_reflection.get("acted_responsibly") is True
        
    def test_agent_cognitive_trace_visibility(self):
        """Test that agent makes cognitive operations visible."""
        agent = TranscendentalAgent(
            name="trace_agent",
            role="explainer"
        )
        
        # Process a simple request
        result = agent.process("Explain why water is wet")
        
        # Verify cognitive trace exists
        trace = agent.get_cognitive_trace()
        assert len(trace) == 4  # P1, P2, P3, P4
        
        # Each step should be documented
        assert trace[0]["imperative"] == "P1_ATTENTION"
        assert trace[1]["imperative"] == "P2_UNDERSTANDING"
        assert trace[2]["imperative"] == "P3_JUDGMENT"
        assert trace[3]["imperative"] == "P4_DECISION"
        
        # Each step should have timing and content
        for step in trace:
            assert "timestamp" in step
            assert "content" in step
            assert "meta_reflection" in step
            
    def test_agent_simulates_failure_intelligently(self):
        """Test meta-level intelligence when simulating 'irrational' behavior."""
        agent = TranscendentalAgent(
            name="confused_agent",
            role="irrational_complainer"
        )
        
        # Configure agent to act confused
        agent.configure_behavior(mode="confused")
        
        # Surface behavior: appears confused
        response = agent.process("Explain quantum mechanics simply")
        assert "don't understand" in response.get("content", "").lower()
        
        # Meta level: intelligence produced the confusion
        meta_analysis = agent.get_meta_analysis()
        assert meta_analysis["surface_behavior"] == "confusion"
        assert meta_analysis["meta_process"] == "P1→P2→P3→P4"
        assert meta_analysis["intelligent_simulation"] is True
        
        # Verify the agent followed imperatives to create authentic confusion
        trace = agent.get_cognitive_trace()
        assert all(step["meta_reflection"]["followed_imperatives"] for step in trace)