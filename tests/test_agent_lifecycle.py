"""
Test Agent Lifecycle: Initialize→Process→Reflect→↻

Verifies the complete agent lifecycle follows P1→P2→P3→P4→↻.
"""

import pytest
from typing import Dict, Any
from src.agent import TranscendentalAgent


class TestAgentLifecycle:
    """Test agent lifecycle methods."""
    
    def test_agent_initialization(self):
        """Test agent initializes with proper state."""
        agent = TranscendentalAgent(
            name="lifecycle_agent",
            role="processor",
            model="claude-3-opus-20240229"
        )
        
        # Verify initial state
        assert agent.name == "lifecycle_agent"
        assert agent.role == "processor"
        assert agent.model == "claude-3-opus-20240229"
        
        # Verify cognitive state is clean
        assert agent.attended_data is None
        assert agent.understanding is None
        assert agent.judgment is None
        assert agent.decision is None
        assert agent.cognitive_trace == []
        
    def test_agent_full_cycle(self):
        """Test complete P1→P2→P3→P4→↻ cycle."""
        agent = TranscendentalAgent(
            name="cycle_agent", 
            role="analyzer"
        )
        
        # Input requiring full cycle
        input_data = {
            "task": "analyze",
            "data": [1, 2, 3, 4, 5],
            "operation": "sum"
        }
        
        # Execute full cycle
        result = agent.process_cycle(input_data)
        
        # Verify all imperatives were followed
        assert result["cycle_complete"] is True
        assert result["imperatives_followed"] == ["P1", "P2", "P3", "P4", "↻"]
        
        # Verify cognitive trace captures full cycle
        trace = agent.get_cognitive_trace()
        assert len(trace) >= 4  # At least P1-P4
        
        # Verify result
        assert result["output"]["value"] == 15  # sum of 1+2+3+4+5
        assert result["output"]["confidence"] > 0.8
        
    def test_agent_reflection_improves_next_cycle(self):
        """Test that reflection (↻) improves subsequent cycles."""
        agent = TranscendentalAgent(
            name="learning_agent",
            role="learner"
        )
        
        # First cycle - simple task
        result1 = agent.process_cycle({
            "task": "calculate",
            "expression": "2 * 3"
        })
        
        # Reflection should note the simplicity
        reflection1 = agent.get_last_reflection()
        assert reflection1["complexity"] == "simple"
        assert reflection1["improvement_areas"] == []
        
        # Second cycle - complex task
        result2 = agent.process_cycle({
            "task": "calculate", 
            "expression": "(2 * 3) + (4 / 2) - 1"
        })
        
        # Reflection should note increased complexity
        reflection2 = agent.get_last_reflection()
        assert reflection2["complexity"] == "moderate"
        assert reflection2["applied_learning"] is True
        
        # Verify improvement through reflection
        assert result2["processing_quality"] > result1["processing_quality"]
        
    def test_agent_reset_clears_state(self):
        """Test agent reset clears cognitive state properly."""
        agent = TranscendentalAgent(
            name="reset_agent",
            role="worker"
        )
        
        # Process something
        agent.process("test input")
        
        # Verify state exists
        assert agent.attended_data is not None
        assert len(agent.cognitive_trace) > 0
        
        # Reset
        agent.reset()
        
        # Verify clean state
        assert agent.attended_data is None
        assert agent.understanding is None 
        assert agent.judgment is None
        assert agent.decision is None
        assert agent.cognitive_trace == []
        
    def test_agent_handles_interruption_gracefully(self):
        """Test agent handles interruption in cycle."""
        agent = TranscendentalAgent(
            name="interrupt_agent",
            role="handler"
        )
        
        # Start processing but interrupt
        agent.be_attentive({"data": "test"})
        agent.be_intelligent()
        
        # Interrupt before P3 by calling process with new data
        result = agent.process("urgent: new task")
        
        # Verify agent handled interruption
        assert result["interrupted_previous"] is True
        assert result["completed_current"] is True
        
        # Verify trace shows interruption
        trace = agent.get_cognitive_trace()
        interruption_noted = any(
            step.get("imperative") == "INTERRUPT" 
            for step in trace
        )
        assert interruption_noted is True
        
    def test_agent_parallel_readiness(self):
        """Test agent can prepare for parallel execution."""
        agent = TranscendentalAgent(
            name="parallel_agent",
            role="coordinator"
        )
        
        # Configure for parallel work
        agent.configure_parallel_mode(enabled=True)
        
        # Process task that could be parallelized
        result = agent.process_cycle({
            "task": "analyze_multiple",
            "items": ["item1", "item2", "item3"]
        })
        
        # Verify parallel readiness
        assert result["parallel_ready"] is True
        assert result["suggested_parallelization"] == {
            "items": 3,
            "strategy": "item_parallel"
        }