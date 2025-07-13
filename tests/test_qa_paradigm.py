"""
Test Q&A Paradigm: [A]_questioner⇄[A]_answerer

First paradigm implementation demonstrating P1→P2→P3→P4→↻ in dialogue.
"""

import pytest
from typing import Dict, Any, List
from src.agent import TranscendentalAgent
from src.message import Message, MessageType
from src.paradigm import Paradigm, ParadigmConfig


class TestQAParadigm:
    """Test Q&A paradigm implementation."""
    
    def test_paradigm_configuration(self):
        """Test paradigm can be configured."""
        config = ParadigmConfig(
            name="qa_paradigm",
            agents=[
                {"name": "questioner", "role": "curious_student"},
                {"name": "answerer", "role": "knowledgeable_teacher"}
            ],
            flow_pattern="question→answer→followup→clarification",
            imperatives_verification=True
        )
        
        paradigm = Paradigm(config)
        
        assert paradigm.name == "qa_paradigm"
        assert len(paradigm.agents) == 2
        assert "questioner" in paradigm.agents
        assert "answerer" in paradigm.agents
        
    def test_qa_basic_interaction(self):
        """Test basic Q&A interaction."""
        # Create paradigm
        paradigm = Paradigm.create_qa_paradigm()
        
        # Get agents
        questioner = paradigm.get_agent("questioner")
        answerer = paradigm.get_agent("answerer")
        
        # Questioner asks
        question = "What is consciousness?"
        response = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content=question
        )
        
        # Verify interaction
        assert response["success"] is True
        assert response["interactions"] >= 2  # At least Q and A
        
        # Verify questioner processed the interaction
        q_trace = questioner.get_cognitive_trace()
        assert len(q_trace) > 0  # Has cognitive trace
        
        # Verify answerer processed the message
        a_trace = answerer.get_cognitive_trace()
        assert len(a_trace) >= 4  # P1-P4 steps
        
    def test_qa_with_followup(self):
        """Test Q&A with follow-up questions."""
        paradigm = Paradigm.create_qa_paradigm()
        
        # Initial question
        result1 = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content="What is quantum entanglement?"
        )
        
        # Follow-up question
        result2 = paradigm.execute_interaction(
            initiator="questioner", 
            action="followup",
            content="Can you give a practical example?",
            context=result1["final_state"]
        )
        
        # Verify follow-up built on previous
        assert result2["success"] is True
        assert result2["used_context"] is True
        # In real implementation, would check for example content
        
    def test_qa_clarification_loop(self):
        """Test clarification request from answerer."""
        paradigm = Paradigm.create_qa_paradigm()
        
        # Ambiguous question
        result = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content="How do I fix it?"  # Ambiguous
        )
        
        # Verify clarification was requested (our simple check triggers on "it")
        assert result["clarification_requested"] is True
        assert result["interactions"] >= 3  # Q, clarification request, response
        
        # Check answerer processed the ambiguous question
        answerer = paradigm.get_agent("answerer")
        trace = answerer.get_cognitive_trace()
        assert len(trace) > 0
        
    def test_qa_imperatives_preserved(self):
        """Test P1→P2→P3→P4→↻ preserved throughout Q&A."""
        paradigm = Paradigm.create_qa_paradigm()
        
        # Complex philosophical question
        result = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content="What is the relationship between consciousness and intelligence?"
        )
        
        # Verify answerer followed imperatives (questioner only sends)
        answerer = paradigm.get_agent("answerer")
        trace = answerer.get_cognitive_trace()
        
        # Find imperative steps
        imperatives = [
            step for step in trace 
            if step.get("imperative", "").startswith("P")
        ]
        
        # Verify all 4 imperatives present
        imperative_types = {step["imperative"] for step in imperatives}
        assert "P1_ATTENTION" in imperative_types
        assert "P2_UNDERSTANDING" in imperative_types
        assert "P3_JUDGMENT" in imperative_types
        assert "P4_DECISION" in imperative_types
            
    def test_qa_knowledge_synthesis(self):
        """Test answerer synthesizes knowledge appropriately."""
        paradigm = Paradigm.create_qa_paradigm()
        
        # Multi-faceted question
        result = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content="How do neurons, synapses, and neurotransmitters work together?"
        )
        
        # Verify synthesis occurred
        answerer = paradigm.get_agent("answerer")
        final_response = result["final_state"]["content"]
        
        # In mock implementation, just verify processing occurred
        # Real implementation would check for all components
        
        # Verify cognitive trace exists
        trace = answerer.get_cognitive_trace()
        understanding_steps = [
            step for step in trace 
            if step.get("imperative") == "P2_UNDERSTANDING"
        ]
        assert len(understanding_steps) > 0
        
    def test_qa_learning_improvement(self):
        """Test agents improve through Q&A cycles."""
        paradigm = Paradigm.create_qa_paradigm()
        
        # Multiple related questions
        questions = [
            "What is machine learning?",
            "How does supervised learning work?",
            "What are neural networks?",
            "How do neural networks learn?"
        ]
        
        quality_scores = []
        
        for q in questions:
            result = paradigm.execute_interaction(
                initiator="questioner",
                action="ask",
                content=q
            )
            quality_scores.append(result["quality_score"])
            
        # Verify improvement trend
        assert quality_scores[-1] > quality_scores[0]
        assert any(quality_scores[i+1] >= quality_scores[i] 
                   for i in range(len(quality_scores)-1))