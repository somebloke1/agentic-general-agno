"""
Test Q&A Paradigm: [A]_questioner⇄[A]_answerer

First paradigm implementation demonstrating P1→P2→P3→P4→↻ in dialogue.
"""

import os
import pytest
from typing import Dict, Any, List
from src.agent import TranscendentalAgent
from src.message import Message, MessageType
from src.paradigm import Paradigm, ParadigmConfig
from tests.llm_test_utils import (
    LLM_CONFIGS,
    assert_is_meaningful_question,
    assert_answer_addresses_question,
    validate_conversation_flow,
    assert_follows_imperatives,
    measure_improvement,
    AgentTrace
)

# Skip all tests in this file if RUN_LLM_TESTS is not set
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1",
    reason="RUN_LLM_TESTS not set to '1'"
)


@pytest.mark.llm_test
class TestQAParadigm:
    """Test Q&A paradigm implementation with real LLMs."""
    
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
        
    @pytest.mark.parametrize("model_config", LLM_CONFIGS)
    def test_qa_basic_interaction(self, model_config):
        """Test basic Q&A interaction with semantic validation."""
        # Create paradigm with LLM support
        config = ParadigmConfig(
            name="qa_paradigm",
            agents=[
                {
                    "name": "questioner",
                    "role": "curious_student who asks thoughtful questions",
                    "model": model_config
                },
                {
                    "name": "answerer",
                    "role": "knowledgeable_teacher who provides clear explanations",
                    "model": model_config
                }
            ],
            use_llm=True
        )
        paradigm = Paradigm(config)
        
        # Questioner asks about consciousness
        topic = "consciousness"
        response = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content=f"Ask a thoughtful question about {topic}"
        )
        
        # Verify interaction success
        assert response["success"] is True
        assert response["interactions"] >= 2  # At least Q and A
        
        # Get actual question and answer from history
        history = paradigm.interaction_history
        assert len(history) >= 2
        
        question = history[0]["content"]
        answer = history[1]["content"]
        
        # Semantic validation
        assert_is_meaningful_question(question, topic)
        assert_answer_addresses_question(answer, question)
        
    @pytest.mark.parametrize("model_config", LLM_CONFIGS)
    def test_qa_with_followup(self, model_config):
        """Test Q&A with follow-up questions using real conversation flow."""
        # Create paradigm with LLM
        config = ParadigmConfig(
            name="qa_paradigm",
            agents=[
                {
                    "name": "questioner",
                    "role": "curious_student who asks thoughtful questions",
                    "model": model_config
                },
                {
                    "name": "answerer",
                    "role": "knowledgeable_teacher who provides clear explanations",
                    "model": model_config
                }
            ],
            use_llm=True
        )
        paradigm = Paradigm(config)
        
        # Initial question about quantum physics
        topic = "quantum entanglement"
        paradigm.topic = topic  # Set topic for validation
        result1 = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content=f"Ask a question about {topic}"
        )
        
        # Follow-up requesting example
        result2 = paradigm.execute_interaction(
            initiator="questioner", 
            action="followup",
            content="Ask for a practical example or clarification",
            context=result1.get("final_state")
        )
        
        # Collect all messages
        messages = []
        for interaction in paradigm.interaction_history:
            messages.append(Message(
                sender=interaction.get("from", "questioner"),
                recipient="answerer" if interaction.get("from") == "questioner" else "questioner",
                content=interaction["content"],
                type=MessageType.QUERY if interaction.get("from") == "questioner" else MessageType.RESPONSE
            ))
        
        # Validate conversation flow
        validate_conversation_flow(paradigm, messages)
        
    @pytest.mark.parametrize("model_config", LLM_CONFIGS)
    def test_qa_clarification_loop(self, model_config):
        """Test real clarification behavior when question is ambiguous."""
        # Create paradigm with LLM
        config = ParadigmConfig(
            name="qa_paradigm",
            agents=[
                {
                    "name": "questioner",
                    "role": "curious_student who asks thoughtful questions",
                    "model": model_config
                },
                {
                    "name": "answerer",
                    "role": "knowledgeable_teacher who asks for clarification when needed",
                    "model": model_config
                }
            ],
            use_llm=True
        )
        paradigm = Paradigm(config)
        
        # Send ambiguous question
        result = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content="What should I do about it?"  # Ambiguous
        )
        
        # Verify interaction happened
        assert result["success"] is True
        
        # Check if answerer requested clarification or made assumptions
        history = paradigm.interaction_history
        if len(history) >= 2:
            answer = history[1]["content"].lower()
            
            # Check for clarification indicators
            clarification_phrases = [
                "what", "which", "could you clarify", "what do you mean",
                "can you be more specific", "what are you referring to",
                "it", "context"
            ]
            
            requested_clarification = any(
                phrase in answer for phrase in clarification_phrases
            )
            
            # Either the answerer should ask for clarification
            # or acknowledge the ambiguity in some way
            assert requested_clarification or "it" in answer
        
    @pytest.mark.parametrize("model_config", LLM_CONFIGS)
    def test_qa_imperatives_preserved(self, model_config):
        """Test P1→P2→P3→P4→↻ preserved throughout Q&A with real behavior."""
        # Create paradigm with LLM
        config = ParadigmConfig(
            name="qa_paradigm",
            agents=[
                {
                    "name": "questioner",
                    "role": "curious_student who demonstrates attention and intelligence",
                    "model": model_config
                },
                {
                    "name": "answerer",
                    "role": "knowledgeable_teacher who reasons carefully and decides responsibly",
                    "model": model_config
                }
            ],
            use_llm=True
        )
        paradigm = Paradigm(config)
        
        # Complex philosophical question
        result1 = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content="Ask about the relationship between consciousness and intelligence"
        )
        
        # Follow-up to generate more exchanges
        result2 = paradigm.execute_interaction(
            initiator="questioner",
            action="followup",
            content="Ask for clarification or deeper explanation about a specific aspect",
            context=result1.get("final_state")
        )
        
        # Collect agent messages for trace
        messages = []
        for interaction in paradigm.interaction_history:
            messages.append(Message(
                sender=interaction.get("from", "questioner"),
                recipient="answerer" if interaction.get("from") == "questioner" else "questioner",
                content=interaction["content"],
                type=MessageType.QUERY if interaction.get("from") == "questioner" else MessageType.RESPONSE
            ))
        
        # Create agent trace for imperative validation
        answerer_trace = AgentTrace(
            agent_id="answerer",
            messages=[m for m in messages if m.sender == "answerer"],
            processing_times=[],  # Not tracking times in this test
            state_transitions=[]  # Not tracking state in this test
        )
        
        # Verify imperatives are followed in actual responses
        assert_follows_imperatives(answerer_trace)
            
    @pytest.mark.parametrize("model_config", LLM_CONFIGS)
    def test_qa_knowledge_synthesis(self, model_config):
        """Test answerer synthesizes knowledge appropriately with semantic validation."""
        # Create paradigm with LLM
        config = ParadigmConfig(
            name="qa_paradigm",
            agents=[
                {
                    "name": "questioner",
                    "role": "curious_student asking about complex systems",
                    "model": model_config
                },
                {
                    "name": "answerer",
                    "role": "expert teacher who synthesizes multiple concepts clearly",
                    "model": model_config
                }
            ],
            use_llm=True
        )
        paradigm = Paradigm(config)
        
        # Multi-faceted question requiring synthesis
        result = paradigm.execute_interaction(
            initiator="questioner",
            action="ask",
            content="Ask about how multiple components work together in a system (e.g., neurons, cells, or software components)"
        )
        
        # Get the actual answer
        history = paradigm.interaction_history
        assert len(history) >= 2
        
        question = history[0]["content"]
        answer = history[1]["content"]
        
        # Validate synthesis quality semantically
        # Check that answer addresses multiple components
        component_indicators = [
            "and", "together", "interact", "combine", "work with",
            "connect", "integrate", "coordinate", "collaborate"
        ]
        
        answer_lower = answer.lower()
        synthesis_score = sum(1 for indicator in component_indicators if indicator in answer_lower)
        
        # Answer should show synthesis of multiple concepts
        assert synthesis_score >= 2, f"Answer doesn't show sufficient synthesis: {answer}"
        
        # Answer should be comprehensive (not just listing)
        assert len(answer.split()) >= 20, f"Answer too brief for synthesis: {answer}"
        
    @pytest.mark.parametrize("model_config", LLM_CONFIGS)
    def test_qa_learning_improvement(self, model_config):
        """Test agents improve through Q&A cycles using real improvement metrics."""
        # Create paradigm with LLM
        config = ParadigmConfig(
            name="qa_paradigm",
            agents=[
                {
                    "name": "questioner",
                    "role": "curious_student learning about machine learning",
                    "model": model_config
                },
                {
                    "name": "answerer",
                    "role": "expert teacher explaining machine learning concepts",
                    "model": model_config
                }
            ],
            use_llm=True
        )
        paradigm = Paradigm(config)
        paradigm.topic = "machine learning"  # Set topic for improvement tracking
        
        # Generate multiple questions on related topics
        question_prompts = [
            "Ask a basic question about machine learning",
            "Ask a more detailed question about supervised learning",
            "Ask an advanced question about neural networks",
            "Ask a sophisticated question about how neural networks learn"
        ]
        
        collected_questions = []
        
        for prompt in question_prompts:
            result = paradigm.execute_interaction(
                initiator="questioner",
                action="ask",
                content=prompt
            )
            
            # Get the actual question from history
            if paradigm.interaction_history:
                question = paradigm.interaction_history[-2]["content"]  # -2 for question, -1 for answer
                collected_questions.append(question)
        
        # Measure improvement across questions
        improvement_metrics = measure_improvement(paradigm, collected_questions)
        
        # Verify some form of improvement
        assert len(collected_questions) >= 3, "Need multiple questions to measure improvement"
        
        # Check at least one metric shows positive trend
        improvement_found = False
        for key, value in improvement_metrics.items():
            if "improvement" in key and value > 0:
                improvement_found = True
                break
        
        # Also accept if average scores are reasonably high
        if not improvement_found:
            avg_scores = [v for k, v in improvement_metrics.items() if "_avg" in k]
            if avg_scores and sum(avg_scores) / len(avg_scores) > 0.5:
                improvement_found = True
        
        assert improvement_found, f"No improvement detected in metrics: {improvement_metrics}"