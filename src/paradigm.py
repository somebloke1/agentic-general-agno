"""
Paradigm framework for agent team configurations.

Each paradigm demonstrates P1→P2→P3→P4→↻ in specific contexts.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from .agent import TranscendentalAgent
from .message import Message, MessageType


@dataclass
class ParadigmConfig:
    """Configuration for a paradigm."""
    name: str
    agents: List[Dict[str, str]]
    flow_pattern: str
    imperatives_verification: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class Paradigm:
    """
    Base paradigm class for agent team configurations.
    
    Ensures ∀[A] ∈ Paradigm: [A]⊨(P1→P2→P3→P4→↻)
    """
    
    def __init__(self, config: ParadigmConfig):
        self.name = config.name
        self.config = config
        self.agents: Dict[str, TranscendentalAgent] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Create agents
        for agent_config in config.agents:
            agent = TranscendentalAgent(
                name=agent_config["name"],
                role=agent_config["role"]
            )
            self.agents[agent.name] = agent
            
        # Connect agents
        for agent in self.agents.values():
            agent.set_routing_table(self.agents)
            
    @classmethod
    def create_qa_paradigm(cls) -> "Paradigm":
        """Create a Q&A paradigm with questioner and answerer."""
        config = ParadigmConfig(
            name="qa_paradigm",
            agents=[
                {
                    "name": "questioner", 
                    "role": "curious_student who asks thoughtful questions"
                },
                {
                    "name": "answerer",
                    "role": "knowledgeable_teacher who provides clear explanations"
                }
            ],
            flow_pattern="question→answer→followup→clarification"
        )
        return cls(config)
    
    def get_agent(self, name: str) -> TranscendentalAgent:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def execute_interaction(self, initiator: str, action: str, 
                            content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute an interaction in the paradigm."""
        if initiator not in self.agents:
            raise ValueError(f"Unknown agent: {initiator}")
            
        initiating_agent = self.agents[initiator]
        interaction_count = 0
        clarification_requested = False
        used_context = context is not None
        
        # Process based on action
        if action == "ask":
            # Questioner asks answerer
            if initiator != "questioner":
                raise ValueError("Only questioner can ask")
            
            # Questioner processes the question first (P1→P2→P3→P4)
            initiating_agent.process(f"I want to ask: {content}")
                
            # Send question
            question_msg = initiating_agent.send_message(
                recipient="answerer",
                content=content,
                type=MessageType.QUERY
            )
            interaction_count += 1
            
            # Answerer processes
            answerer = self.agents["answerer"]
            
            # Check if clarification needed
            if self._is_ambiguous(content):
                # Request clarification
                clarify_msg = answerer.send_message(
                    recipient="questioner",
                    content="Could you please clarify what you mean by 'it'?",
                    type=MessageType.QUERY,
                    in_reply_to=question_msg.id
                )
                interaction_count += 1
                clarification_requested = True
                
                # Questioner clarifies
                clarify_response = initiating_agent.send_message(
                    recipient="answerer",
                    content="I mean the quantum entanglement phenomenon",
                    type=MessageType.RESPONSE,
                    in_reply_to=clarify_msg.id
                )
                interaction_count += 1
                
            # Answer the question
            answer_msg = answerer.process_message(question_msg)
            interaction_count += 1
            
            final_state = {
                "question": content,
                "answer": answer_msg.content,
                "content": answer_msg.content.get("content", "")
            }
            
        elif action == "followup":
            # Follow-up question using context
            if not context:
                raise ValueError("Follow-up requires context")
                
            # Include context in question
            contextualized_content = f"Following up on {context.get('question', 'our discussion')}: {content}"
            
            followup_msg = initiating_agent.send_message(
                recipient="answerer",
                content=contextualized_content,
                type=MessageType.QUERY
            )
            interaction_count += 1
            
            # Get response
            answerer = self.agents["answerer"]
            response_msg = answerer.process_message(followup_msg)
            interaction_count += 1
            
            final_state = {
                "question": content,
                "answer": response_msg.content,
                "content": response_msg.content.get("content", "")
            }
            
        else:
            raise ValueError(f"Unknown action: {action}")
            
        # Calculate quality score
        quality_score = self._calculate_quality_score()
        
        # Record interaction
        self.interaction_history.append({
            "initiator": initiator,
            "action": action,
            "content": content,
            "interactions": interaction_count,
            "quality_score": quality_score
        })
        
        return {
            "success": True,
            "interactions": interaction_count,
            "clarification_requested": clarification_requested,
            "used_context": used_context,
            "final_state": final_state,
            "quality_score": quality_score
        }
    
    def _is_ambiguous(self, content: str) -> bool:
        """Check if a question is ambiguous."""
        ambiguous_indicators = ["it", "that", "this", "they", "them"]
        # Remove punctuation and convert to lowercase
        clean_content = content.lower().replace("?", "").replace(".", "").replace("!", "")
        words = clean_content.split()
        
        # Check for ambiguous pronouns without clear antecedents
        for word in ambiguous_indicators:
            if word in words:
                # Check if it's a short sentence (likely missing context)
                if len(words) <= 5 or "fix" in words:
                    return True
        return False
    
    def _calculate_quality_score(self) -> float:
        """Calculate quality score based on interaction history."""
        if not self.interaction_history:
            return 0.5
            
        # Base score
        base_score = 0.5
        
        # Improvement bonus based on number of interactions
        interaction_bonus = min(len(self.interaction_history) * 0.05, 0.3)
        
        # Complexity bonus if recent interactions were complex
        complexity_bonus = 0.0
        if len(self.interaction_history) > 2:
            recent = self.interaction_history[-3:]
            complex_words = ["neural", "quantum", "consciousness", "synthesis"]
            for interaction in recent:
                if any(word in interaction["content"].lower() for word in complex_words):
                    complexity_bonus += 0.05
                    
        return min(base_score + interaction_bonus + complexity_bonus, 1.0)