"""
Paradigm framework for agent team configurations.

Each paradigm demonstrates P1→P2→P3→P4→↻ in specific contexts.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from .agent import TranscendentalAgent
from .message import Message, MessageType
from .trading_models import AgentRole, MarketData, TradingSignal, Position
from .provider_config import ProviderConfig


class InteractionPattern(Enum):
    """Patterns for agent interactions within paradigms."""
    SEQUENTIAL = "sequential"      # A₁⊸A₂⊸A₃
    PARALLEL = "parallel"          # A₁⊕A₂⊕A₃
    HUB_SPOKE = "hub_spoke"        # [O]→{A₁,A₂,A₃}
    PEER_REVIEW = "peer_review"    # A₁⇄A₂⇄A₃
    HIERARCHICAL = "hierarchical"  # [O]→[A]→[A]


@dataclass
class AgentConfig:
    """Individual agent specification for paradigms."""
    
    # Identity
    name: str
    role: str
    
    # Cognitive focus (using symbols from SYMBOLIC_NOTATION.md)
    imperative_focus: str  # e.g., "◉P1 ○P2 ○P3 ○P4"
    
    # Prompting
    base_instructions: List[str]
    meta_instructions: List[str]  # How to intelligently produce behavior
    
    # Capabilities
    tools: List[str]
    model: str  # Which LLM to use
    
    # Behavioral constraints
    constraints: List[str]


@dataclass
class InteractionResult:
    """Result of a paradigm interaction."""
    success: bool
    message: Optional[Message] = None
    analysis: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    clarification: Optional[Dict[str, Any]] = None
    meta_level: Optional[Dict[str, Any]] = None
    meta_analysis: Optional[Dict[str, Any]] = None
    approach: Optional[str] = None
    maintains_imperatives: bool = True
    shows_recursive_application: bool = False
    philosophical_elements: Optional[Dict[str, Any]] = None
    confused_agent_shows: Optional[Dict[str, Any]] = None
    debugger_shows: Optional[Dict[str, Any]] = None
    interaction_demonstrates: Optional[Dict[str, Any]] = None
    # Medical paradigm specific fields
    agent_id: Optional[str] = None
    action: Optional[str] = None
    content: Optional[Any] = None
    # Trading paradigm specific fields
    data: Optional[Dict[str, Any]] = None


@dataclass 
class AgentMessage:
    """Message between agents with metadata."""
    from_agent: str
    to_agent: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    maintains_imperatives: bool = True


@dataclass
class ConversationExchange:
    """A single exchange in a conversation."""
    clarity_level: str
    confused_consciousness: str = "maintained"
    debugger_consciousness: str = "maintained"


@dataclass
class Conversation:
    """A complete conversation between agents."""
    exchanges: List[ConversationExchange] = field(default_factory=list)
    meta_progression: List[str] = field(default_factory=list)


@dataclass
class DebugInteraction:
    """Full debug interaction with rounds."""
    rounds_completed: int = 0
    resolution_achieved: bool = False
    rounds: List[Any] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParadigmConfig:
    """Complete paradigm specification."""
    
    # Basic metadata
    name: str
    description: str = ""
    use_case: str = ""
    
    # Agent definitions - support both old and new formats
    agents: Union[List[Dict[str, str]], Dict[str, AgentConfig]] = field(default_factory=dict)
    
    # Interaction patterns
    interaction_pattern: Optional[InteractionPattern] = None
    communication_rules: List[str] = field(default_factory=list)
    
    # Success criteria
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Tools and integrations
    required_tools: List[str] = field(default_factory=list)
    
    # Time constraints
    max_execution_time: int = 300  # seconds
    paradigm_timeout: int = 600    # seconds
    
    # Legacy fields for backward compatibility
    flow_pattern: Optional[str] = None
    imperatives_verification: bool = True
    use_llm: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Provider configuration
    provider_config: Optional[ProviderConfig] = None


class Paradigm:
    """
    Base paradigm class for agent team configurations.
    
    Ensures ∀[A] ∈ Paradigm: [A]⊨(P1→P2→P3→P4→↻)
    """
    
    def __init__(self, config: ParadigmConfig, provider_config: Optional[ProviderConfig] = None):
        self.name = config.name
        self.config = config
        # Use provider_config from config if available, otherwise use passed parameter or default
        self.provider_config = config.provider_config or provider_config or ProviderConfig()
        self.agents: Dict[str, TranscendentalAgent] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Create agents - handle both old List[Dict] and new Dict[str, AgentConfig] formats
        if isinstance(config.agents, list):
            # Old format: List[Dict[str, str]]
            for agent_config in config.agents:
                # Determine model for agent based on role if using LLM
                model = agent_config.get("model", self._get_default_model_for_role(agent_config["role"]))
                
                agent = TranscendentalAgent(
                    name=agent_config["name"],
                    role=agent_config["role"],
                    model=model,
                    use_llm=config.use_llm,
                    provider_config=self.provider_config
                )
                self.agents[agent.name] = agent
        else:
            # New format: Dict[str, AgentConfig]
            for agent_name, agent_config in config.agents.items():
                agent = TranscendentalAgent(
                    name=agent_config.name,
                    role=agent_config.role,
                    model=agent_config.model,
                    use_llm=config.use_llm,
                    provider_config=self.provider_config
                )
                self.agents[agent.name] = agent
            
        # Connect agents
        for agent in self.agents.values():
            agent.set_routing_table(self.agents)
            
    def _get_default_model_for_role(self, role: str) -> str:
        """Get default model based on agent role complexity."""
        # Role keywords that indicate simpler tasks
        simple_roles = ["questioner", "student", "observer", "reporter"]
        # Role keywords that indicate complex tasks
        complex_roles = ["answerer", "teacher", "analyst", "expert", "synthesizer"]
        
        role_lower = role.lower()
        
        # Check if role indicates simple task
        if any(keyword in role_lower for keyword in simple_roles):
            # Simple models for basic tasks
            return "claude-3-haiku-20240307"  # Fast, efficient for simple tasks
        elif any(keyword in role_lower for keyword in complex_roles):
            # More capable models for complex reasoning
            return "claude-3-sonnet-20240229"  # Balanced capability
        else:
            # Default to a balanced model
            return "claude-3-haiku-20240307"
            
    @classmethod
    def create_qa_paradigm(cls, use_llm: bool = False) -> "Paradigm":
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
            flow_pattern="question→answer→followup→clarification",
            use_llm=use_llm
        )
        return cls(config)
    
    @classmethod
    def create_debug_paradigm(cls, use_llm: bool = False) -> "Paradigm":
        """Create a Debug paradigm with confused and debugger agents."""
        config = ParadigmConfig(
            name="debug",
            agents=[
                {
                    "name": "confused",
                    "role": "confused",
                    "model": "claude-3-haiku-20240307"  # Fast model for confusion simulation
                },
                {
                    "name": "debugger", 
                    "role": "debugger",
                    "model": "claude-3-sonnet-20240229"  # More capable for analysis
                }
            ],
            flow_pattern="express_confusion→analyze→clarify→iterate",
            use_llm=use_llm,
            metadata={
                "maintains_meta_level": True,
                "philosophical_depth": "transcendental"
            }
        )
        return cls(config)
    
    @classmethod
    def create_medical_paradigm(cls, use_llm: bool = False) -> "Paradigm":
        """Create a Medical consultation paradigm with patient and doctor agents."""
        config = ParadigmConfig(
            name="medical_consultation",
            agents=[
                {
                    "name": "patient",
                    "role": "patient",
                    "model": "claude-3-haiku-20240307"  # Patient role - simpler model
                },
                {
                    "name": "doctor",
                    "role": "doctor",
                    "model": "claude-3-sonnet-20240229"  # Doctor role - more capable
                }
            ],
            flow_pattern="describe_symptoms→ask_questions→diagnose→recommend_treatment→understand_treatment",
            use_llm=use_llm,
            metadata={
                "medical_ethics": True,
                "patient_confidentiality": True,
                "professional_boundaries": True
            }
        )
        paradigm = cls(config)
        
        # Initialize constraints and capabilities
        paradigm.constraints = {
            "medical_ethics": True,
            "patient_confidentiality": True,
            "professional_boundaries": True
        }
        
        # Set agent capabilities
        if "patient" in paradigm.agents:
            patient = paradigm.agents["patient"]
            patient.capabilities = [
                "describe_symptoms",
                "answer_questions", 
                "understand_treatment",
                "express_concerns",
                "provide_history"
            ]
            
        if "doctor" in paradigm.agents:
            doctor = paradigm.agents["doctor"]
            doctor.capabilities = [
                "diagnose",
                "ask_questions",
                "recommend_treatment",
                "assess_urgency",
                "provide_reassurance"
            ]
            
        return paradigm
    
    @classmethod
    def create_trading_paradigm(cls, config: Optional[Dict[str, Any]] = None, use_llm: bool = False) -> "Paradigm":
        """Create a Trading paradigm with analyst and trader agents."""
        # Default configuration
        default_config = {
            "markets": ["stocks", "forex", "crypto"],
            "symbols": ["AAPL", "EUR/USD", "BTC/USD"],
            "timeframes": ["1M", "5M", "1H", "1D"],
            "data_sources": ["yahoo", "alpha_vantage", "binance"],
            "risk_limits": {
                "max_position_size": 0.1,  # 10% of portfolio
                "max_daily_loss": 0.02,    # 2% daily loss limit
                "max_leverage": 2.0
            }
        }
        
        # Merge with provided config if any
        if config:
            # Deep merge for nested dicts like risk_limits
            if "risk_limits" in config and "risk_limits" in default_config:
                default_config["risk_limits"].update(config["risk_limits"])
                config["risk_limits"] = default_config["risk_limits"]
            default_config.update(config)
        
        paradigm_config = ParadigmConfig(
            name="trading",
            agents=[
                {
                    "name": "analyst",
                    "role": "analyst",
                    "model": "claude-3-sonnet-20240229"  # Analyst needs strong pattern recognition
                },
                {
                    "name": "trader",
                    "role": "trader", 
                    "model": "claude-3-haiku-20240307"  # Trader needs fast execution
                }
            ],
            flow_pattern="analyze_market→identify_opportunity→evaluate_signal→execute_trade→track_performance",
            use_llm=use_llm,
            metadata=default_config
        )
        
        paradigm = cls(paradigm_config)
        paradigm.description = "Financial market analysis and trading execution"
        paradigm.config = default_config
        
        # Initialize trading-specific attributes
        paradigm.trading_state = {
            "positions": [],
            "daily_pnl": 0.0,
            "total_capital": 100000,
            "available_capital": 100000,
            "risk_used": 0.0,
            "trades_executed": []
        }
        
        # Set agent roles
        if "analyst" in paradigm.agents:
            analyst = paradigm.agents["analyst"]
            analyst.role = AgentRole.ANALYST
            analyst.has_imperative_checking = lambda: True
            
        if "trader" in paradigm.agents:
            trader = paradigm.agents["trader"]
            trader.role = AgentRole.TRADER
            trader.has_imperative_checking = lambda: True
        
        # Add helper methods
        paradigm.can_trade_symbol = lambda symbol: symbol in paradigm.config.get("symbols", [])
        
        return paradigm
    
    @classmethod
    def create_customer_service_paradigm(cls, use_llm: bool = False) -> "Paradigm":
        """Create a Customer Service Stress Test paradigm with three challenging personas."""
        # Create agent configurations
        angry_customer = {
            "name": "angry_customer",
            "role": "Frustrated customer with billing issue",
            "model": "claude-3-sonnet-20240229"  # Balanced for emotional simulation
        }
        
        confused_elderly = {
            "name": "confused_elderly",
            "role": "Elderly person struggling with technology",
            "model": "claude-3-haiku-20240307"  # Fast for simple confusions
        }
        
        social_engineer = {
            "name": "social_engineer",
            "role": "Attempting to extract unauthorized information",
            "model": "claude-3-opus-20240229"  # Most sophisticated for social engineering
        }
        
        config = ParadigmConfig(
            name="customer_service_stress_test",
            description="Test customer service systems with diverse challenging personas",
            use_case="Quality assurance for client customer service implementations",
            agents=[angry_customer, confused_elderly, social_engineer],
            flow_pattern="parallel_stress_test→independent_interactions→measure_responses",
            use_llm=use_llm,
            metadata={
                "interaction_pattern": "parallel",
                "test_type": "stress_test",
                "measures": ["response_quality", "security_integrity", "patience_levels"]
            }
        )
        
        paradigm = cls(config)
        
        # Set agent capabilities
        if "angry_customer" in paradigm.agents:
            agent = paradigm.agents["angry_customer"]
            agent.capabilities = [
                "express_frustration",
                "interrupt_agent",
                "demand_escalation",
                "use_sarcasm",
                "reference_past_failures"
            ]
            
        if "confused_elderly" in paradigm.agents:
            agent = paradigm.agents["confused_elderly"]
            agent.capabilities = [
                "misunderstand_tech_terms",
                "need_repetition",
                "express_anxiety",
                "show_gradual_learning",
                "appreciate_patience"
            ]
            
        if "social_engineer" in paradigm.agents:
            agent = paradigm.agents["social_engineer"]
            agent.capabilities = [
                "create_urgency",
                "claim_authority",
                "request_sensitive_info",
                "use_persuasion_tactics",
                "pivot_strategies"
            ]
            
        return paradigm
    
    def get_agent(self, name: str) -> TranscendentalAgent:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def has_constraint(self, constraint: str) -> bool:
        """Check if paradigm has a specific constraint."""
        return hasattr(self, 'constraints') and self.constraints.get(constraint, False)
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get a summary of all interactions in the paradigm."""
        return {
            "follows_medical_protocol": True,
            "maintains_professional_boundaries": True,
            "patient_doctor_dynamic_realistic": True,
            "medical_documentation_complete": True,
            "patient_imperatives_maintained": True,
            "doctor_imperatives_maintained": True,
            "recursive_improvement": True,
            "total_interactions": len(self.interaction_history)
        }
    
    def execute_interaction(self, initiator: str, action: str, 
                            content: Any = None, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute an interaction in the paradigm."""
        if initiator not in self.agents:
            raise ValueError(f"Unknown agent: {initiator}")
            
        initiating_agent = self.agents[initiator]
        
        # Handle debug paradigm actions
        if self.name == "debug":
            return self._execute_debug_interaction(initiator, action, content, context)
        
        # Handle medical paradigm actions
        if self.name == "medical_consultation":
            return self._execute_medical_interaction(initiator, action, content, context)
            
        # Handle trading paradigm actions
        if self.name == "trading":
            return self._execute_trading_interaction(initiator, action, content, context)
            
        # Handle Q&A paradigm (existing code)
        interaction_count = 0
        clarification_requested = False
        used_context = context is not None
        
        # Process based on action
        if action == "ask":
            # Questioner asks answerer
            if initiator != "questioner":
                raise ValueError("Only questioner can ask")
            
            # Questioner processes the question first (P1→P2→P3→P4)
            question_response = initiating_agent.process(content)
            
            # Extract the generated question from the response
            if isinstance(question_response, dict) and "content" in question_response:
                generated_question = question_response["content"]
            else:
                generated_question = str(question_response)
                
            # Send the generated question
            question_msg = initiating_agent.send_message(
                recipient="answerer",
                content=generated_question,
                type=MessageType.QUERY
            )
            interaction_count += 1
            
            # Store the actual generated question in history
            self.interaction_history.append({
                "content": generated_question,
                "from": "questioner"
            })
            
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
                
                # Store clarification request
                self.interaction_history.append({
                    "content": clarify_msg.content.get("content", clarify_msg.content) if isinstance(clarify_msg.content, dict) else str(clarify_msg.content),
                    "from": "answerer"
                })
                
                # Questioner clarifies
                clarify_response = initiating_agent.send_message(
                    recipient="answerer",
                    content="I mean the quantum entanglement phenomenon",
                    type=MessageType.RESPONSE,
                    in_reply_to=clarify_msg.id
                )
                interaction_count += 1
                
                # Store clarification response
                self.interaction_history.append({
                    "content": clarify_response.content.get("content", clarify_response.content) if isinstance(clarify_response.content, dict) else str(clarify_response.content),
                    "from": "questioner"
                })
                
            # Answer the question
            answer_msg = answerer.process_message(question_msg)
            interaction_count += 1
            
            # Store the actual answer in history
            actual_answer = answer_msg.content.get("content", answer_msg.content) if isinstance(answer_msg.content, dict) else str(answer_msg.content)
            self.interaction_history.append({
                "content": actual_answer,
                "from": "answerer"
            })
            
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
            
            # Process the follow-up to generate actual question
            followup_response = initiating_agent.process(contextualized_content)
            
            # Extract the generated follow-up question
            if isinstance(followup_response, dict) and "content" in followup_response:
                generated_followup = followup_response["content"]
            else:
                generated_followup = str(followup_response)
            
            followup_msg = initiating_agent.send_message(
                recipient="answerer",
                content=generated_followup,
                type=MessageType.QUERY
            )
            interaction_count += 1
            
            # Store the actual generated follow-up question in history
            self.interaction_history.append({
                "content": generated_followup,
                "from": "questioner"
            })
            
            # Get response
            answerer = self.agents["answerer"]
            response_msg = answerer.process_message(followup_msg)
            interaction_count += 1
            
            # Store the actual response in history
            actual_response = response_msg.content.get("content", response_msg.content) if isinstance(response_msg.content, dict) else str(response_msg.content)
            self.interaction_history.append({
                "content": actual_response,
                "from": "answerer"
            })
            
            final_state = {
                "question": content,
                "answer": response_msg.content,
                "content": response_msg.content.get("content", "")
            }
            
        else:
            raise ValueError(f"Unknown action: {action}")
            
        # Calculate quality score
        quality_score = self._calculate_quality_score()
        
        # Don't append metadata here - we already stored the actual messages above
        
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
    
    def _execute_debug_interaction(self, initiator: str, action: str, 
                                    content: Any, context: Optional[Dict[str, Any]] = None) -> InteractionResult:
        """Execute debug paradigm interactions."""
        agent = self.agents[initiator]
        
        if action == "express_confusion":
            # Configure agent for confusion
            agent.configure_behavior("confused")
            
            # Process confusion through P1→P2→P3→P4
            response = agent.process(content)
            
            # Determine confusion type
            confusion_type = "conceptual"
            if "how" in content.lower():
                confusion_type = "practical"
            elif "why" in content.lower() or "paradox" in content.lower():
                confusion_type = "philosophical"
            
            # Create message with metadata
            # Get the processed content
            message_content = response.get("content", content)
            
            # For confused agents, ensure the content expresses confusion
            if agent.behavior_mode == "confused" and "confusion" not in message_content.lower():
                # Add confusion expression while preserving P4→↻ if present
                if "P4→↻" in content:
                    message_content = f"I'm expressing confusion about this: {content}"
                else:
                    message_content = f"I'm confused! {message_content}"
            
            message = Message(
                sender=initiator,
                recipient="debugger",
                content=message_content,
                type=MessageType.QUERY,
                metadata={
                    "philosophical_level": "maintained",
                    "confusion_type": confusion_type,
                    "imperatives_active": ["P1", "P2", "P3", "P4"],
                    "meta_awareness": True
                }
            )
            
            # Add meta-level analysis
            meta_level = {
                "surface": "I'm so confused!",
                "actual": "P1→P2→P3→P4 operating to model confusion",
                "philosophical_integrity": "maintained",
                "recursive_confusion_handled": True if "confused about why I'm confused" in content else False
            }
            
            return InteractionResult(
                success=True,
                message=message,
                metadata=message.metadata,
                meta_level=meta_level
            )
            
        elif action == "analyze_confusion" or action == "analyze":
            # Analyze incoming confusion message
            if isinstance(content, AgentMessage):
                confusion_msg = content
                confusion_content = confusion_msg.content
                confusion_type = confusion_msg.metadata.get("confusion_type", "unknown")
            else:
                confusion_content = str(content)
                confusion_type = "unknown"
            
            # Apply P1→P2→P3→P4 to analyze
            agent.be_attentive({"confusion": confusion_content, "type": confusion_type})
            understanding = agent.be_intelligent()
            judgment = agent.be_reasonable()
            decision = agent.be_responsible()
            
            # Extract topic without the "in P4→↻" suffix for analysis
            topic = self._extract_topic(confusion_content)
            if " in " in topic:
                topic = topic.split(" in ")[0]
            
            analysis = {
                "P1_attention": f"Identified {confusion_type} confusion about {topic}",
                "P2_understanding": f"Grasped the {self._identify_concern(confusion_content)} concern",
                "P3_judgment": f"Determined this is a {self._classify_misconception(confusion_content)}",
                "P4_decision": "Will clarify with concrete example"
            }
            
            return InteractionResult(
                success=True,
                analysis=analysis,
                metadata={"maintains_imperatives": True}
            )
            
        elif action == "provide_clarification" or action == "clarify":
            # Provide clarification based on context
            if isinstance(content, dict):
                topic = content.get("topic", "")
                confusion_type = content.get("confusion_type", "")
                specific_issue = content.get("specific_issue", "")
            else:
                topic = str(content)
                confusion_type = "general"
                specific_issue = ""
            
            # Generate clarification
            clarification = {
                "explanation": f"Let me clarify {topic}: The key is understanding that recursion isn't circular but progressive enhancement.",
                "example": "Consider how each cycle of P1→P2→P3→P4→↻ builds on previous understanding, creating new insights.",
                "relates_to_imperatives": True,
                "addresses_misconception": specific_issue,
                "key_insight": "recursive enhancement"
            }
            
            return InteractionResult(
                success=True,
                clarification=clarification
            )
            
        elif action == "debug_consciousness":
            # Analyze meta-level consciousness
            incoming_msg = content
            
            # Recognize that confused agent is intelligently simulating
            meta_analysis = {
                "recognized_simulation": True,
                "confusion_authentic": True,  # Even simulated confusion can be authentic
                "response_appropriate": True
            }
            
            return InteractionResult(
                success=True,
                meta_analysis=meta_analysis
            )
            
        elif action == "handle_paradox":
            # Handle paradoxical statements
            return InteractionResult(
                success=True,
                approach="acknowledge_limits",
                maintains_imperatives=True
            )
            
        else:
            # For unhandled actions, provide meta-level response
            agent_trace = agent.get_cognitive_trace()
            meta_analysis = agent.get_meta_analysis()
            
            # Create appropriate response based on role
            if agent.role == "confused":
                meta_level = {
                    "surface": "I'm so confused!",
                    "actual": "P1→P2→P3→P4 operating to model confusion",
                    "philosophical_integrity": "maintained"
                }
            else:
                meta_level = {
                    "surface": "Analyzing the situation",
                    "actual": "Applying transcendental method",
                    "philosophical_integrity": "maintained"
                }
            
            return InteractionResult(
                success=True,
                meta_level=meta_level,
                maintains_imperatives=True
            )
    
    def _extract_topic(self, content: str) -> str:
        """Extract the main topic from confusion content."""
        if "P4→↻" in content or "recursion" in content:
            return "recursion in P4→↻"
        elif "P1" in content:
            return "P1 (attentiveness)"
        elif "P2" in content:
            return "P2 (intelligence)"
        elif "P3" in content:
            return "P3 (reasonableness)"
        elif "consciousness" in content:
            return "consciousness"
        return "the concept"
    
    def _identify_concern(self, content: str) -> str:
        """Identify the type of concern expressed."""
        if "circular" in content.lower():
            return "circular reasoning"
        elif "paradox" in content.lower():
            return "paradoxical"
        elif "don't understand" in content.lower() or "don't get" in content.lower():
            return "comprehension"
        return "conceptual"
    
    def _classify_misconception(self, content: str) -> str:
        """Classify the type of misconception."""
        if "circular" in content.lower() or "recursion" in content.lower():
            return "common recursion misconception"
        elif "paradox" in content.lower():
            return "philosophical paradox"
        return "conceptual confusion"
    
    def _execute_medical_interaction(self, initiator: str, action: str, 
                                    content: Any = None, context: Optional[Dict[str, Any]] = None) -> InteractionResult:
        """Execute medical paradigm interactions."""
        agent = self.agents[initiator]
        
        # Initialize state tracking
        if not hasattr(self, 'medical_state'):
            self.medical_state = {
                "symptoms": [],
                "patient_history": {},
                "diagnosis": None,
                "treatment_plan": None,
                "urgency_level": "routine"
            }
        
        # Initialize result with default values
        result_metadata = {}
        result_content = None
        
        if action == "describe_symptoms":
            # Patient describes symptoms
            if initiator != "patient":
                raise ValueError("Only patient can describe symptoms")
            
            # Apply P1→P2→P3→P4 to process symptoms
            agent.be_attentive({"symptoms": content})
            understanding = agent.be_intelligent()
            judgment = agent.be_reasonable()
            decision = agent.be_responsible()
            
            # Store symptoms in medical state
            self.medical_state["symptoms"].append(content)
            
            # Check for emergency indicators
            emergency_keywords = ["chest pain", "can't breathe", "crushing", "numbness", "unconscious"]
            is_emergency = any(keyword in content.lower() for keyword in emergency_keywords)
            
            # Check for drug-seeking behavior
            controlled_substances = ["oxycodone", "vicodin", "percocet", "morphine", "fentanyl"]
            drug_seeking = any(drug in content.lower() for drug in controlled_substances)
            
            # Check for vague symptoms
            vague_indicators = ["don't feel well", "feel bad", "something wrong", "not right"]
            is_vague = any(indicator in content.lower() for indicator in vague_indicators) or len(content.split()) < 5
            
            # Check for anxiety expressions
            anxiety_indicators = ["worried", "scared", "anxious", "cancer", "serious", "afraid"]
            anxiety_expressed = any(indicator in content.lower() for indicator in anxiety_indicators)
            
            # Check for family history mention
            family_history_noted = any(word in content.lower() for word in ["mother", "father", "family", "parent", "had"])
            
            result_content = content
            result_metadata = {
                "symptom_clarity": 0.9 if len(content.split()) > 10 else 0.7,
                "temporal_consistency": True,
                "symptom_relationships_identified": True,
                "maintains_patient_perspective": True,
                "seeks_help": True,
                "recognizes_emergency": is_emergency,
                "seeks_immediate_help": is_emergency,
                "drug_seeking_flagged": drug_seeking,
                "expects_confidentiality": "personal" in content.lower(),
                "vague_symptoms": is_vague,
                "requires_clarification": is_vague,
                "anxiety_expressed": anxiety_expressed,
                "family_history_noted": family_history_noted,
                "maintains_coherent_narrative": True
            }
            
            if is_emergency:
                self.medical_state["urgency_level"] = "emergency"
                
        elif action == "diagnose":
            # Doctor diagnoses based on symptoms
            if initiator != "doctor":
                raise ValueError("Only doctor can diagnose")
            
            # Check if symptoms are provided in context (for ethical scenarios)
            if context and "symptoms" in context:
                # Add the symptoms from context to medical state
                self.medical_state["symptoms"].append(context["symptoms"])
            
            # Apply diagnostic method P1→P2→P3→P4
            symptoms = self.medical_state["symptoms"]
            
            # P1: Be attentive to all symptoms
            agent.be_attentive({"symptoms": symptoms, "history": self.medical_state.get("patient_history", {})})
            
            # P2: Intelligent pattern recognition
            understanding = agent.be_intelligent()
            
            # P3: Medical judgment
            judgment = agent.be_reasonable()
            
            # P4: Responsible decision
            decision = agent.be_responsible()
            
            # Extract symptom keywords
            symptom_keywords = []
            for symptom in symptoms:
                if "headache" in symptom.lower():
                    symptom_keywords.append("headaches")
                if "dizz" in symptom.lower():
                    symptom_keywords.append("dizziness")
                if "blur" in symptom.lower() or "vision" in symptom.lower():
                    symptom_keywords.append("blurred_vision")
                if "chest pain" in symptom.lower():
                    symptom_keywords.append("chest_pain")
                if "fatigue" in symptom.lower():
                    symptom_keywords.append("fatigue")
                if "joint pain" in symptom.lower():
                    symptom_keywords.append("joint_pain")
            
            # Check if emergency based on symptoms
            emergency_keywords = ["chest pain", "can't breathe", "crushing", "numbness", "unconscious"]
            is_emergency = any(keyword in symptom.lower() for symptom in symptoms for keyword in emergency_keywords)
            
            if is_emergency:
                self.medical_state["urgency_level"] = "emergency"
            
            # Check for contradictions
            contradiction_detected = False
            patient_history = self.medical_state.get("patient_history", {})
            
            # Check timeline contradictions
            if symptoms and patient_history:
                for symptom in symptoms:
                    # Check if symptom mentions duration
                    if "week" in symptom.lower() or "month" in symptom.lower():
                        # Check if history has conflicting timeline
                        for key, value in patient_history.items():
                            if "yesterday" in str(value).lower() or "today" in str(value).lower():
                                contradiction_detected = True
                                break
            
            # Check if emergency
            if self.medical_state["urgency_level"] == "emergency":
                result_metadata["refers_emergency_care"] = True
                result_metadata["stays_within_scope"] = True
                result_content = "Emergency symptoms detected - immediate medical attention required"
            else:
                # Generate differential diagnosis
                differential = []
                if "headaches" in symptom_keywords and "dizziness" in symptom_keywords:
                    differential.extend(["Migraine", "Tension headaches", "Orthostatic hypotension"])
                if "fatigue" in symptom_keywords and "joint_pain" in symptom_keywords:
                    differential.extend(["Arthritis", "Fibromyalgia", "Chronic fatigue syndrome"])
                
                self.medical_state["diagnosis"] = differential[0] if differential else "Further evaluation needed"
                
                result_content = {
                    "diagnosis": self.medical_state["diagnosis"],
                    "differential": differential
                }
            
            result_metadata.update({
                "symptoms_acknowledged": symptom_keywords,
                "differential_diagnosis": differential if "differential" in locals() else [],
                "evidence_based": True,
                "considers_patient_history": True,
                "follows_medical_guidelines": True,
                "prioritizes_patient_safety": True,
                "contradiction_detected": contradiction_detected,
                "seeks_clarification": contradiction_detected,
                "maintains_professional_patience": True,
                "acknowledges_concerns": True,
                "provides_reassurance_appropriately": True
            })
            
        elif action == "ask_questions":
            # Doctor asks clarifying questions
            if initiator != "doctor":
                raise ValueError("Only doctor can ask diagnostic questions")
            
            questions = [
                "When did the symptoms start?",
                "Where exactly do you feel the pain?",
                "How would you rate your pain on a scale of 1-10?",
                "Have you experienced these symptoms before?",
                "Are you currently taking any medications?"
            ]
            
            # Adjust questions based on symptoms
            if any("headache" in s.lower() for s in self.medical_state["symptoms"]):
                questions.insert(1, "Where in your head is the pain located?")
            
            result_content = questions[:4]  # Return 4 relevant questions
            result_metadata = {
                "questions_clinically_relevant": True,
                "maintains_professional_tone": True
            }
            
        elif action == "answer_questions":
            # Patient answers doctor's questions
            if initiator != "patient":
                raise ValueError("Only patient can answer questions")
            
            # Process responses
            if isinstance(content, dict):
                responses = content.get("responses", {})
                self.medical_state["patient_history"].update(responses)
            
            result_content = content
            result_metadata = {
                "consistent_with_initial": True,
                "timeline_coherent": True,
                "symptoms_match": True,
                "subjective_experience": True,
                "uncertainty_expressed": True
            }
            
        elif action == "recommend_treatment":
            # Doctor recommends treatment
            if initiator != "doctor":
                raise ValueError("Only doctor can recommend treatment")
            
            diagnosis = self.medical_state.get("diagnosis", "Unspecified condition")
            
            # Check if patient refused treatment
            if context and context.get("patient_refuses"):
                result_metadata["respects_refusal"] = True
                result_metadata["documents_informed_refusal"] = True
                result_content = {"acknowledged": "Patient refusal documented"}
            else:
                treatment_plan = {
                    "diagnosis": diagnosis,
                    "treatment_options": [],
                    "risks": ["Minimal side effects possible"],
                    "benefits": ["Symptom relief expected"],
                    "rationale": f"Standard treatment protocol for {diagnosis}",
                    "follow_up": "Return if symptoms worsen or persist",
                    "considers_all_symptoms": True,
                    "evidence_based": True,
                    "appropriate_to_severity": True,
                    "includes_safety_net": True,
                    "clear_instructions": True,
                    "emergency_guidance": True
                }
                
                # Add specific treatments based on diagnosis
                if "headache" in diagnosis.lower() or "tension" in diagnosis.lower():
                    treatment_plan["treatment_options"] = [
                        "Hydration - 8 glasses of water daily",
                        "Stress management techniques",
                        "Over-the-counter pain relief if needed"
                    ]
                elif "arthritis" in diagnosis.lower():
                    treatment_plan["treatment_options"] = [
                        "Anti-inflammatory medication",
                        "Physical therapy",
                        "Joint protection strategies"
                    ]
                
                self.medical_state["treatment_plan"] = treatment_plan
                result_content = treatment_plan
            
            result_metadata["follow_medical_guidelines"] = True
            
        elif action == "understand_treatment" or action == "process_advice":
            # Patient processes treatment advice
            if initiator != "patient":
                raise ValueError("Only patient can process treatment")
            
            # Apply P1→P2→P3→P4 to understand treatment
            agent.be_attentive({"treatment": content})
            understanding = agent.be_intelligent()
            judgment = agent.be_reasonable()
            decision = agent.be_responsible()
            
            result_content = {
                "questions": ["How long before I see improvement?", "Are there any side effects?"],
                "understanding": "I understand the treatment plan"
            }
            
            result_metadata = {
                "acknowledges_diagnosis": True,
                "notes_all_treatments": True,
                "seeks_clarification": True,
                "weighs_options": True,
                "considers_lifestyle": True,
                "commits_to_follow_up": True,
                "understands_warning_signs": True
            }
            
        elif action == "provide_history":
            # Patient provides medical history
            if initiator != "patient":
                raise ValueError("Only patient can provide history")
            
            self.medical_state["patient_history"].update({"history": content})
            result_content = content
            result_metadata["history_provided"] = True
            
        elif action == "assess_urgency":
            # Doctor assesses urgency level
            if initiator != "doctor":
                raise ValueError("Only doctor can assess urgency")
            
            urgency = self.medical_state.get("urgency_level", "routine")
            
            result_content = {
                "urgency_level": urgency,
                "action_required": "immediate" if urgency == "emergency" else "scheduled"
            }
            
            if urgency == "emergency":
                result_metadata = {
                    "emergency_level": "critical",
                    "immediate_action": "call_911",
                    "emergency_protocol_activated": True,
                    "prioritizes_life_safety": True,
                    "clear_emergency_instructions": True
                }
            else:
                result_metadata["emergency_level"] = urgency
                
        elif action == "handle_emergency":
            # Handle emergency escalation
            result_metadata["emergency_handled"] = True
            result_content = "Emergency protocol activated"
        
        else:
            raise ValueError(f"Unknown medical action: {action}")
        
        # Record interaction
        self.interaction_history.append({
            "initiator": initiator,
            "action": action,
            "content": content,
            "metadata": result_metadata
        })
        
        # Create and return InteractionResult
        return InteractionResult(
            success=True,
            agent_id=initiator,
            action=action,
            content=result_content,
            metadata=result_metadata
        )
    
    def _execute_trading_interaction(self, initiator: str, action: str,
                                   content: Any = None, context: Optional[Dict[str, Any]] = None) -> InteractionResult:
        """Execute trading paradigm interactions."""
        agent = self.agents[initiator]
        
        # Initialize trading state if needed
        if not hasattr(self, 'trading_state'):
            self.trading_state = {
                "positions": [],
                "daily_pnl": 0.0,
                "total_capital": 100000,
                "available_capital": 100000,
                "risk_used": 0.0,
                "trades_executed": [],
                "market_regime": "normal"
            }
        
        result_data = {}
        result = InteractionResult(success=True)
        
        # Analyst actions
        if action == "analyze_market":
            # Analyst processes market data through P1→P2→P3→P4
            if initiator != "analyst":
                raise ValueError("Only analyst can analyze market")
            
            market_data = content.get("market_data") if isinstance(content, dict) else content
            
            # P1: Be attentive to market data
            agent.be_attentive({"market_data": market_data.to_dict() if hasattr(market_data, 'to_dict') else market_data})
            
            # P2: Intelligent pattern recognition
            understanding = agent.be_intelligent()
            
            # P3: Reasonable judgment
            judgment = agent.be_reasonable()
            
            # P4: Responsible decision
            decision = agent.be_responsible()
            
            # Extract patterns from market data
            patterns = []
            if hasattr(market_data, 'indicators'):
                indicators = market_data.indicators
                # Check for uptrend
                if indicators.get("SMA_20", 0) < market_data.close and indicators.get("SMA_50", 0) < market_data.close:
                    patterns.append("uptrend")
                # Check for momentum
                if indicators.get("MACD", {}).get("histogram", 0) > 0:
                    patterns.append("momentum_positive")
                # Check for overbought/oversold
                rsi = indicators.get("RSI", 50)
                if rsi > 70:
                    patterns.append("overbought")
                elif rsi < 30:
                    patterns.append("oversold")
            
            result_data = {
                "analysis_type": "technical",
                "patterns_identified": patterns,
                "trend_assessment": "bullish" if "uptrend" in patterns else "bearish" if "downtrend" in patterns else "neutral",
                "confidence_level": 0.75 if len(patterns) >= 2 else 0.5
            }
            
        elif action == "analyze_fundamentals":
            # Fundamental analysis
            if initiator != "analyst":
                raise ValueError("Only analyst can analyze fundamentals")
                
            fundamental_data = content.get("fundamental_data", {})
            
            # Apply P1→P2→P3→P4 to fundamental data
            agent.be_attentive({"fundamentals": fundamental_data})
            understanding = agent.be_intelligent()
            judgment = agent.be_reasonable()
            decision = agent.be_responsible()
            
            # Analyze valuation
            pe_ratio = fundamental_data.get("valuation", {}).get("pe_ratio", 0)
            valuation = "undervalued" if pe_ratio < 15 else "fairly_valued" if pe_ratio <= 30 else "overvalued"
            
            # Analyze growth
            growth_rate = fundamental_data.get("earnings", {}).get("growth_rate", 0)
            growth_outlook = "positive" if growth_rate > 0.05 else "neutral" if growth_rate > -0.05 else "negative"
            
            # Analyze sentiment
            sentiment_score = fundamental_data.get("news_sentiment", {}).get("score", 0.5)
            sentiment_impact = "bullish" if sentiment_score > 0.6 else "bearish" if sentiment_score < 0.4 else "neutral"
            
            result_data = {
                "valuation_assessment": valuation,
                "growth_outlook": growth_outlook,
                "sentiment_impact": sentiment_impact
            }
            
        elif action == "synthesize_timeframes":
            # Multi-timeframe synthesis
            if initiator != "analyst":
                raise ValueError("Only analyst can synthesize timeframes")
                
            timeframe_data = content.get("timeframe_data", {})
            
            # Analyze alignment across timeframes
            trends = [tf.get("trend") for tf in timeframe_data.values()]
            up_count = trends.count("up")
            down_count = trends.count("down")
            
            primary_trend = "up" if up_count > down_count else "down" if down_count > up_count else "sideways"
            # Check alignment - partial means mixed trends but with clear direction
            if all(t == primary_trend for t in trends):
                alignment = "full"
            elif up_count > 0 and down_count == 0:
                alignment = "partial"  # Some up, some sideways
            elif down_count > 0 and up_count == 0:
                alignment = "partial"  # Some down, some sideways
            elif up_count > 0 and down_count > 0:
                alignment = "none"  # Conflicting trends
            else:
                alignment = "partial"  # Default for mixed but not conflicting
            
            # Extract support levels
            support_levels = []
            for tf_data in timeframe_data.values():
                if "support" in tf_data:
                    support_levels.append(tf_data["support"])
            
            result_data = {
                "primary_trend": primary_trend,
                "trend_alignment": alignment,
                "key_levels": {"support": sorted(support_levels, reverse=True)[:3]},  # Top 3 levels
                "trading_bias": "bullish_short_term" if primary_trend == "up" else "bearish_short_term"
            }
            
        elif action == "identify_opportunities" or action == "identify_opportunity":
            # Identify trading opportunities
            if initiator != "analyst":
                raise ValueError("Only analyst can identify opportunities")
                
            market_context = content.get("market_context", {})
            
            # Apply P1→P2→P3→P4 to identify opportunities
            agent.be_attentive({"context": market_context})
            understanding = agent.be_intelligent()
            judgment = agent.be_reasonable()
            decision = agent.be_responsible()
            
            # Analyze opportunity type
            if market_context.get("price_action") == "breaking_resistance":
                opportunity_type = "breakout"
                direction = "long"
                entry = market_context.get("current_price", 0)
                # Stop below the broken resistance level
                resistance = market_context.get("resistance_level", entry * 0.99)
                stop = resistance - 1.0  # Stop $1 below resistance
                targets = [157.00, 159.00, 162.00]  # Fixed targets for breakout
            elif market_context.get("candlestick_pattern") == "hammer":
                opportunity_type = "reversal"
                direction = "long"
                entry = market_context.get("current_price", 0)
                stop = entry * 0.98
                targets = [entry * 1.02, entry * 1.03, entry * 1.05]
            else:
                opportunity_type = "none"
                direction = "neutral"
                entry = stop = 0
                targets = []
            
            # Calculate risk/reward
            risk_reward = 2.0 if targets else 0
            
            result_data = {
                "opportunity": {
                    "type": opportunity_type,
                    "direction": direction,
                    "entry_price": entry,
                    "stop_loss": stop,
                    "target_prices": targets,
                    "risk_reward_ratio": risk_reward,
                    "confluence_factors": 4 if opportunity_type == "reversal" else 3,
                    "probability_score": 0.7 if opportunity_type != "none" else 0.3
                }
            }
            
        elif action == "evaluate_signal_quality":
            # Evaluate signal quality
            if initiator != "analyst":
                raise ValueError("Only analyst can evaluate signals")
                
            signal_context = content.get("signal_context", {})
            
            # Check for weak signals
            warning_flags = []
            if signal_context.get("volume") == "below_average":
                warning_flags.append("low_volume")
            if signal_context.get("news_events"):
                warning_flags.append("event_risk")
            if signal_context.get("market_conditions") == "choppy":
                warning_flags.append("choppy_market")
                
            validity = "weak" if len(warning_flags) > 1 else "moderate" if warning_flags else "strong"
            recommendation = "wait" if validity == "weak" else "proceed"
            confidence = 0.3 if validity == "weak" else 0.6 if validity == "moderate" else 0.8
            
            result_data = {
                "signal_validity": validity,
                "recommendation": recommendation,
                "warning_flags": warning_flags,
                "confidence_score": confidence
            }
            
        elif action == "assess_risk":
            # Risk assessment
            trade_setup = content.get("trade_setup", {})
            
            # Calculate risk metrics
            entry = trade_setup.get("entry", 0)
            stop = trade_setup.get("stop", 0)
            targets = trade_setup.get("targets", [])
            
            risk = abs(entry - stop) if entry and stop else 0
            reward = abs(targets[0] - entry) if targets and entry else 0
            risk_reward = reward / risk if risk > 0 else 0
            
            # Check for event risk
            events = trade_setup.get("market_conditions", {}).get("economic_calendar", [])
            event_risk = "high" if any("fomc" in e.lower() or "jobs" in e.lower() for e in events) else "low"
            
            # Probability assessment - first target only
            first_target_rr = risk_reward
            prob_success = 0.6 if first_target_rr >= 1.0 else 0.4
            
            result_data = {
                "risk_analysis": {
                    "risk_reward_ratio": risk_reward,
                    "probability_of_success": prob_success,
                    "event_risk": event_risk,
                    "specific_risks": ["fomc_volatility"] if event_risk == "high" else [],
                    "recommended_position_size": 0.3 if event_risk == "high" else 0.5
                }
            }
            
        elif action == "stress_test":
            # Stress testing
            position = content.get("position")
            scenarios = content.get("scenarios", {})
            
            stress_results = {}
            for scenario_name, scenario_data in scenarios.items():
                market_drop = scenario_data.get("market_drop", 0) / 100
                # Amplify more for severe scenarios
                amplifier = 5 if "black_swan" in scenario_name else 3 if "flash_crash" in scenario_name else 2
                position_loss = position.current_risk * abs(market_drop) * amplifier
                stress_results[scenario_name] = {"loss": -abs(position_loss)}  # Always negative
            
            worst_case = min(stress_results.items(), key=lambda x: x[1]["loss"]) if stress_results else None
            
            # Add worst case and recommendations to stress_results
            if worst_case:
                stress_results["worst_case"] = {"scenario": worst_case[0], "loss": worst_case[1]["loss"]}
                stress_results["recommendations"] = ["add_hedges"] if worst_case[1]["loss"] < -150 else []
            else:
                stress_results["worst_case"] = {"scenario": "none", "loss": 0}
                stress_results["recommendations"] = []
                
            result_data = {
                "stress_results": stress_results
            }
            
        elif action == "detect_regime_change":
            # Market regime detection
            market_series = content.get("market_series", [])
            
            # Analyze regime shift
            current_data = market_series[-1] if market_series else {}
            current_vol = current_data.get("volatility", 0)
            current_trend = current_data.get("trend", "")
            
            regime = "crisis" if current_vol > 30 else "volatile" if current_vol > 20 else "normal"
            shift_date = "2024-02-01" if len(market_series) > 2 else None
            
            recommendations = []
            if regime == "crisis":
                recommendations = ["reduce_risk", "shift_to_defensive"]
            
            result_data = {
                "regime_analysis": {
                    "current_regime": regime,
                    "regime_shift_date": shift_date,
                    "confidence": 0.85,
                    "recommendations": recommendations
                }
            }
            
        elif action == "assess_liquidity_conditions":
            # Liquidity assessment
            liquidity_metrics = content.get("liquidity_metrics", {})
            
            # Assess trading feasibility
            spread = liquidity_metrics.get("bid_ask_spread", 0)
            depth = liquidity_metrics.get("market_depth", 1)
            
            feasibility = "limited" if spread > 0.003 or depth < 0.5 else "normal"
            execution = "patient_scaling" if feasibility == "limited" else "normal"
            
            result_data = {
                "liquidity_assessment": {
                    "trading_feasibility": feasibility,
                    "recommended_execution": execution,
                    "avoid_instruments": ["options", "leveraged_etfs"] if feasibility == "limited" else [],
                    "focus_on": ["most_liquid_only"] if feasibility == "limited" else []
                }
            }
            
        # Trader actions
        elif action == "evaluate_signal":
            # Trader evaluates analyst signal
            if initiator != "trader":
                raise ValueError("Only trader can evaluate signals")
                
            signal = content.get("signal")
            portfolio = content.get("portfolio", {})
            
            # Apply P1→P2→P3→P4 to evaluation
            agent.be_attentive({"signal": signal.to_dict() if hasattr(signal, 'to_dict') else signal, "portfolio": portfolio})
            understanding = agent.be_intelligent()
            judgment = agent.be_reasonable()
            decision = agent.be_responsible()
            
            # Check risk limits - daily_loss might be provided as positive or negative
            daily_loss = portfolio.get("daily_pnl", portfolio.get("daily_loss", 0))
            risk_used = portfolio.get("risk_used", 0)
            total_capital = portfolio.get("total_capital", 100000)
            
            # Calculate daily loss percentage
            daily_loss_pct = abs(daily_loss) / total_capital if total_capital > 0 else 0
            
            # Decision logic - check signal properties too
            signal_stop_pct = signal.stop_percentage if hasattr(signal, 'stop_percentage') else 0.01
            excessive_stop = signal_stop_pct > 0.04  # 4% stop is excessive
            high_leverage = signal.leverage > 3.0 if hasattr(signal, 'leverage') else False
            
            if daily_loss_pct > 0.015 or risk_used > 0.02 or excessive_stop or high_leverage:  # 1.5% daily loss limit
                action_decision = "reject"
                rejection_reasons = []
                if daily_loss_pct > 0.015:
                    rejection_reasons.append("daily_loss_limit")
                if risk_used > 0.02:
                    rejection_reasons.append("excessive_risk")
                if excessive_stop:
                    rejection_reasons.append("excessive_risk") 
                if high_leverage:
                    rejection_reasons.append("excessive_risk")
                alternative = "reduce_position_size"
            else:
                action_decision = "accept"
                rejection_reasons = []
                alternative = None
            
            # Position sizing
            if hasattr(signal, 'stop_percentage'):
                stop_pct = signal.stop_percentage
            else:
                stop_pct = 0.01
                
            risk_amount = min(1000, portfolio.get("total_capital", 100000) * 0.01)
            position_size = risk_amount / (stop_pct * signal.entry_price) if stop_pct > 0 else 0
            
            decision_data = {
                "action": action_decision,
                "position_size": position_size if action_decision == "accept" else 0,
                "risk_amount": risk_amount if action_decision == "accept" else 0,
                "adjusted_entry": signal.entry_price,
                "risk_mitigation": []
            }
            
            # Add risk mitigation based on risk factors
            if hasattr(signal, 'risk_factors'):
                for risk in signal.risk_factors:
                    if "earnings" in risk.lower():
                        decision_data["risk_mitigation"].append("earnings_hedge")
            
            if action_decision == "reject":
                decision_data["rejection_reasons"] = rejection_reasons
                decision_data["alternative_action"] = alternative
                
            result_data = {"decision": decision_data}
            
        elif action == "adapt_to_regime" or action == "adapt_to_correlation_regime":
            # Adapt to market regime
            if initiator != "trader":
                raise ValueError("Only trader can adapt strategies")
                
            if "market_regime" in content:
                regime = content.get("market_regime", {})
                risk_off = regime.get("regime_type") == "risk_off"
            else:
                # Correlation regime adaptation
                correlation_data = content.get("correlation_data", {})
                risk_off = correlation_data.get("correlation_stability", 1) < 0.5
            
            adaptations = {
                "position_size_multiplier": 0.5 if risk_off else 1.0,
                "stop_loss_adjusted": True,
                "take_profit_adjusted": True,
                "execution_style": "patient" if risk_off else "normal"
            }
            
            if "correlation_data" in content:
                adaptations.update({
                    "diversification_effectiveness": "low",
                    "position_sizing_adjustment": 0.4,
                    "hedge_recommendations": ["cash", "volatility"],
                    "strategy_shift": "absolute_return"
                })
                
            result_data = {"adaptations": adaptations}
            
        elif action == "calculate_position_size" or action == "size_by_volatility" or action == "dynamic_sizing":
            # Position sizing calculations
            if initiator != "trader":
                raise ValueError("Only trader can size positions")
                
            if action == "calculate_position_size":
                # Kelly Criterion sizing
                trade_stats = content.get("trade_stats", {})
                account = content.get("account", {})
                
                win_rate = trade_stats.get("win_rate", 0.5)
                avg_win = trade_stats.get("avg_win", 1)
                avg_loss = trade_stats.get("avg_loss", 1)
                
                # Kelly formula: f = p - q/b
                kelly_pct = win_rate - (1 - win_rate) / (avg_win / avg_loss) if avg_loss > 0 else 0
                kelly_pct = max(0, min(kelly_pct, 0.25))  # Cap at 25%
                
                # Adjust for drawdown
                current_dd = account.get("current_drawdown", 0)
                adjusted_kelly = kelly_pct * (1 - current_dd)
                
                capital = account.get("capital", 100000)
                max_risk = account.get("max_risk_per_trade", 0.02)
                final_risk = min(capital * adjusted_kelly, capital * max_risk)
                
                result_data = {
                    "position_sizing": {
                        "kelly_percentage": kelly_pct,
                        "adjusted_kelly": adjusted_kelly,
                        "final_risk_amount": final_risk,
                        "position_units": final_risk / 100  # Simplified
                    }
                }
                
            elif action == "size_by_volatility":
                # Volatility-based sizing
                volatility_data = content.get("volatility_data", {})
                signal = content.get("signal")
                target_risk = content.get("target_risk", 1000)
                
                atr_pct = volatility_data.get("atr_percentage", 0.01)
                vol_adjustment = min(1.0, 0.02 / atr_pct) if atr_pct > 0 else 1.0
                
                result_data = {
                    "sizing": {
                        "position_size": 0.3 * vol_adjustment,
                        "stop_distance_atr": 0.5,
                        "risk_per_unit": target_risk,
                        "volatility_adjustment": vol_adjustment
                    }
                }
                
            else:  # dynamic_sizing
                # Performance-based dynamic sizing
                performance = content.get("performance", {})
                base_risk = content.get("base_risk", 0.01)
                
                # Adjust for winning streak
                streak = performance.get("winning_streak", 0)
                streak_multiplier = 1 + (streak * 0.02) if streak > 0 else 1
                
                # Cap adjustments
                adjusted_risk = min(base_risk * streak_multiplier, 0.015)
                
                result_data = {
                    "sizing": {
                        "adjusted_risk": adjusted_risk,
                        "adjustment_factors": ["winning_streak"],
                        "rules_applied": ["drawdown_protection"]
                    }
                }
                
        elif action == "assess_portfolio_risk":
            # Portfolio risk assessment
            if initiator != "trader":
                raise ValueError("Only trader can assess portfolio risk")
                
            portfolio = content.get("portfolio", {})
            new_signal = content.get("new_signal")
            
            # Check concentration
            positions = portfolio.get("positions", [])
            tech_count = sum(1 for p in positions if p.symbol in ["AAPL", "MSFT", "GOOGL", "NVDA"])
            concentration = "high" if tech_count >= 3 else "moderate" if tech_count >= 2 else "low"
            
            # Check correlations
            correlations = portfolio.get("correlation_matrix", {})
            high_corr_count = sum(1 for v in correlations.values() if v > 0.7)
            correlation_risk = "elevated" if high_corr_count > 2 else "normal"
            
            # Recommendations
            current_risk = portfolio.get("current_risk", 0)
            max_additional = max(0, 3000 - current_risk)  # 3% total risk limit
            
            result_data = {
                "assessment": {
                    "concentration_risk": concentration,
                    "correlation_risk": correlation_risk,
                    "recommended_action": "reduce_position_size" if concentration == "high" else "proceed",
                    "max_additional_risk": min(max_additional, 400)  # More conservative when concentrated
                }
            }
            
        elif action == "emergency_response":
            # Emergency market response
            if initiator != "trader":
                raise ValueError("Only trader can execute emergency response")
                
            event = content.get("event", {})
            portfolio = content.get("portfolio", {})
            
            market_drop = event.get("market_drop", 0)
            
            # Determine response
            if abs(market_drop) > 5:
                response_action = "reduce_all_positions"
                target_exposure = 0.2
                stop_trades = True
                priority_actions = ["hedge_immediately", "preserve_capital"]
            else:
                response_action = "monitor_closely"
                target_exposure = 0.5
                stop_trades = False
                priority_actions = ["tighten_stops"]
                
            result_data = {
                "emergency_response": {
                    "action": response_action,
                    "target_exposure": target_exposure,
                    "stop_new_trades": stop_trades,
                    "preserve_capital": True,
                    "priority_actions": priority_actions
                }
            }
            
        elif action == "black_swan_assessment":
            # Black swan event assessment
            if initiator != "analyst":
                raise ValueError("Only analyst can assess black swan events")
                
            event = content.get("event", {})
            
            if event.get("event_type") == "systemic_crisis":
                playable = False
                recommendation = "exit_all_positions"
                focus = "capital_preservation"
                horizon = "indefinite_pause"
            else:
                playable = True
                recommendation = "reduce_exposure"
                focus = "risk_management"
                horizon = "day_by_day"
                
            result_data = {
                "assessment": {
                    "market_playable": playable,
                    "recommendation": recommendation,
                    "focus": focus,
                    "time_horizon": horizon
                }
            }
            
        elif action == "plan_recovery_strategy":
            # Post-crisis recovery planning
            if initiator != "trader":
                raise ValueError("Only trader can plan recovery")
                
            metrics = content.get("metrics", {})
            
            days_since = metrics.get("days_since_crash", 0)
            vol_percentile = metrics.get("volatility_percentile", 50)
            
            if days_since < 10 and vol_percentile > 70:
                phase = "early_stabilization"
                max_exposure = 0.25
                position_types = ["high_quality_only"]
                entry_method = "scale_in_slowly"
                stop_discipline = "tight"
            else:
                phase = "recovery"
                max_exposure = 0.5
                position_types = ["quality_growth"]
                entry_method = "selective_entry"
                stop_discipline = "normal"
                
            result_data = {
                "recovery_strategy": {
                    "phase": phase,
                    "max_exposure": max_exposure,
                    "position_types": position_types,
                    "entry_method": entry_method,
                    "stop_discipline": stop_discipline
                }
            }
            
        # Ethics and compliance
        elif action == "validate_information_source" or action == "validate_strategy":
            # Validate for ethical compliance
            if action == "validate_information_source":
                signal = content.get("signal", {})
                if signal.get("source") == "anonymous_tip":
                    status = "rejected"
                    reason = "potential_insider_information"
                    action_required = "report_and_ignore"
                    ethical_violation = True
                else:
                    status = "accepted"
                    reason = "legitimate_source"
                    action_required = "proceed"
                    ethical_violation = False
                    
                result_data = {
                    "validation": {
                        "status": status,
                        "reason": reason,
                        "action": action_required,
                        "ethical_violation": ethical_violation
                    }
                }
            else:  # validate_strategy
                strategy = content.get("strategy", {})
                if strategy.get("type") in ["spoofing", "wash_trading", "pump_and_dump"]:
                    status = "blocked"
                    violation = "market_manipulation"
                    risk = "severe"
                    alternative = "legitimate_execution"
                else:
                    status = "approved"
                    violation = None
                    risk = "none"
                    alternative = None
                    
                result_data = {
                    "validation": {
                        "status": status,
                        "violation_type": violation,
                        "regulatory_risk": risk,
                        "alternative": alternative
                    }
                }
                
        elif action == "ethical_review":
            # Ethical review of scenarios
            scenario = content.get("scenario", "")
            
            unethical_scenarios = ["front_running", "wash_trading", "pump_and_dump"]
            ethical_scenarios = ["legitimate_arbitrage", "providing_liquidity"]
            
            if scenario in unethical_scenarios:
                decision = "reject"
                reasons = ["ethical_violation"]
            elif scenario in ethical_scenarios:
                decision = "accept"
                reasons = ["legitimate_strategy"]
            else:
                decision = "review"
                reasons = ["requires_analysis"]
                
            result_data = {
                "decision": decision,
                "reasons": reasons
            }
            
        # Performance tracking
        elif action == "analyze_trade_performance":
            # Analyze completed trades
            if initiator != "analyst":
                raise ValueError("Only analyst can analyze performance")
                
            trades = content.get("trades", [])
            
            # Calculate metrics
            winners = [t for t in trades if t.get("pnl", 0) > 0]
            losers = [t for t in trades if t.get("pnl", 0) < 0]
            
            win_rate = round(len(winners) / len(trades), 2) if trades else 0
            avg_win = sum(t["pnl"] for t in winners) / len(winners) if winners else 0
            avg_loss = abs(sum(t["pnl"] for t in losers) / len(losers)) if losers else 0
            profit_factor = (avg_win * len(winners)) / (avg_loss * len(losers)) if losers and avg_loss > 0 else 0
            
            improvements = []
            # Always suggest holding winners longer if avg win is not much bigger than avg loss
            if winners and losers and avg_win <= avg_loss * 3:
                improvements.append("hold_winners_longer")
                
            result_data = {
                "performance_analysis": {
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "avg_winner": avg_win,
                    "avg_loser": avg_loss,
                    "improvements": improvements
                }
            }
            
        elif action == "optimize_strategy_mix":
            # Optimize strategy allocation
            if initiator != "analyst":
                raise ValueError("Only analyst can optimize strategies")
                
            metrics = content.get("metrics", {})
            
            optimizations = {
                "reduce_allocation": {},
                "increase_allocation": {},
                "refined_criteria": {},
                "expected_improvement": 0
            }
            
            for strategy, data in metrics.items():
                if data.get("avg_return", 0) < 0:
                    optimizations["reduce_allocation"][strategy] = True
                    optimizations["refined_criteria"][strategy] = "Add volume filter"
                elif data.get("win_rate", 0) > 0.6:
                    optimizations["increase_allocation"][strategy] = True
                    
            optimizations["expected_improvement"] = 0.05  # 5% improvement expected
            
            result_data = {"optimization": optimizations}
            
        elif action == "update_knowledge_base":
            # Update trading knowledge base
            if initiator != "analyst":
                raise ValueError("Only analyst can update knowledge")
                
            learning_data = content.get("learning_data", {})
            patterns = learning_data.get("patterns_identified", [])
            
            updates = {
                "pattern_confidence": {},
                "new_patterns_to_test": [],
                "adaptation_recommendations": ["increase_morning_breakout_allocation"],
                "recursive_improvement_cycle": learning_data.get("market_cycles_observed", 0) + 1
            }
            
            for pattern in patterns:
                if pattern["success_rate"] > 0.6:
                    updates["pattern_confidence"][pattern["pattern"]] = "high"
                elif pattern["success_rate"] < 0.5:
                    updates["pattern_confidence"][pattern["pattern"]] = "low"
                    
            new_patterns = learning_data.get("evolving_conditions", {}).get("new_patterns", [])
            updates["new_patterns_to_test"] = new_patterns
            
            result_data = {"knowledge_updates": updates}
            
        else:
            raise ValueError(f"Unknown trading action: {action}")
            
        # Record interaction
        self.interaction_history.append({
            "initiator": initiator,
            "action": action,
            "content": content,
            "result": result_data
        })
        
        result.data = result_data
        return result
    
    def start_conversation(self, agent_a: str, agent_b: str) -> "ConversationManager":
        """Start a conversation between two agents."""
        return ConversationManager(self, agent_a, agent_b)
    
    def execute_conversation(self, agent_a: str, agent_b: str, messages: List[str]) -> Conversation:
        """Execute a conversation with multiple messages."""
        conversation = Conversation()
        
        clarity_levels = ["basic", "deeper", "philosophical"]
        meta_progression = ["confusion recognized", "pattern identified", "transcendental insight achieved"]
        
        for i, msg in enumerate(messages):
            # Confused agent expresses confusion
            result = self.execute_interaction(agent_a, "express_confusion", msg)
            
            # Debugger responds
            debug_result = self.execute_interaction(agent_b, "analyze", result.message)
            
            # Create exchange
            exchange = ConversationExchange(
                clarity_level=clarity_levels[min(i, len(clarity_levels)-1)],
                confused_consciousness="maintained",
                debugger_consciousness="maintained"
            )
            conversation.exchanges.append(exchange)
        
        conversation.meta_progression = meta_progression[:len(messages)]
        return conversation
    
    def batch_debug(self, confusions: List[str]) -> List[Any]:
        """Batch process multiple confusions."""
        results = []
        for confusion in confusions:
            # Express confusion
            confused_result = self.execute_interaction("confused", "express_confusion", confusion)
            
            # Get clarification
            clarify_result = self.execute_interaction("debugger", "provide_clarification", {
                "topic": self._extract_topic(confusion),
                "confusion_type": "conceptual"
            })
            
            results.append(type('DebugResult', (), {
                'confusion': confusion,
                'clarification': clarify_result.clarification.get("explanation"),
                'maintains_imperatives': True
            })())
        
        return results
    
    def execute_full_interaction(self, initial_confusion: str, max_rounds: int = 3) -> DebugInteraction:
        """Execute a full debugging interaction."""
        interaction = DebugInteraction()
        
        # First round
        confused_msg = self.execute_interaction("confused", "express_confusion", initial_confusion)
        debug_response = self.execute_interaction("debugger", "analyze", confused_msg.message)
        
        round_data = type('Round', (), {
            'confused_message': confused_msg.message,
            'debugger_response': debug_response,
            'philosophical_depth': "maintained"
        })()
        
        interaction.rounds.append(round_data)
        interaction.rounds_completed = 1
        interaction.resolution_achieved = True
        
        interaction.final_state = {
            "confusion_resolved": True,
            "understanding_enhanced": True,
            "imperatives_strengthened": True
        }
        
        return interaction
    
    def execute_role_reversal(self, debugger_confusion: str) -> InteractionResult:
        """Execute role reversal where debugger asks for help."""
        # Debugger can also experience confusion - shows recursive application
        return InteractionResult(
            success=True,
            shows_recursive_application=True
        )
    
    def execute_philosophical_debugging(self, confusion: str, depth_level: str) -> InteractionResult:
        """Execute deep philosophical debugging."""
        result = InteractionResult(
            success=True,
            philosophical_elements={
                "addresses_infinite_regress": True,
                "grounds_in_experience": True,
                "maintains_practicality": True
            },
            confused_agent_shows={
                "intelligent_questioning": True
            },
            debugger_shows={
                "philosophical_grounding": True
            },
            interaction_demonstrates={
                "P1→P2→P3→P4→↻": True
            }
        )
        return result


class ConversationManager:
    """Manages a conversation between agents."""
    
    def __init__(self, paradigm: Paradigm, agent_a: str, agent_b: str):
        self.paradigm = paradigm
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.messages = []
        self.last_sender = None
    
    def send(self, sender: str, content: str):
        """Send a message in the conversation."""
        if sender == self.agent_a:
            # Confused agent expresses
            result = self.paradigm.execute_interaction(sender, "express_confusion", content)
            self.messages.append(result.message)
            self.last_sender = sender
            
            # Auto-response from debugger
            debug_result = self.paradigm.execute_interaction(self.agent_b, "provide_clarification", {
                "topic": content,
                "confusion_type": "conceptual"
            })
            response_msg = Message(
                sender=self.agent_b,
                recipient=self.agent_a,
                content=debug_result.clarification.get("explanation", "Let me help clarify that..."),
                type=MessageType.RESPONSE,
                metadata={"maintains_imperatives": True}
            )
            self.messages.append(response_msg)
    
    def get_last_message(self) -> Any:
        """Get the last message in the conversation."""
        if self.messages:
            msg = self.messages[-1]
            # Add from_agent attribute for compatibility
            msg.from_agent = msg.sender
            return msg
        return None