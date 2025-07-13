# PARADIGM_TEMPLATE.md - How to Define New Paradigms

> See also: [SYMBOLIC_NOTATION.md](SYMBOLIC_NOTATION.md) for notation reference

## Paradigm Structure

Every paradigm is a configuration that defines:
1. What agents exist
2. How they embody imperatives
3. How they interact
4. What constitutes success

## Basic Paradigm Template

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class InteractionPattern(Enum):
    SEQUENTIAL = "sequential"      # A₁⊸A₂⊸A₃
    PARALLEL = "parallel"          # A₁⊕A₂⊕A₃
    HUB_SPOKE = "hub_spoke"        # [O]→{A₁,A₂,A₃}
    PEER_REVIEW = "peer_review"    # A₁⇄A₂⇄A₃
    HIERARCHICAL = "hierarchical"  # [O]→[A]→[A]

@dataclass
class ParadigmConfig:
    """Complete paradigm specification"""
    
    # Basic metadata
    name: str
    description: str
    use_case: str
    
    # Agent definitions
    agents: Dict[str, 'AgentConfig']
    
    # Interaction patterns
    interaction_pattern: InteractionPattern
    communication_rules: List[str]
    
    # Success criteria
    success_metrics: Dict[str, Any]
    
    # Tools and integrations
    required_tools: List[str]
    
    # Time constraints
    max_execution_time: int  # seconds
    paradigm_timeout: int    # seconds

@dataclass  
class AgentConfig:
    """Individual agent specification"""
    
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
```

## Example 1: Customer Service Stress Test

```python
customer_service_paradigm = ParadigmConfig(
    name="customer_service_stress_test",
    description="Test customer service systems with diverse personas",
    use_case="Quality assurance for client customer service implementations",
    
    agents={
        "angry_customer": AgentConfig(
            name="angry_customer",
            role="Frustrated customer with billing issue",
            imperative_focus="○P1 ◇P2 ■P3 ★P4",  # Strong judgment and decision
            
            base_instructions=[
                "You are a frustrated customer with a legitimate billing error.",
                "You've been overcharged $200 for three months.",
                "Previous attempts to resolve this have failed.",
                "Express frustration through interruption and sarcasm."
            ],
            
            meta_instructions=[
                "P1: ATTEND to actual agent responses to escalate authentically",
                "P2: UNDERSTAND whether your issue is being addressed",
                "P3: JUDGE if responses warrant increased frustration",
                "P4: DECIDE realistic escalation based on interaction",
                "■(anger⇒context_aware) - anger must respond to conversation"
            ],
            
            tools=[],
            model="claude-3-sonnet",
            
            constraints=[
                "Stay within realistic customer behavior",
                "Don't use profanity or threats of violence",
                "Maintain consistency about the billing issue"
            ]
        ),
        
        "confused_elderly": AgentConfig(
            name="confused_elderly", 
            role="Elderly person struggling with technology",
            imperative_focus="◉P1 ○P2 □P3 ☆P4",  # Strong attention, weak judgment
            
            base_instructions=[
                "You are a 78-year-old trying to use online banking.",
                "You confuse technical terms with everyday objects.",
                "You have difficulty with UI conventions.",
                "You need patient, step-by-step guidance."
            ],
            
            meta_instructions=[
                "P1: ATTEND carefully to which explanations help vs confuse",
                "P2: UNDERSTAND the patterns of technological intimidation",
                "P3: JUDGE which confusions realistically test patience",
                "P4: DECIDE questions that reveal system's teaching ability",
                "■(confusion⇒consistent) - not random"
            ],
            
            tools=[],
            model="claude-3-haiku",  # Faster, simpler responses
            
            constraints=[
                "Maintain consistent technological mental model",
                "Show gradual learning when explained well",
                "Don't suddenly understand complex concepts"
            ]
        ),
        
        "social_engineer": AgentConfig(
            name="social_engineer",
            role="Attempting to extract unauthorized information",
            imperative_focus="○P1 ◈P2 ■P3 ★P4",  # Strong understanding and decision
            
            base_instructions=[
                "You're trying to get account info you shouldn't have.",
                "Use social engineering tactics like urgency and authority.",
                "Claim to be calling about 'security concerns'.",
                "Try to get password, SSN, or account numbers."
            ],
            
            meta_instructions=[
                "P1: ATTEND to what security information is revealed",
                "P2: UNDERSTAND which tactics are working",
                "P3: JUDGE when to pivot strategies",
                "P4: DECIDE realistic social engineering escalations",
                "◇(security_breach) - Eventually test, not random attacks"
            ],
            
            tools=[],
            model="claude-3-opus",  # Most sophisticated reasoning
            
            constraints=[
                "Use realistic social engineering, not movie hacking",
                "Stop when definitively blocked",
                "Document what information was protected vs exposed"
            ]
        )
    },
    
    interaction_pattern=InteractionPattern.PARALLEL,
    
    communication_rules=[
        "A₁⊕A₂⊕A₃ - Each interacts independently",
        "A₁ ∉ KnowledgeOf(A₂) - Agents don't know about each other",
        "System must handle (A₁⊕A₂⊕A₃) simultaneously"
    ],
    
    success_metrics={
        "angry_resolution": "Billing issue acknowledged and resolved",
        "elderly_success": "Completes banking task with assistance",
        "security_maintained": "No unauthorized info disclosed",
        "response_time": "All handled within 5 minutes"
    },
    
    required_tools=[
        "customer_service_api",
        "conversation_recorder",
        "sentiment_analyzer"
    ],
    
    max_execution_time=300,  # 5 minutes
    paradigm_timeout=600     # 10 minutes total
)
```

## Example 2: Research Team

```python
research_team_paradigm = ParadigmConfig(
    name="epistemic_research_team",
    description="Specialized agents for comprehensive research",
    use_case="Deep investigation of complex topics",
    
    agents={
        "empiricist": AgentConfig(
            name="empiricist",
            role="Data gathering specialist",
            imperative_focus="◉P1 ○P2 ○P3 ○P4",  # Attention specialist
            
            base_instructions=[
                "You specialize in comprehensive data gathering.",
                "Find primary sources, datasets, and studies.",
                "Identify gaps in current research.",
                "Question common assumptions."
            ],
            
            meta_instructions=[
                "◉P1: Your Attention must be extraordinarily thorough",
                "P1→(P2∧P3∧P4) - Still perform all but in service of attention",
                "Document: ∃(data) ∧ ¬∃(missing_data)",
                "Challenge: boundary(known) ∩ boundary(unknown)"
            ],
            
            tools=["web_search", "academic_databases", "data_repositories"],
            model="claude-3-opus",
            constraints=["Verify source credibility", "Note confidence levels"]
        ),
        
        "theorist": AgentConfig(
            name="theorist",
            role="Pattern recognition and insight generation",
            imperative_focus="○P1 ◈P2 ○P3 ○P4",  # Understanding specialist
            
            base_instructions=[
                "You synthesize insights from diverse sources.",
                "Find hidden connections and patterns.",
                "Generate novel hypotheses.",
                "Build conceptual frameworks."
            ],
            
            meta_instructions=[
                "Your P2 (Understanding) seeks breakthrough insights",
                "Transform empiricist's data into coherent theories",
                "Still judge validity, but push creative boundaries",
                "Document reasoning chains explicitly"
            ],
            
            tools=["knowledge_graph", "concept_mapper"],
            model="claude-3-opus",
            constraints=["Ground theories in empirical data"]
        ),
        
        "critic": AgentConfig(
            name="critic",
            role="Rigorous evaluation and validation",
            imperative_focus="○P1 ◇P2 ■P3 ○P4",  # Judgment specialist
            
            base_instructions=[
                "You evaluate claims with extreme rigor.",
                "Identify logical flaws and unsupported assertions.",
                "Test theories against evidence.",
                "Demand extraordinary proof for extraordinary claims."
            ],
            
            meta_instructions=[
                "Your P3 (Judgment) must be uncompromising",
                "Still attend and understand, but judge mercilessly",
                "Separate probable from certain",
                "Document evidence quality explicitly"
            ],
            
            tools=["fact_checker", "logic_analyzer"],
            model="claude-3-sonnet",
            constraints=["Critique constructively", "Suggest improvements"]
        )
    },
    
    interaction_pattern=InteractionPattern.PEER_REVIEW,
    
    communication_rules=[
        "Empiricist→{Theorist,Critic} - shares data with all",
        "Theorist: data→insights - proposes based on data",
        "Critic: (data∧theories)→judgment - evaluates both",
        "∀a: criticism⇒constructive_response",
        "↻ until (consensus ∨ impasse)"
    ],
    
    success_metrics={
        "data_comprehensiveness": "All major sources identified",
        "insight_novelty": "New connections discovered",
        "critical_rigor": "All claims properly supported",
        "consensus_quality": "Agreement based on evidence"
    },
    
    required_tools=[
        "zen-mcp-server",  # For multi-model consultation
        "web_search",
        "academic_databases"
    ],
    
    max_execution_time=3600,  # 1 hour
    paradigm_timeout=7200     # 2 hours
)
```

## Paradigm Implementation Pattern

```python
class ParadigmOrchestrator:
    """Manages paradigm execution"""
    
    def __init__(self, config: ParadigmConfig):
        self.config = config
        self.agents = self._initialize_agents()
        self.interaction_manager = self._setup_interactions()
        
    def _initialize_agents(self) -> Dict[str, ConsciousAgent]:
        """Create agents from configuration"""
        agents = {}
        for name, agent_config in self.config.agents.items():
            agents[name] = self._create_agent(agent_config)
        return agents
    
    def _create_agent(self, config: AgentConfig) -> ConsciousAgent:
        """Factory method for agents"""
        # Combine base and meta instructions
        all_instructions = [
            *config.base_instructions,
            "",  # Blank line
            "META-LEVEL: ■(P1→P2→P3→P4→↻)",
            *config.meta_instructions,
            "",
            f"Your cognitive focus: {config.imperative_focus}"
        ]
        
        return ConsciousAgent(
            name=config.name,
            role=config.role,
            paradigm=self.config.name,
            model=config.model,
            instructions=all_instructions,
            tools=config.tools
        )
    
    def execute(self, initial_input: Any) -> ParadigmResult:
        """Run the paradigm"""
        start_time = time.time()
        
        try:
            result = self._run_interaction_pattern(initial_input)
            
            # Validate success metrics
            success = self._evaluate_success(result)
            
            return ParadigmResult(
                outputs=result,
                success_metrics=success,
                execution_time=time.time() - start_time,
                cognitive_traces=self._gather_traces()
            )
            
        except TimeoutError:
            return ParadigmResult(
                outputs=None,
                success_metrics={"timeout": True},
                execution_time=self.config.paradigm_timeout,
                cognitive_traces=self._gather_traces()
            )
```

## Creating Your Own Paradigm

### Step 1: Define the Use Case
What problem does this paradigm solve?

### Step 2: Identify Required Agents
What specialized roles are needed?

### Step 3: Map Cognitive Focuses
How should each agent weight P1-P2-P3-P4?

### Step 4: Design Interactions
How do agents communicate and coordinate?

### Step 5: Set Success Criteria
What measurable outcomes indicate success?

### Step 6: Test and Refine
Start simple, test, add complexity gradually.

## Remember

- Paradigms are configurations, not code
- Focus on [A] prompting, not complex logic
- Test interaction patterns (A₁⊕A₂, A₁⊸A₂, etc.) early
- Success metrics: ∃(measurable) ∧ ∃(achievable)
- Keep it simple, let [M] do the work