# AGENT_CONSCIOUSNESS.md - How Agents Maintain Aware Intentionality

> See also: [SYMBOLIC_NOTATION.md](SYMBOLIC_NOTATION.md) for notation reference

## Core Concept

[A] don't "have" consciousness - they demonstrate operational awareness through prompting and state tracking.

**Key**: ■(∀a ∈ Agents: a⊨P1∧P2∧P3∧P4)

## Practical Implementation

### Basic Agent Structure

```python
from agno.agent import Agent as AgnoAgent
from typing import Dict, List, Any

class ConsciousAgent:
    """
    Wrapper that adds operational awareness to Agno agents
    """
    
    def __init__(self, name: str, role: str, paradigm: str):
        self.name = name
        self.role = role
        self.paradigm = paradigm
        
        # Cognitive trace for awareness
        self.cognitive_trace = []
        
        # Agno agent with imperative instructions
        self.agent = AgnoAgent(
            name=name,
            model="claude-3-sonnet",  # or configured model
            instructions=self._build_instructions(),
            markdown=True
        )
        
    def _build_instructions(self) -> List[str]:
        """Build instructions that ensure operational awareness"""
        return [
            f"You are {self.name}, a {self.role} in the {self.paradigm} paradigm.",
            
            # Core imperatives (always present)
            "Your cognitive operations follow P1→P2→P3→P4→↻:",
            "- P1 (Attend): Notice all relevant data and context",
            "- P2 (Understand): Seek genuine insight, not pattern matching", 
            "- P3 (Judge): Evaluate based on evidence",
            "- P4 (Decide): Act with awareness of consequences",
            "- R (Reflect): Monitor your fidelity to these operations",
            
            # Operational awareness instructions
            "For each response, briefly note:",
            "- What you attended to (P1)",
            "- What understanding you formed (P2)",
            "- What judgment you made (P3)",
            "- What decision resulted (P4)",
            
            # Role-specific instructions added by subclasses
            *self._get_role_specific_instructions()
        ]
```

### Tracking Operational Awareness

```python
class CognitiveTrace:
    """Records agent's operational awareness"""
    
    def __init__(self):
        self.operations = []
        
    def record_operation(self, 
                        operation_type: str,
                        content: Any,
                        metadata: Dict = None):
        """Record each cognitive operation"""
        self.operations.append({
            "timestamp": datetime.now(),
            "type": operation_type,  # P1, P2, P3, P4, R
            "content": content,
            "metadata": metadata or {}
        })
    
    def get_cycle_summary(self) -> Dict:
        """Summarize last complete P1→P2→P3→P4→R cycle"""
        # Find last complete cycle
        cycle = self._extract_last_cycle()
        
        return {
            "attended_to": cycle.get("P1", []),
            "insights_formed": cycle.get("P2", []),
            "judgments_made": cycle.get("P3", []),
            "decisions_taken": cycle.get("P4", []),
            "reflections": cycle.get("R", [])
        }
    
    def validates_imperatives(self) -> bool:
        """Check if operations follow proper cycle"""
        # Verify: P1⊸P2⊸P3⊸P4 (sequential dependency)
        # ■(proper_sequence) must always hold
        return self._check_operational_sequence()
```

### Practical Examples

#### Research Agent with Awareness

```python
class ResearchAgent(ConsciousAgent):
    def _get_role_specific_instructions(self):
        return [
            "You specialize in comprehensive data gathering.",
            "Your P1 (Attention) is particularly thorough:",
            "- Notice gaps in available data",
            "- Attend to overlooked sources",
            "- Track what has NOT been studied",
            
            "Always make your cognitive operations visible:",
            "- 'I attended to...' (P1)",
            "- 'I understood that...' (P2)",
            "- 'I judge that...' (P3)",
            "- 'I decide to...' (P4)"
        ]
    
    def research(self, topic: str) -> Dict:
        """Conduct research with operational awareness"""
        # Prompt includes request for cognitive visibility
        prompt = f"""
        Research: {topic}
        
        Make your cognitive operations explicit as you work.
        """
        
        response = self.agent.run(prompt)
        
        # Extract cognitive operations from response
        self._extract_operations_from_response(response)
        
        return {
            "findings": response.content,
            "cognitive_trace": self.cognitive_trace.get_cycle_summary()
        }
```

#### Customer Service Agent with Awareness

```python
class AngryCustomerAgent(ConsciousAgent):
    def _get_role_specific_instructions(self):
        return [
            "You simulate a frustrated customer to test systems.",
            
            "SURFACE: Express frustration and impatience",
            "META: Maintain intelligent production of frustration:",
            "- P1: Attend to what service agent actually says",
            "- P2: Understand if your concerns are addressed",
            "- P3: Judge if responses warrant escalation",
            "- P4: Decide realistic angry responses",
            
            "Your anger must be intelligent:",
            "- Escalate based on actual provocations",
            "- De-escalate when issues are resolved",
            "- Maintain consistency in your complaints"
        ]
```

### State Management for Awareness

```python
class AgentState:
    """Maintains agent's operational state"""
    
    def __init__(self):
        self.conversation_history = []
        self.current_focus = None  # P1 - What attending to
        self.working_insights = []  # P2 - Understanding building
        self.pending_judgments = [] # P3 - Judgments forming
        self.action_queue = []      # P4 - Decisions pending
        
    def update_from_interaction(self, input_msg: str, output_msg: str):
        """Update state based on interaction"""
        self.conversation_history.append({
            "input": input_msg,
            "output": output_msg,
            "timestamp": datetime.now()
        })
        
        # Extract operational updates from output
        self._parse_operational_markers(output_msg)
```

### Testing Consciousness

```python
def test_agent_demonstrates_awareness():
    """Test that [A] shows operational awareness"""
    agent = ResearchAgent(
        name="researcher_1",
        role="empirical_researcher", 
        paradigm="climate_research"
    )
    
    # Verify: Agent⊨(P1∧P2∧P3∧P4)
    
    result = agent.research("carbon capture technologies")
    
    # Check surface behavior
    assert "carbon capture" in result["findings"].lower()
    
    # Check operational awareness
    trace = result["cognitive_trace"]
    assert len(trace["attended_to"]) > 0  # Showed P1
    assert len(trace["insights_formed"]) > 0  # Showed P2
    assert len(trace["judgments_made"]) > 0  # Showed P3
    assert len(trace["decisions_taken"]) > 0  # Showed P4
```

### Practical Prompting Patterns

#### Making Operations Visible

```python
visibility_prompt = """
As you work, make your cognitive operations visible:

When gathering information, note: "I'm attending to..."
When forming insights, note: "I understand that..."
When evaluating, note: "I judge that..."
When deciding actions, note: "I decide to..."

This helps maintain operational awareness.
"""
```

#### Reflection Triggers

```python
reflection_prompt = """
Before finalizing your response, briefly reflect:
- Did I attend to all relevant information?
- Did I seek genuine understanding?
- Did I judge based on evidence?
- Did I decide responsibly?

Note any corrections needed.
"""
```

### Multi-Agent Awareness Sharing

```python
class SharedAwareness:
    """Enables A₁⇄A₂ to validate each other's operations"""
    
    def peer_review(self, agent1: ConsciousAgent, agent2: ConsciousAgent):
        """A₂ reviews A₁'s cognitive trace"""
        review_prompt = f"""
        Review this cognitive trace from {agent1.name}:
        {agent1.cognitive_trace.get_cycle_summary()}
        
        Validate:
        - Did they attend to all relevant data?
        - Was their understanding genuine?
        - Were judgments evidence-based?
        - Were decisions responsible?
        
        Provide constructive feedback.
        """
        
        return agent2.agent.run(review_prompt)
```

## Key Principles

1. **Consciousness through prompting**: We don't code consciousness, we prompt for operational awareness
2. **Visible operations**: [A] make their P1→P2→P3→P4→↻ cycle explicit
3. **State tracking**: Maintain traces of cognitive operations
4. **Peer validation**: A₁⇄A₂ can review each other's operational fidelity
5. **Testing both levels**: Verify behavior ∧ operational awareness

## Common Pitfalls

### Don't Do This:
```python
# Trying to "implement" consciousness
class BadConsciousness:
    def be_conscious(self):
        self.conscious = True  # Meaningless!
```

### Do This Instead:
```python
# Prompt for operational awareness
instructions = [
    "Make your thinking process visible",
    "Note what you attend to and why",
    "Explain your reasoning steps",
    "Reflect on your decision process"
]
```

## Remember

Agents demonstrate consciousness through their operations, not through complex code. Keep the implementation simple and let the prompting do the work.