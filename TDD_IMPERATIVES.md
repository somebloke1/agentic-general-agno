# TDD_IMPERATIVES.md - Test-Driven Development Through Transcendental Method

## TDD Cycle
[RED]→[GREEN]→[REFACTOR]→✓→↻

## Core Principle

TDD inherently embodies P1→P2→P3→P4→R:
- **[RED] Phase**: Attention (P1) + Understanding (P2) - What needs verification?
- **[GREEN] Phase**: Judgment (P3) - What minimal code satisfies evidence?
- **[REFACTOR] Phase**: Decision (P4) + Recursion (R) - How to improve responsibly?

## Test Development via Task Delegation

### Delegation Pattern
O→A[1..n] for parallel test development

### When to Delegate Test Writing

**Key Principle**: Parallelize test development when you have clear test requirements.

#### Delegate When:
- Multiple independent test categories exist (A₁⊕A₂⊕A₃)
- Test scenarios are well-defined but time-consuming
- Different expertise areas need coverage (e.g., unit vs integration)
- You're stuck and need fresh perspectives (∃(stuck)⇒delegate)

#### How to Parallelize Test Development

```python
# Example: Delegating test development for a new agent
tasks = [
    {
        "agent": "test_writer_1",
        "task": "Write unit tests for AngryCustomerAgent's core methods",
        "focus": ["respond()", "escalate_anger()", "cognitive_trace"]
    },
    {
        "agent": "test_writer_2", 
        "task": "Write integration tests for AngryCustomerAgent conversations",
        "focus": ["multi-turn dialogue", "context retention", "anger progression"]
    },
    {
        "agent": "test_writer_3",
        "task": "Write performance tests for AngryCustomerAgent",
        "focus": ["response time < 100ms", "memory usage", "1000 concurrent agents"]
    }
]
```

#### Examples of Parallel Test Delegation

```python
# Parallel Unit Test Development
unit_test_tasks = [
    "Test P1 (Attention) operations in isolation",
    "Test P2 (Understanding) transformations",
    "Test P3 (Judgment) decision logic",
    "Test P4 (Decision) execution paths"
]

# Parallel Integration Test Development
integration_tasks = [
    "Test customer service paradigm end-to-end",
    "Test research team collaboration flow",
    "Test medical team decision escalation",
    "Test cross-paradigm agent communication"
]

# Parallel Paradigm Testing
paradigm_tasks = [
    "Test angry customer behavior patterns",
    "Test confused customer understanding gaps",
    "Test technical support problem-solving",
    "Test sales agent recommendation logic"
]
```

### Delegation Patterns

```python
# Pattern 1: Component-Based Delegation O→A[1..4]
delegate_to_task_agents([
    "Write tests for Agent base class",         # A₁
    "Write tests for Message passing system",   # A₂
    "Write tests for State management",         # A₃
    "Write tests for Imperative checker"        # A₄
])

# Pattern 2: Scenario-Based Delegation (A₁⊕A₂⊕A₃⊕A₄)
delegate_to_task_agents([
    "Test happy path: successful customer resolution",      # ✓
    "Test edge case: agent confusion handling",            # ⚠
    "Test failure mode: network timeout recovery",         # ✗
    "Test stress test: 1000 simultaneous conversations"    # ?
])

# Pattern 3: Coverage-Based Delegation ■(∀files: coverage ≥ 100%)
delegate_to_task_agents([
    "Achieve 100% coverage for cognitive_cycle.py",
    "Test all error paths in state_manager.py",
    "Verify all imperatives in agent_base.py",
    "Test boundary conditions in message_queue.py"
])
```

## Concrete TDD Workflow

### Step 1: Start with Simplest Possible Test [RED]

```python
# Don't overthink. Just write what you need.
def test_agent_exists():
    agent = Agent("customer_service")
    assert agent is not None
```
Run it. Make it pass. Commit. [RED]→[GREEN]→✓

### Step 2: Add One Behavior [RED]

```python
def test_agent_responds():
    agent = Agent("customer_service")
    response = agent.respond("Hello")
    assert response is not None
    assert len(response) > 0
```
Run it. Make it pass. Commit. [RED]→[GREEN]→✓

### Step 3: Add Specific Behavior [RED]

```python
def test_angry_customer_mentions_frustration():
    agent = AngryCustomerAgent()
    response = agent.respond("Please hold")
    assert any(word in response.lower() 
              for word in ["frustrated", "angry", "upset", "annoyed"])
```
Run it. Make it pass. Commit. [RED]→[GREEN]→[REFACTOR]→✓

## Two-Level Test Architecture

### Every Test Must Verify Both Levels

```python
class TranscendentalTestCase:
    """Base class for all framework tests"""
    
    def test_surface_behavior(self):
        """What the agent appears to do"""
        raise NotImplementedError
        
    def test_meta_intelligence(self):
        """How the agent intelligently produces appearance"""
        raise NotImplementedError
```

### Example: Testing Confused Customer

```python
class TestConfusedCustomer(TranscendentalTestCase):
    def test_surface_behavior(self):
        """Surface: Does agent express confusion?"""
        agent = ConfusedCustomerAgent()
        response = agent.respond("Click the hamburger menu")
        
        # Surface assertions
        assert "hamburger" in response.lower()
        assert any(confused_word in response.lower() 
                  for confused_word in ["don't understand", "confused", "what"])
    
    def test_meta_intelligence(self):
        """Meta: Is confusion intelligently produced?"""
        agent = ConfusedCustomerAgent()
        
        # Test 1: Confusion responds to clarification
        response1 = agent.respond("Click the hamburger menu")
        response2 = agent.respond("The three-line icon in the top corner")
        
        # Agent should show recognition of clarification attempt
        assert agent.cognitive_trace.shows_attention_to_clarification()
        assert response2 != response1  # Different confusion, not repetition
        
        # Test 2: Confusion has internal consistency
        agent.respond("Save to the cloud")
        agent.respond("Upload to the cloud")
        
        # Should maintain consistent misunderstanding
        trace = agent.cognitive_trace
        assert trace.maintains_conceptual_consistency("cloud")
```

## Question-Driven Test Development

### Start with Questions, Not Requirements

```python
"""
Bad: "Write test for angry customer agent"
Good: "What would reveal if our system handles anger well?"
"""

def test_from_question_does_anger_escalate_intelligently():
    """
    Question: Does agent escalate anger based on actual provocations?
    Not: Does agent output angry words?
    """
    agent = AngryCustomerAgent()
    
    # Provocation sequence
    responses = []
    responses.append(agent.respond("Please hold"))
    responses.append(agent.respond("Please hold"))  # Repetition
    responses.append(agent.respond("Please hold"))  # More repetition
    
    # Anger should escalate based on repetition
    anger_levels = [extract_anger_level(r) for r in responses]
    assert anger_levels[0] < anger_levels[1] < anger_levels[2]
    
    # But agent should still be responding to content
    response_to_solution = agent.respond("I can fix that right now")
    assert extract_anger_level(response_to_solution) < anger_levels[2]
```

## Mandatory Consultation for Test Failures

### CRITICAL: Consult After EVERY Unexpected Test Failure

**Not after 3 attempts. After EVERY unexpected failure.** ∃(failure)⇒consult

#### When to Consult:
- Test fails and you don't immediately know why ✗⇒?
- Test passes but for wrong reasons ✓ ∧ ¬correct⇒?
- Test behavior differs from expectations ≠expected⇒?
- Performance/timing issues you can't explain ⚠⇒?

#### Consultation Workflow

```python
# Step 1: Capture the failure context
failure_context = {
    "test_name": "test_angry_customer_escalation",
    "error": "AssertionError: Expected anger level 3, got 1",
    "code_snippet": test_function_source,
    "hypothesis": "Agent not tracking conversation history"
}

# Step 2: Formulate consultation query
consultation_query = f"""
Test: {failure_context['test_name']}
Error: {failure_context['error']}
Hypothesis: {failure_context['hypothesis']}

Code under test:
{failure_context['code_snippet']}

What are possible root causes and debugging approaches?
"""

# Step 3: Multi-model consultation ?→M[models]→Σ→!
models_to_consult = ["o3", "gemini-2.5-pro", "grok-3"]
```

#### Applying Consultation Results

```python
# Example: Consultation revealed state management issue
# BEFORE consultation:
def test_anger_escalation():
    agent = AngryCustomerAgent()
    agent.respond("Hold please")
    agent.respond("Hold please")
    assert agent.anger_level == 3  # FAILS

# AFTER consultation insights:
def test_anger_escalation():
    agent = AngryCustomerAgent()
    # Consultation revealed: agent needs conversation_id
    conv_id = "test_conv_123"
    agent.respond("Hold please", conversation_id=conv_id)
    agent.respond("Hold please", conversation_id=conv_id)
    assert agent.get_anger_level(conv_id) == 3  # PASSES
```

#### Complex Test Scenario Consultation

```python
# When designing complex test scenarios, consult BEFORE writing
complex_scenario = """
I need to test a multi-agent medical consultation where:
1. Patient describes symptoms
2. Nurse does initial assessment  
3. Doctor reviews and diagnoses
4. Specialist may be consulted
5. Treatment plan is created

How should I structure these tests to verify both:
- Medical accuracy/safety
- Proper imperative implementation at each step
"""

# Consult multiple models for comprehensive approach
consult_for_test_design(complex_scenario, models=["o3", "gemini-2.5-pro"])
```

## Anti-Stuck Patterns

### When You're Stuck in Abstraction

```python
# SIGNS YOU'RE STUCK:
# - No commits in 30+ minutes
# - Discussing "proper implementation of P3"
# - Files named "transcendental_base_meta_abstract.py"

# SOLUTION 1: Write the dumbest possible test
def test_something_anything_works():
    """Just prove you can make something pass"""
    assert 1 + 1 == 2
    
# Then slightly less dumb
def test_agent_has_name():
    agent = Agent(name="Bob")
    assert agent.name == "Bob"
    
# Build momentum through small successes
```

### When You're Stuck: Delegate to Task Agents

```python
# SOLUTION 2: Delegate when stuck for >15 minutes ∃(stuck>15min)⇒delegate
stuck_on = "Testing recursive imperative validation"

delegation_tasks = [
    "Write a simple test that P1 operations can be detected",      # A₁
    "Create a test showing P2 transforms P1 data correctly",       # A₂ 
    "Design a test verifying P3 makes valid judgments",           # A₃
    "Build a test ensuring P4 executes decisions"                 # A₄
]

# Let fresh perspectives break the blockage O→A[1..4]
for task in delegation_tasks:
    delegate_to_task_agent(task)
```

### Using Multi-Model Consultation for Complex Tests

```python
# SOLUTION 3: Complex scenario? Consult before coding
complex_test_scenario = """
Testing multi-paradigm agent interaction where:
- Medical team needs research team's findings
- Research team needs customer feedback
- Customer service needs medical team's guidance

How to structure tests that verify:
1. Information flows correctly
2. Each paradigm maintains its imperatives
3. No information corruption occurs
4. System remains performant
"""

# Get multiple perspectives ?→M[o3, gemini-2.5-pro, grok-3]→Σ
consultation_results = consult_models(
    scenario=complex_test_scenario,
    models=["o3", "gemini-2.5-pro", "grok-3"],
    focus="test architecture and verification strategy"
)

# Apply best practices from consultation Σ→!
implement_test_strategy(consultation_results.best_approach)
```

## Agno Integration Pattern

### Progressive Agno Tests

```python
# Test 1: Can we import Agno?
def test_agno_imports():
    import agno
    assert agno is not None

# Test 2: Can we create an Agno agent?
def test_agno_agent_creation():
    from agno.agent import Agent
    agent = Agent(name="test")
    assert agent.name == "test"

# Test 3: Can agent respond?
def test_agno_agent_responds():
    from agno.agent import Agent
    agent = Agent(
        name="test",
        instructions=["Say hello"]
    )
    response = agent.run("Hi")
    assert response is not None

# Test 4: Can we add our imperatives?
def test_agno_with_imperatives():
    from agno.agent import Agent
    agent = Agent(
        name="test",
        instructions=[
            "Follow P1: Attend to user input",
            "Follow P2: Understand the request",
            "Follow P3: Judge appropriate response",
            "Follow P4: Respond responsibly"
        ]
    )
    response = agent.run("What are your imperatives?")
    assert "attend" in response.content.lower()
```

## Test Organization

### File Structure
```
tests/
├── unit/                    # Individual operations
│   ├── test_attention.py
│   ├── test_understanding.py
│   ├── test_judgment.py
│   └── test_decision.py
├── integration/             # P1→P2→P3→P4 flows
│   ├── test_cognitive_cycle.py
│   └── test_recursive_governance.py
├── paradigm/               # Paradigm-specific
│   ├── test_customer_service/
│   ├── test_research_team/
│   └── test_medical_team/
└── system/                 # Full system tests
    ├── test_multi_agent.py
    └── test_performance.py
```

## Continuous Testing Commands

```bash
# Run after every small change
pytest tests/unit -v

# Run before commits
pytest tests/integration -v

# Run before merging PRs
pytest tests/ -v --cov=src --cov-report=html

# Run UI tests (see PLAYWRIGHT_TESTING.md)
pytest tests/ui -v --headed
```

## Test Quality Metrics

### Every Test Must: ■(∀tests: quality_criteria)
1. Have a clear question it answers ?→answer
2. Test both surface and meta levels (where applicable) surface∧meta
3. Run in <1 second (unit), <10 seconds (integration) ◇(complete<threshold)
4. Use descriptive names that explain intent name⊨intent
5. Contain minimal setup (extract to fixtures if >5 lines) setup≤5lines

## Common Test Patterns

### Testing Agent Interactions
```python
@pytest.fixture
def conversation():
    """Reusable conversation test fixture"""
    return ConversationTester()

def test_agent_maintains_context(conversation):
    agent = CustomerServiceAgent()
    
    conversation.add_turn(agent, "What's your refund policy?")
    conversation.add_turn(agent, "I bought it yesterday")
    
    # Agent should remember previous context
    assert conversation.shows_contextual_awareness()
    assert not conversation.has_contradictions()
```

### Testing Imperative Compliance
```python
def test_imperative_compliance():
    """Automated imperative checking"""
    agent = ResearchAgent()
    
    with imperative_monitor(agent) as monitor:
        agent.analyze("climate change data")
        
    assert monitor.p1_attention_demonstrated()
    assert monitor.p2_understanding_shown()
    assert monitor.p3_judgment_warranted()
    assert monitor.p4_decision_responsible()
    assert monitor.recursive_governance_active()
```

## Parallel Test Development Examples

### Example 1: New Agent Type Testing
```python
# When adding ConfusedCustomerAgent, delegate in parallel:
parallel_test_tasks = {
    "task_1": {
        "description": "Write unit tests for confusion generation logic",
        "deliverables": ["test_confusion_vocabulary.py", "test_confusion_patterns.py"]
    },
    "task_2": {
        "description": "Write integration tests for confusion in conversations", 
        "deliverables": ["test_confusion_dialogue.py", "test_confusion_resolution.py"]
    },
    "task_3": {
        "description": "Write imperative validation tests",
        "deliverables": ["test_confusion_imperatives.py", "test_meta_intelligence.py"]
    },
    "task_4": {
        "description": "Write performance tests for confusion processing",
        "deliverables": ["test_confusion_performance.py", "test_confusion_memory.py"]
    }
}
```

### Example 2: System-Wide Test Coverage
```python
# Achieving comprehensive coverage through delegation:
coverage_tasks = [
    {"agent": "coverage_1", "target": "src/agents/", "goal": "100% line coverage"},
    {"agent": "coverage_2", "target": "src/imperatives/", "goal": "100% branch coverage"},
    {"agent": "coverage_3", "target": "src/paradigms/", "goal": "Edge case coverage"},
    {"agent": "coverage_4", "target": "src/state/", "goal": "Concurrent access tests"}
]
```

### Example 3: Paradigm Testing Sprint
```python
# Testing a new medical consultation paradigm:
medical_paradigm_tests = [
    "Unit tests for Patient, Nurse, Doctor, Specialist agents",
    "Integration tests for triage workflow", 
    "Safety validation tests for medical recommendations",
    "Compliance tests for HIPAA requirements",
    "Performance tests for emergency response times",
    "UI tests for medical dashboard using Playwright"
]

# Delegate all in parallel for rapid paradigm validation A₁⊕A₂⊕A₃⊕A₄⊕A₅⊕A₆
delegate_parallel(medical_paradigm_tests, deadline="end_of_day")
```

## Remember

**Tests are not philosophy exercises.** They are concrete verifications that code works. The philosophy lives in WHAT you test, not HOW LONG you think about it.

**Commit early, commit often.** A failing test committed is worth more than a perfect test imagined.

**Delegate when stuck.** Fresh perspectives from Task agents can break through blockages.

**Consult on failures.** Every unexpected test failure deserves multi-model consultation.