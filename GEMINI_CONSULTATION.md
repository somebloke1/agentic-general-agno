# GEMINI_CONSULTATION.md - Multi-Model Consultation via zen-mcp

## Consultation Pattern
?→M[models]→Σ→!

## Overview

When Claude Code encounters complex problems or needs alternative perspectives, use zen-mcp-server to consult other models while maintaining transcendental method fidelity.

## Setup

```bash
# Ensure zen-mcp-server is configured
claude mcp add zen-server zen-mcp-server

# Set environment variables
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key" 
export XAI_API_KEY="your-key"  # For Grok
```

## When to Consult Other Models

### MANDATORY Consultation Triggers ■(MUST CONSULT)

These are REQUIREMENTS, not suggestions. You MUST consult when:

1. **EVERY unexpected test failure** - Not after 3 attempts, EVERY SINGLE FAILURE ∃(failure)⇒consult
2. **EVERY major architectural decision** - Before committing to any design ∃(architecture)⇒consult
3. **EVERY time confidence is low/medium** - If unsure, consult immediately ∃(confidence<high)⇒consult
4. **Before FIRST push to GitHub** - Always validate before initial push ∃(first_push)⇒consult
5. **Before EVERY PR to main** - No exceptions, consultation required ∃(PR→main)⇒consult

### Optional Consultation Triggers (MAY CONSULT)

Additional situations where consultation is beneficial:

1. **Stuck on implementation** (>30 min no progress) ∃(stuck>30min)⇒?
2. **Performance optimization** questions performance?⇒consult
3. **Multiple valid approaches exist** ∃(approaches>1)⇒?
4. **Current approach feels wrong** ¬(feels_right)⇒?
5. **Need domain expertise** ∃(domain_gap)⇒?
6. **Want creative alternatives** creativity?⇒consult

## Consultation Patterns

### Pattern 1: Quick Validation ?→M[gemini-pro]→!

```python
# When unsure about approach
"""
@zen-chat model=gemini-pro

I'm implementing {specific feature} using {approach}.
Current code: {code snippet}
Concern: {specific doubt}

Is this approach sound? What am I missing?
"""
```

### Pattern 2: Architecture Review ?→M[o3]→Σ→!

```python
"""
@zen-chat model=o3

Reviewing agent architecture for paradigm: {paradigm_name}

Current structure:
- Agents: {list agents and roles}
- Interactions: {describe pattern}
- Imperatives: {how P1→P2→P3→P4→R is maintained}

Questions:
1. Will this scale to {N} concurrent agents?
2. How to maintain imperative fidelity under load?
3. Better patterns for this use case?

Context: Building for enterprise client with {requirements}
"""
```

### Pattern 3: Debug Assistance ✗→?→M[o3]→!

```python
"""
@zen-debug model=o3

Complex bug in recursive governance:
- Symptom: {what's happening}
- Expected: {what should happen}
- Code: {relevant code}
- Tests failing: {test names}

Tried:
1. {attempt 1}
2. {attempt 2}

Need fresh perspective on root cause.
"""
```

### Pattern 4: Creative Solutions ?→M[gemini-pro]→Σ→!

```python
"""
@zen-chat model=gemini-pro

Building {paradigm type} paradigm.
Standard approach would be {conventional solution}.

Looking for creative alternatives that:
- Maintain P1→P2→P3→P4→R integrity
- Reduce complexity
- Improve performance

What unconventional approaches might work?
"""
```

### Pattern 5: Task Delegation Planning O→?→M[o3]→A[1..n]

```python
"""
@zen-chat model=o3

Planning to delegate multiple tasks for {goal}.

Available tools:
- Task for parallel operations
- Agent tools for specialized work
- File operations for batch processing

Current tasks to delegate:
1. {task 1 description}
2. {task 2 description}
3. {task 3 description}

How should I structure these delegations for:
- Maximum parallelism
- Dependency management
- Error handling
- Result synthesis

Provide specific delegation strategy.
"""
```

## Enforcement Mechanisms

### Tracking Consultations

You MUST maintain a consultation log in your working memory:

```python
# Consultation tracking template
consultation_log = {
    "mandatory_consultations": {
        "test_failures": [],
        "architectural_decisions": [],
        "low_confidence_moments": [],
        "pre_github_push": None,
        "pre_pr_merge": None
    },
    "optional_consultations": [],
    "skipped_mandatory": []  # THIS SHOULD ALWAYS BE EMPTY
}

# Example entry
consultation_log["mandatory_consultations"]["test_failures"].append({
    "timestamp": "2024-01-15 10:30",
    "test": "test_agent_imperatives",
    "failure": "P2 validation not working",
    "consulted": "o3",
    "resolution": "Fixed imperative checking logic"
})
```

### What Happens If Consultation Is Skipped

**IMMEDIATE CONSEQUENCES:**
1. Work is considered incomplete ✗
2. Code cannot be pushed to GitHub 🔒
3. PR will be rejected in review ✗
4. You must backtrack and consult properly ↻
5. Document why consultation was skipped ■(must_document)

**PROPER RECOVERY:**
```python
# If you realize you skipped mandatory consultation
"""
@zen-chat model=o3

MISSED MANDATORY CONSULTATION:
- Trigger: Test failure on {test_name}
- Action taken without consulting: {what you did}
- Current state: {describe current situation}

Please review my approach and suggest corrections.
"""
```

### Examples of Proper Consultation Workflow

#### Example 1: Test Failure Consultation
```python
# Test fails unexpectedly
"""
❌ WRONG APPROACH:
1. Test fails
2. Try to fix it yourself
3. Push broken code

✅ CORRECT APPROACH:
1. Test fails
2. IMMEDIATELY consult:
"""

@zen-debug model=o3

Unexpected test failure:
- Test: test_recursive_governance_cycle
- Expected: Agent maintains P1→P2→P3→P4 cycle
- Actual: P3 judgment phase skipped
- Error: AssertionError at line 145

Code under test: {paste relevant code}
Test code: {paste test}

What's causing P3 to be skipped?
"""
```

#### Example 2: Architecture Decision Consultation
```python
# Before implementing major component
"""
@zen-consensus models=[gemini-pro,o3,grok-3]

MANDATORY ARCHITECTURE CONSULTATION:

About to implement message routing system.

Option A: Central message broker
- All agents register with broker
- Broker handles routing logic
- Single point of control

Option B: Peer-to-peer messaging
- Agents discover each other
- Direct communication
- Distributed control

Requirements:
- Must scale to 100+ agents
- Must maintain imperative tracking
- Must support paradigm flexibility

Which architecture better serves our goals?
"""
```

#### Example 3: Low Confidence Consultation
```python
# When confidence drops
"""
@zen-chat model=gemini-pro

MANDATORY LOW CONFIDENCE CONSULTATION:

Working on: Agent state persistence
Confidence level: LOW

What I'm unsure about:
- Should states be stored in memory or disk?
- How to handle state versioning?
- Recovery from corrupted states?

Current partial implementation: {code}

Guide me toward a robust solution.
"""
```

## Multi-Model Consensus

### For Critical Decisions ?→M[gemini-pro,o3,grok-3]→Σ→!

```python
"""
@zen-consensus models=[gemini-pro,o3,grok-3]

Critical decision: {describe decision}

Option A: {description}
- Pros: {list}
- Cons: {list}

Option B: {description}  
- Pros: {list}
- Cons: {list}

Context: {relevant constraints}
Imperatives: Must maintain {which aspects of P1→P2→P3→P4→R}

What's the best path forward and why?
"""
```

### Response Processing

```python
# After receiving consensus M[...]→Σ
def process_model_consultation(responses):
    """Extract actionable insights from model responses"""
    
    consensus_points = find_agreements(responses)    # Σ
    unique_insights = find_unique_perspectives(responses)
    warnings = extract_warnings(responses)    # ⚠
    
    # Apply transcendental method to consultation P1→P2→P3→P4→R
    return {
        "P1_attended": "What all models noticed",
        "P2_insights": "Genuine understanding across models",  
        "P3_judgment": "Evidence-based consensus",
        "P4_decision": "Responsible path forward",    # !
        "R_reflection": "How this improves our approach"    # ↻
    }
```

## Specific Model Strengths

### Gemini Pro
Best for:
- System design questions
- Performance optimization
- Creative problem solving
- Pattern recognition

Example:
```python
"""
@zen-chat model=gemini-pro

Optimizing agent communication pattern.
Current: Sequential message passing taking {X}ms per cycle
Need: <{Y}ms for real-time feel

What patterns could maintain imperative fidelity while improving speed?
"""
```

### O3
Best for:
- Complex debugging
- Logical analysis
- Code generation
- Test design

Example:
```python
"""
@zen-chat model=o3

Need comprehensive test suite for {component}.
Must verify both surface behavior and meta-level imperatives.

Generate test cases that:
1. Verify P1→P2→P3→P4 cycle
2. Check failure modes
3. Validate recursive governance
"""
```

### Grok
Best for:
- Alternative perspectives
- Challenging assumptions
- Edge case identification
- Unconventional solutions

Example:
```python
"""
@zen-chat model=grok-3

Current assumption: Agents must process P1→P2→P3→P4 sequentially.

Challenge this. What if they could parallelize while maintaining fidelity?
What breaks? What improves?
"""
```

## Integration Workflows

### Workflow 1: Progressive Enhancement

```python
# Start simple
initial_solution = implement_basic_version()    # [A]

# Consult on improvements ?→M[gemini-pro]
enhanced = consult_model("gemini-pro", 
    f"How to enhance {initial_solution} for {specific_goal}")

# Validate enhancements M[o3]→✓ ∨ ✗
validated = consult_model("o3",
    f"Verify this enhancement maintains imperatives: {enhanced}")

# Apply if validated ✓⇒apply
if validated.maintains_imperatives:
    apply_enhancement(enhanced)
```

### Workflow 2: Stuck-Breaking

```python
# When stuck for >30 minutes ∃(stuck>30min)⇒consult
if time_stuck > 30:
    # Step 1: Explain to Gemini ?→M[gemini-pro]
    explanation = explain_to_model("gemini-pro", current_problem)
    
    # Step 2: Get fresh approach M→alternative
    new_approach = get_alternative_approach(explanation)
    
    # Step 3: Validate with O3 alternative→M[o3]→✓ ∨ ✗
    validation = validate_approach("o3", new_approach)
    
    # Step 4: Implement if sound ✓⇒implement
    if validation.is_sound:
        implement_fresh_approach(new_approach)
```

### Workflow 3: Paradigm Validation

```python
# Before finalizing paradigm
def validate_paradigm_design(paradigm_config):
    # Get perspectives ?→M[gemini-pro,o3,grok-3]
    gemini_view = consult_on_structure("gemini-pro", paradigm_config)
    o3_view = consult_on_logic("o3", paradigm_config)
    grok_view = consult_on_edge_cases("grok-3", paradigm_config)
    
    # Synthesize feedback M[...]→Σ
    synthesis = synthesize_feedback([gemini_view, o3_view, grok_view])
    
    # Apply improvements Σ→!
    return apply_validated_improvements(paradigm_config, synthesis)
```

## Best Practices

### DO:
- Maintain context between consultations
- Apply P1→P2→P3→P4→R to model responses
- Document why you consulted and what you learned
- Test suggestions before wholesale adoption
- Thank models (they appreciate it)

### DON'T:
- Blindly follow model suggestions
- Consult on every minor decision  
- Forget your transcendental method
- Let models override core imperatives
- Create consultation loops

## Example Consultation Session

```python
"""
@zen-chat model=gemini-pro

Context: Building customer service paradigm
Stuck on: How to make angry customer agent escalate realistically

Current approach:
- Track frustration_level integer
- Increment on triggers
- Threshold-based responses

Problems:
- Feels mechanical
- Not responding to actual conversation
- Escalation seems random

How can I make this more authentically intelligent while maintaining P1→P2→P3→P4→R?
"""

# Gemini suggests conversation pattern analysis

"""
@zen-chat model=o3

Following up on Gemini's suggestion about pattern analysis.
How to implement conversation pattern tracking that:
1. Maintains cognitive trace
2. Identifies repetition/dismissal
3. Guides realistic escalation
4. Stays simple to test

Code structure suggestions?
"""

# O3 provides implementation

"""
@zen-reflect

Consultation synthesis:
- P1: Both models attended to mechanical vs authentic behavior
- P2: Understood need for conversation-aware escalation
- P3: Judged pattern analysis as sound approach
- P4: Decided on trackable implementation
- R: This improves paradigm realism significantly

Implementing with tests for both levels.
"""
```

## Emergency Consultation

### When to Declare Emergency

An emergency consultation is MANDATORY when:
1. Multiple systems failing simultaneously ∃(failures>1)⇒emergency
2. Client demo/deadline at risk ∃(deadline_risk)⇒emergency
3. Data loss or corruption potential ∃(data_risk)⇒emergency
4. Complete blocker with no workaround ∃(blocker ∧ ¬workaround)⇒emergency
5. Security vulnerability discovered ∃(security_issue)⇒emergency

### Emergency Consultation Protocol

```python
"""
@zen-thinkdeep model=gemini-pro thinking_mode=max

🚨 EMERGENCY CONSULTATION REQUIRED 🚨

Severity: CRITICAL
Blocking: {what is blocked}
Deadline: {time remaining}

SYMPTOMS:
- {list all symptoms with timestamps}
- {error logs with full stack traces}
- {when it started}
- {what changed before failure}

ATTEMPTED FIXES:
1. {fix attempt 1} → {result}
2. {fix attempt 2} → {result}
3. {fix attempt 3} → {result}

SYSTEM STATE:
- Last known good state: {description}
- Current state: {description}
- Critical data at risk: {yes/no, what data}

HELP NEEDED:
1. Root cause analysis
2. Immediate mitigation
3. Proper fix approach
4. Rollback strategy if needed

This is blocking {specific impact}.
"""
```

### Post-Emergency Protocol

After emergency resolution:
1. Document root cause
2. Add regression tests
3. Update runbooks
4. Consult on prevention:

```python
"""
@zen-chat model=o3

POST-EMERGENCY REVIEW:

Incident: {description}
Root cause: {what we found}
Fix applied: {what we did}

How can we prevent this class of failure?
What monitoring should we add?
"""
```

## Remember

### MANDATORY REQUIREMENTS: ■(∀situations: consult)
- **MUST consult on EVERY test failure** - No exceptions ✗⇒consult
- **MUST consult on EVERY architectural decision** - Before implementation architecture⇒consult
- **MUST consult when confidence is low/medium** - Immediately confidence<high⇒consult
- **MUST consult before first GitHub push** - Always first_push⇒consult
- **MUST consult before every PR to main** - Required PR→main⇒consult

### BEST PRACTICES:
- Consultation enhances, doesn't replace your judgment
- Always apply transcendental method to responses
- Document insights for future reference
- Track all consultations in your log
- Stay focused on shipping working code

### ENFORCEMENT:
- Skipped mandatory consultations = incomplete work ✗
- No push without consultation = enforced policy 🔒
- PR rejection for missing consultations = guaranteed ✗→[PR]
- This is not optional, it's required process ■(mandatory)