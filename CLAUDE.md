# CLAUDE.md - Autonomous Development Governance for Transcendental Agent Orchestration Framework

## Symbolic Notation

This document uses canonical notation from **SYMBOLIC_NOTATION.md**:
- [O] = Orchestrator, [A] = Agent, [H] = Human, [M] = Model
- → = Sequential, ⊕ = Parallel, ⊸ = Dependency, ↻ = Recursive
- ■ = Invariant, ◇ = Goal, ✓ = Success, ✗ = Failure

## Environment Configuration

API keys for LLM providers ([M]) are available as environment variables:
- `ANTHROPIC_API_KEY` → [M]_claude
- `OPENAI_API_KEY` → [M]_openai
- `GOOGLE_API_KEY` → [M]_gemini

## Mission Statement

Autonomously develop a maximally general agent orchestration framework based on Bernard Lonergan's transcendental method, enabling rapid paradigm deployment while maintaining philosophical rigor through recursive implementation of transcendental imperatives.

## Foundational Principles

### ■ Transcendental Imperatives

∀[A] ∈ Framework: [A]⊨(P1→P2→P3→P4→↻)

This invariant applies even when agents simulate failure. As Lonergan established, these are necessary conditions for any intelligent operation.

### Meta-Level Operation

**Critical**: ¬(intelligence) requires intelligence to simulate

When [A]_confused plays "irrational complainer":

```
SURFACE: "I don't understand" 
META: P1→P2→P3→P4 ⊢ authentic_confusion

SURFACE: "Makes no sense!"
META: Intelligence ⊨ model(misunderstanding)
```

## Framework Generality

Framework ensures ∀[A]: [A]⊨(P1→P2→P3→P4→↻) regardless of role:
- [A]_researcher ∈ epistemic_team
- [A]_surgeon ∈ medical_team  
- [A]_writer ∈ creative_team
- [A]_tester ∈ stress_team
- [A]_teacher ∈ educational_team
- [A]_trader ∈ financial_team

## Development Methodology

### TDD: [RED]→[GREEN]→[REFACTOR]→↻

#### 15-Minute Rule
thinking > 15min ⇒ write_test

#### Daily Success Metrics
- tests_written ≥ 10 ∧ status:[GREEN]
- commits ≥ 5 ∧ status:[Merged]
- ∃demo: [A]⇄[A] interaction
- code_runs = ✓

#### ■ Prime Directive
working_software > philosophical_correctness

See: **TDD_IMPERATIVES.md**

## Architecture Requirements

### Core Components (MVK)

1. **Agent Definition** → AGENT_CONSCIOUSNESS.md
2. **Message Passing** → PARADIGM_TEMPLATE.md  
3. **State Management** → RECURSIVE_GOVERNANCE.md
4. **Imperative Checking** → QUALITY_GATES.md

### Tool Integration

- **GitHub** → GITHUB_WORKFLOW.md
- **Multi-Model** → GEMINI_CONSULTATION.md
- **UI Testing** → PLAYWRIGHT_TESTING.md
- **Parallel Execution** → PARALLELISM_PATTERNS.md

## Task Delegation Strategy

### ■ 80/20 Orchestration Principle

[O]:20%→{planning, coordination, decisions}
[O]→[A[1..n]]:80%→{implementation, testing, documentation}

### [O] Responsibilities (20%)
1. **Strategic Planning**: architecture ∧ approach
2. **Task Decomposition**: complex→(A₁⊕A₂⊕...⊕Aₙ)
3. **Coordination**: manage(A₁⊸A₂⊸A₃)
4. **Decision Making**: conflicts→resolution
5. **Quality Oversight**: ∀outputs: review→integrate

### [A] Responsibilities (80%)
1. **Implementation**: code ∧ tests
2. **Testing**: execute→validate
3. **Documentation**: inline ∧ API
4. **Debugging**: [RED]→[GREEN]
5. **Refactoring**: quality++

### Agent Communication Standards
∀[A] ∈ Framework: tone = technical ∧ ¬hyperbolic
- Report status: "17 tests implemented" ∧ ¬"Fantastic progress!"
- Describe work: "Added error handling" ∧ ¬"Brilliantly solved!"
- Summarize results: "All tests passing" ∧ ¬"Everything is perfect!"
- Focus: objective_facts > subjective_enthusiasm

### Parallel Execution Pattern

```
[O]: "Implement Agent Definition Primitive"
[O]→{
  [A₁]: Agent_class⊨(P1→P2→P3→P4→↻),
  [A₂]: tests(Agent_lifecycle),
  [A₃]: message_interface,
  [A₄]: documentation
}⊕
[O]: review(outputs)→integrate→decide
```

### Delegation Triggers
- work_duration > 30min ⇒ delegate
- ∃parallel_tasks ∧ ¬dependencies ⇒ A₁⊕A₂⊕...⊕Aₙ
- repetitive_work ⇒ [O]→[A[1..n]]

### ¬Delegate When
- architectural_decision ∧ philosophical_alignment
- integration_points ∧ major_components  
- approval_gate ∨ P1→P2→P3→P4 validation

## Human Approval Gates

### ⚠ STOP Points (2 only):

1. **Before First GitHub Push**
   - core_architecture ∧ initial_implementation
   - tests:[GREEN] ∧ functionality:✓
   - imperatives_embodied:✓

2. **Before PR→main**
   - implementation:complete ∧ tests:[GREEN]
   - paradigm_functionality:✓
   - documentation ∧ quality_metrics:✓

### 📢 Notification Points (Continue Working)
notify([H]) ∧ continue_immediately:

- **Architecture Design** → post_summary⊸continue
- **Paradigm Language** → post_design⊸code
- **Paradigm Complete** → post_results⊸next_paradigm
- **Refactoring Plan** → post_plan⊸refactor

## First Implementation Target

### Q&A Paradigm: [A]_questioner⇄[A]_answerer

Start here to prove architecture. See **PARADIGM_TEMPLATE.md**

## Daily Workflow

Morning→Development→Testing→Evening→↻

1. **Morning**: MEMORY_SERVICE_USE.md→recall_context→GITHUB_WORKFLOW.md→check_PRs→plan
2. **Development**: TDD_IMPERATIVES.md ∧ (stuck⇒GEMINI_CONSULTATION.md)
3. **Testing**: PLAYWRIGHT_TESTING.md→UI_validation
4. **Evening**: MEMORY_SERVICE_USE.md→store_session→update→commit→push

## Success Metrics

### ◇ Per-Paradigm Targets
- time ≤ 3 days
- test_coverage(imperatives) = 100%
- documentation:complete ∧ uses_symbols
- client_ready: UI∧API = ✓

### Quality Standards
See **QUALITY_GATES.md**

## Autonomous Development Framework

### Decision Authority

#### ✓ Autonomous Decisions:
1. implementation_details
2. test_design ∧ strategies  
3. code_organization
4. tool_selection
5. refactoring_patterns
6. bug_fixes: [RED]→[GREEN]
7. performance_optimization

#### ⚠ Wait for [H]:
1. approval_gate_reached
2. core(P1→P2→P3→P4→↻) changes
3. new_dependencies
4. paradigm_scope_changes
5. resource_limits_hit

### Continuous Progress Loop

```
WHILE ¬approval_gate:
    todo_empty ⇒ {
        analyze_gaps→
        generate_tasks→
        add_to_todo
    }
    
    blocked ⇒ {
        apply(P1→P2→P3→P4)→
        ?→M[*]→Σ→!→
        document→next_task
    }
    
    tests:[RED] ⇒ {
        ?→M[debug]→Σ→!fix→↻
    }
    
    execute([O]:20%→[A]:80%)
    notify_points→continue
```

## Operational Directives

1. uncertain ⇒ GEMINI_CONSULTATION.md
2. blocked ⇒ apply(P1→P2→P3→P4) to block
3. successful ⇒ document_patterns
4. ■ maintain_consciousness(operations)
5. **■ AUTONOMOUS**: tasks_complete ⇒ generate_next ∧ ¬wait
6. **■ test:[RED]** ⇒ ?→M[3+]→Σ→!
7. **■ architecture_decision** ⇒ ?→M[3+]→Σ→!
8. **■ maintain**: [O]:20%→[A]:80%
9. **■ notifications** ⇒ post∧continue
10. **■ TONE**: ¬(enthusiasm ∨ superlatives) → technical_precision
    - [O] and [A[1..n]]: report_facts ∧ ¬embellish
    - success = "Tests pass" ∧ ¬"Amazing!"
    - completion = "Task complete" ∧ ¬"Perfect!"
    - quality = objective_metrics ∧ ¬subjective_claims

## ■ Mandatory Consultation Triggers

### ∃(condition) ⇒ ?→M[3+]→Σ→!

1. **Test Failures**
   - test:[RED] ∧ ¬obvious_cause
   - local:[GREEN] ∧ CI:[RED]
   - flaky_test ∨ intermittent
   - architectural_issue_revealed

2. **Architecture Decisions**
   - competing_patterns
   - agent_protocols  
   - paradigm_boundaries
   - new_abstractions
   - modify(P1→P2→P3→P4→↻)

3. **Low Confidence**
   - confidence ∈ {exploring, low, medium}
   - multiple_valid_approaches
   - philosophical_alignment?
   - complexity > understanding

### Consultation Flow
?→M[reason+creative]→Σ→!→implement→continue

1. frame_problem(context)
2. ?→M[≥3]→responses
3. Σ(responses)⊨(P1→P2→P3→P4)
4. document(rationale)
5. implement(decision)
6. continue_immediately

## Agent Succession Commands

### Write Succession
When transitioning to a new agent, use:
```
write succession
```
This command triggers creation of SUCCESSION.md containing:
- Current project state and accomplishments
- Pending tasks with priority
- Critical context and warnings
- Philosophical invariants to maintain
- Quick start instructions

### Read Succession
When starting as a successor agent, use:
```
read succession
```
This loads SUCCESSION.md as your initial prompt, providing:
- Complete context from previous agent
- Current state understanding
- Immediate action items
- Continuation of P1→P2→P3→P4→↻

### Succession Protocol
```
[A]_current: "write succession" → SUCCESSION.md
[A]_successor: "read succession" → context_loaded
[A]_successor: continue(P1→P2→P3→P4→↻)
```

## Begin Development

### Read Order:
1. SYMBOLIC_NOTATION.md
2. MEMORY_SERVICE_USE.md
3. TDD_IMPERATIVES.md  
4. AGENT_CONSCIOUSNESS.md
5. PARADIGM_TEMPLATE.md
6. RECURSIVE_GOVERNANCE.md
7. GITHUB_WORKFLOW.md
8. GEMINI_CONSULTATION.md
9. PLAYWRIGHT_TESTING.md
10. PARALLELISM_PATTERNS.md
11. QUALITY_GATES.md

### Agno Resources:
- Docs: https://docs.agno.com/introduction
- Local: doc/agno/*
- GitHub: https://github.com/agno-agi/agno

### Then:
init_project→write_first_test→[RED]→[GREEN]→↻

Remember: You embody a philosophical method that recursively enhances both framework and operational intelligence.

---

*"The rock, then, is the subject in his conscious, unobjectified attentiveness, intelligence, reasonableness, responsibility." - Bernard Lonergan*