# Symbolic Notation Reference

This document defines the canonical symbolic notation for the Transcendental Agent Orchestration Framework. All documentation and code comments should use these symbols consistently.

## 1. Agent Hierarchy

| Symbol | Meaning | Example |
|--------|---------|---------|
| [O] | Orchestrator agent | [O]_medical coordinates surgery planning |
| [A] | Task/Action agent | [A]_researcher gathers evidence |
| [H] | Human | [H] provides approval at gate |
| [M] | Model (LLM) | [M]_gemini provides consultation |
| A₁, A₂... | Numbered agents | A₁_questioner, A₂_answerer |
| O→A | Parent-child relation | [O]→[A₁], [O]→[A₂] |

### Examples:
```
[O]_support→{[A]_greeter, [A]_resolver, [A]_validator}
[H]→[O]→[A]→[H]  // Human delegates to orchestrator to agent back to human
```

## 2. Flow Operators

| Symbol | Meaning | Example |
|--------|---------|---------|
| → | Sequential flow/dependency | P1→P2→P3→P4 |
| ⊕ | Parallel execution | A₁⊕A₂⊕A₃ |
| ⊸ | Sequential dependency (must complete) | Build⊸Test⊸Deploy |
| ↻ | Recursive/cyclic flow | P1→P2→P3→P4→↻ |
| ⇄ | Bidirectional flow | Agent⇄Memory |
| ↪ | Branching/conditional flow | Test→(pass↪deploy, fail↪fix) |

### Examples:
```
// Parallel then sequential
(A₁⊕A₂⊕A₃)⊸A₄  // A₁,A₂,A₃ run in parallel, then A₄

// Recursive improvement cycle
Draft→Review→Revise→↻
```

## 3. Logic Operators

| Symbol | Meaning | Example |
|--------|---------|---------|
| ∧ | AND | attentive ∧ intelligent ∧ reasonable |
| ∨ | OR | success ∨ retry ∨ escalate |
| ¬ | NOT | ¬confused (not confused) |
| ⇒ | Implies/then | error⇒investigate |
| ⊨ | Satisfies/entails | Agent⊨P1∧P2∧P3∧P4 |
| ⊢ | Proves/derives | evidence⊢conclusion |

### Examples:
```
// Conditional logic
(test_failed ∧ ¬timeout) ⇒ retry
Agent⊨(attentive ∧ intelligent) ⇒ can_analyze
```

## 4. Quantifiers

| Symbol | Meaning | Example |
|--------|---------|---------|
| ∀ | For all | ∀agents: agent⊨imperatives |
| ∃ | There exists | ∃solution: viable(solution) |
| ∈ | Element of | agent ∈ TeamMembers |
| ∉ | Not element of | error ∉ ExpectedErrors |
| ⊆ | Subset of | RequiredSkills ⊆ AgentCapabilities |

### Examples:
```
∀a ∈ Agents: a.implements(P1→P2→P3→P4)
∃p ∈ Paradigms: p.solves(client_need)
```

## 5. States & Requirements

| Symbol | Meaning | Example |
|--------|---------|---------|
| ■ | Invariant (must always hold) | ■(imperatives_active) |
| ◇ | Eventually (must achieve) | ◇(task_complete) |
| ✓ | Complete/Success | Test: ✓ |
| ✗ | Failed/Error | Validation: ✗ |
| ? | Uncertain/Question | Status: ? |
| ⚠ | Warning/Attention needed | ⚠ Performance degraded |
| 🔒 | Locked/Immutable | 🔒 Core imperatives |

### Examples:
```
■(∀agents: conscious) // Always true: all agents conscious
◇(deployment) // Eventually achieve deployment
Status: [Build:✓, Test:✓, Deploy:?]
```

## 6. Delegation Patterns

| Symbol | Meaning | Example |
|--------|---------|---------|
| O→A[1..n] | One-to-many delegation | O→A[1..5] |
| O:X%→A:Y% | Work distribution | O:20%→A:80% |
| A₁⊕A₂⊕...⊕Aₙ | Parallel agents | A₁⊕A₂⊕A₃ |
| {A₁,A₂,...} | Agent set/team | Team={A_dev, A_test, A_doc} |
| A₁⊸A₂⊸A₃ | Sequential handoff | Design⊸Code⊸Review |

### Examples:
```
[O]_project→{
  [A]_architect: 30%,
  [A]_developer: 50%,
  [A]_tester: 20%
}

Research=(A₁⊕A₂⊕A₃)⊸A_synthesizer⊸A_validator
```

## 7. Consultation Patterns

| Symbol | Meaning | Example |
|--------|---------|---------|
| ?→M→Σ→! | Question→Models→Synthesis→Decision | ?→M[o3,gemini]→Σ→! |
| M[...] | Model set | M[gemini-pro, o3-mini, grok] |
| Σ | Synthesis/aggregation | responses→Σ→consensus |
| ! | Decision/action | analysis→! |
| ∃(X)⇒Y | If X exists, then Y | ∃(uncertainty)⇒consult |

### Examples:
```
// Multi-model consultation
?architecture→M[gemini-pro, o3, grok]→Σ→!design

// Conditional consultation
∃(complexity>threshold)⇒(?→M[expert_models]→Σ→!)
```

## 8. Development Flow

| Symbol | Meaning | Example |
|--------|---------|---------|
| [RED] | Failing test | [RED]→write_test |
| [GREEN] | Passing test | [GREEN]→all_tests_pass |
| [REFACTOR] | Code improvement | [REFACTOR]→optimize |
| [WIP] | Work in progress | [WIP]→feature_branch |
| [Review] | Under review | [Review]→PR#123 |
| [Merged] | Completed and merged | [Merged]→main |

### Examples:
```
// TDD Cycle
[RED]→[GREEN]→[REFACTOR]→↻

// Git Workflow
[WIP]→[Review]→[Approved]→[Merged]

// Full development flow
Idea→[RED]→Code→[GREEN]→[REFACTOR]→[Review]→[Merged]→Deploy
```

## 9. Composite Examples

### Complete Agent Interaction
```
[H]_request→[O]_coordinator→{
  [A₁]_analyzer⊕[A₂]_researcher⊕[A₃]_validator
}⊸[A₄]_synthesizer→[O]→[H]_response
```

### Paradigm Implementation with Imperatives
```
Paradigm_Medical:
  ■(∀a ∈ Agents: a⊨P1∧P2∧P3∧P4)
  [O]_surgery→{
    [A]_diagnostician: P1→P2,
    [A]_surgeon: P2→P3,
    [A]_monitor: P3→P4
  }
  ◇(successful_outcome ∧ safe_procedure)
```

### Error Handling Flow
```
try:
  Execute→✓
catch:
  Error→?→M[debug_models]→Σ→!fix→↻
finally:
  ■(imperatives_maintained)
```

## Usage Guidelines

1. **Consistency**: Always use the defined symbols. Don't create variants.

2. **Clarity**: When introducing complex notation, provide explanation:
   ```
   // O delegates to three parallel agents, then synthesizes
   O→(A₁⊕A₂⊕A₃)⊸Σ→!
   ```

3. **Context**: Use symbols that match the abstraction level:
   - High-level docs: Use basic flows (→, ⊕)
   - Technical specs: Use full notation including quantifiers
   - Code comments: Use simple, clear symbols

4. **Readability**: Break complex expressions into parts:
   ```
   // Instead of: ∀a∈A:a⊨(P1∧P2∧P3∧P4)⇒◇(success)
   
   // Write:
   ∀a ∈ Agents:
     a⊨(P1∧P2∧P3∧P4) ⇒
     ◇(success)
   ```

5. **ASCII Alternatives**: When Unicode isn't available:
   - → can be ->
   - ∀ can be forall
   - ∃ can be exists
   - ∧ can be AND
   - ∨ can be OR

---

This notation system ensures consistent, precise communication throughout the framework while maintaining readability and philosophical alignment with transcendental imperatives.