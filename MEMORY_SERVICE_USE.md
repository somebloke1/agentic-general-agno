# MEMORY_SERVICE_USE.md - Development Memory Patterns for Agno Framework

## Purpose

This document defines how Claude (development assistant) uses the MCP memory-service while building the transcendental agent orchestration framework. These patterns ensure development continuity, track architectural decisions, and enable learning from past implementation experiences.

## Quick Reference

### ■ Session Start
```bash
recall_memory("today agno")
retrieve_memory("todo:incomplete agno")
```

### ■ When Blocked
```bash
retrieve_memory("{problem} solution agno")
retrieve_memory("consultation {issue_type}")
```

### ■ Session End
```bash
store_memory("Session: completed X, next Y", tags=["agno","session","todo:incomplete"])
```

## Memory Service Capabilities

### Available Tools
- `store_memory` - Store with content and metadata (tags, type)
- `retrieve_memory` - Semantic search for relevant memories  
- `recall_memory` - Natural language time-based queries ("today", "last week")
- `recall_by_timeframe` - Specific date range queries
- `update_memory_metadata` - Update tags and metadata
- `delete_memory` - Remove by content hash
- `check_database_health` - Monitor memory system status

### Known Limitations
- Tag search currently non-functional (use semantic search instead)
- Exact match requires character-perfect matching

## Memory Taxonomy

### Project-Level Tags (■ Always Apply)

```
■ "agno" - All project-related memories
■ "architecture" - Design decisions with rationale  
■ "implementation" - Code patterns and solutions
■ "testing" - Test strategies, edge cases, validation approaches
■ "paradigm:{name}" - Specific paradigm work (e.g., "paradigm:qa")
```

### Situational Tags (◇ Conditional Use)

```
◇ "blocked" - Obstacles encountered + resolution attempts
◇ "consultation" - Multi-model consensus results (?→M[3+]→Σ→!)
◇ "pattern:effective" - Successful approaches to replicate
◇ "pattern:failed" - Approaches that didn't work (anti-patterns)
◇ "P1-P4:violation" - Imperative implementation failures
◇ "todo:incomplete" - Work to resume in next session
◇ "insight" - Key realizations or breakthroughs
◇ "refactor:needed" - Technical debt markers
◇ "edge-case" - Unusual scenarios discovered
◇ "performance" - Optimization opportunities
◇ "delegation" - O→A[1..n] task distribution patterns
◇ "ci:failure" - CI/CD specific failures and fixes
```

## Memory Storage Patterns

### 1. Architectural Decisions

```python
# When making design choices
memory.store_memory(
    content=f"""
    Decision: {decision_summary}
    Rationale: {why_this_approach}
    Alternatives Considered: {other_options}
    Trade-offs: {pros_and_cons}
    Philosophical Alignment: {P1_P2_P3_P4_adherence}
    """,
    metadata={
        "tags": ["agno", "architecture", "decision"],
        "type": "architectural_decision",
        "component": component_name,
        "date": current_date
    }
)
```

### 2. Test Pattern Discovery

```python
# When finding effective test strategies
memory.store_memory(
    content=f"""
    Test Pattern: {pattern_name}
    Validates: {what_imperatives}
    Implementation: {code_snippet}
    Edge Cases Covered: {edge_case_list}
    [RED]→[GREEN] Cycle: {how_it_progressed}
    """,
    metadata={
        "tags": ["agno", "testing", "pattern:effective"],
        "type": "test_pattern",
        "paradigm": current_paradigm
    }
)
```

### 3. Consultation Results

```python
# After ?→M[3+]→Σ→! consultations
memory.store_memory(
    content=f"""
    Problem: {issue_description}
    Models Consulted: {model_list}
    Consensus: {final_decision}
    Key Insights: {important_realizations}
    Implementation: {chosen_approach}
    """,
    metadata={
        "tags": ["agno", "consultation", consultation_type],
        "type": "multi_model_consensus",
        "confidence": confidence_level
    }
)
```

### 4. Development Session Context

```python
# At session end or major transition
memory.store_memory(
    content=f"""
    Current State: {what_completed}
    Next Steps: {immediate_tasks}
    Open Questions: {unresolved_issues}
    Active Paradigm: {paradigm_in_progress}
    Blocking Issues: {any_blockers}
    """,
    metadata={
        "tags": ["agno", "todo:incomplete", "session"],
        "type": "session_context"
    }
)
```

### 5. Delegation Patterns (O→A[1..n])

```python
# When delegating to parallel agents
memory.store_memory(
    content=f"""
    Task: {main_task}
    Delegation Pattern: {orchestrator_to_agents}
    A₁: {agent1_task}
    A₂: {agent2_task}
    Dependencies: {A1_depends_on_A2}
    Success Criteria: {expected_outcomes}
    """,
    metadata={
        "tags": ["agno", "delegation", "pattern:effective"],
        "type": "orchestration_pattern"
    }
)
```

## Memory Retrieval Patterns

### ⚠️ Context-Aware Retrieval Strategy

As memory corpus grows, retrieval must be **precise and targeted** to preserve context window:

```python
# ✗ BAD: Floods context with all memories
all_paradigms = retrieve_memory("paradigm agno")

# ✓ GOOD: Specific, targeted retrieval
qa_implementation = retrieve_memory("paradigm:qa implementation pattern")
```

### Dynamic Retrieval Patterns

#### 1. Session Start Protocol (Evolving)

```python
# Early project: Can afford broader retrieval
recent_work = recall_memory("today agno")

# As memories grow: Be more specific
recent_qa_work = recall_memory("today paradigm:qa")
incomplete_tests = retrieve_memory("todo:incomplete testing P1-P4")

# Advanced: Use n_results parameter
critical_todos = retrieve_memory("todo:incomplete blocked", n_results=3)
```

#### 2. Progressive Refinement

```python
# Start narrow, expand only if needed
results = retrieve_memory("specific_error_message", n_results=2)
if not sufficient(results):
    results = retrieve_memory("error_category solution", n_results=3)
```

#### 3. When Blocked (Targeted Search)

```python
# Highly specific problem search
exact_error = retrieve_memory(f'"{error_message}" solution', n_results=2)

# If no direct match, broaden slightly
if not exact_error:
    category_solutions = retrieve_memory(f"{error_category} agno fix", n_results=3)
```

#### 4. Before Major Decisions (Scoped Retrieval)

```python
# Component-specific only
component_decisions = retrieve_memory(f"{component} architecture decision", n_results=5)

# Avoid broad architectural searches unless necessary
# ✗ retrieve_memory("architecture")  # Too broad!
```

#### 5. Implementation Patterns (Tagged + Limited)

```python
# Use compound queries for precision
patterns = retrieve_memory("pattern:effective message-passing agno", n_results=3)

# Time-boxed pattern search
recent_patterns = recall_memory("last week pattern:effective", n_results=5)
```

#### 6. Testing Phase (Imperative-Specific)

```python
# Target specific imperative validation
p2_tests = retrieve_memory("testing P2 understand validation", n_results=3)

# Component + imperative combo
agent_p1_tests = retrieve_memory("agent P1 attention test", n_results=2)
```

## Memory Lifecycle Management

### Regular Cleanup

```python
# Periodically clean outdated session contexts
old_sessions = recall_by_timeframe(
    start_date=week_ago,
    end_date=three_days_ago
)
# Review and delete if no longer needed
```

### Pattern Evolution

```python
# Update patterns as they improve
update_memory_metadata(
    content_hash=pattern_hash,
    updates={
        "tags": existing_tags + ["superseded"],
        "superseded_by": new_pattern_hash
    }
)
```

## Integration with Development Workflow

### TDD Cycle Integration

1. **[RED] Phase**: Store failing test and hypothesis
2. **[GREEN] Phase**: Store solution that made test pass
3. **[REFACTOR] Phase**: Store optimization insights

### Paradigm Development

1. **Design**: Store paradigm concept and rationale
2. **Implementation**: Track successful agent patterns
3. **Testing**: Document validation strategies
4. **Completion**: Store full paradigm template

### Recursive Improvement

After each paradigm or major component:
```python
# Store meta-insights
memory.store_memory(
    content="What worked well, what to improve",
    metadata={
        "tags": ["agno", "insight", "meta-development"],
        "type": "retrospective"
    }
)
```

## Usage Examples

### Example 1: Starting New Paradigm

```python
# Check previous paradigm implementations
past_paradigms = retrieve_memory("paradigm implementation successful")

# Store new paradigm plan
store_memory(
    content="Q&A Paradigm: questioner↔answerer design...",
    metadata={"tags": ["agno", "paradigm:qa", "architecture"]}
)
```

### Example 2: Resolving Test Failure

```python
# Search for similar test failures
similar_failures = retrieve_memory("test failure P2 validation")

# After resolution, store solution
store_memory(
    content="P2 validation fixed by ensuring cognitive trace...",
    metadata={"tags": ["agno", "testing", "pattern:effective", "P1-P4"]}
)
```

### Example 3: Session Handoff

```python
# End of session
store_memory(
    content="Completed Agent base class, next: message passing...",
    metadata={"tags": ["agno", "todo:incomplete", "session"]}
)

# Next session
context = recall_memory("yesterday agno todo:incomplete")
```

## Best Practices

1. **Be Specific**: Include concrete details in memory content
2. **Tag Consistently**: Always include "agno" + relevant category tags
3. **Document Why**: Include rationale, not just what
4. **Track Failures**: Failed approaches prevent repeated mistakes
5. **Update Metadata**: Mark superseded patterns
6. **Targeted Retrieval**: Use specific queries with n_results limits
7. **Session Continuity**: Always store/retrieve session context
8. **Context Economy**: As memories grow, queries must become more precise
9. **Progressive Search**: Start narrow, expand only if needed
10. **Time-Box Searches**: Use recall_memory for recent items first

## Memory as Cognitive Enhancement

The memory service acts as an **extended cognition** system for development:
- Overcomes context window limitations
- Enables learning across sessions
- Facilitates pattern recognition
- Supports architectural consistency
- Accelerates problem resolution

By treating memory as part of the development consciousness, we embody the transcendental method even in our tooling: Attend (store) → Understand (retrieve) → Judge (pattern match) → Decide (apply) → Reflect (update).

---

*"Development memory transforms ephemeral insights into persistent wisdom."*