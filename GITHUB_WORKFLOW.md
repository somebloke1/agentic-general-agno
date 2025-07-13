# GITHUB_WORKFLOW.md - GitHub Integration and Workflow

## Workflow Flow
[WIP]→[Review]→[Approved]→[Merged]

## Repository Structure

```
transcendental-agents/
├── .github/
│   ├── workflows/          # CI/CD pipelines
│   ├── ISSUE_TEMPLATE/     # Issue templates
│   └── pull_request_template.md
├── src/
│   ├── core/               # Framework core
│   ├── paradigms/          # Paradigm implementations
│   └── agents/             # Agent base classes
├── tests/
│   ├── unit/              
│   ├── integration/
│   └── paradigm/
├── docs/
│   ├── CLAUDE.md          # This governance doc
│   ├── paradigms/         # Paradigm documentation
│   └── api/               # API documentation
├── scripts/
│   ├── setup.py
│   └── validate_paradigm.py
└── requirements.txt
```

## Branching Strategy

### Branch Types

```bash
main                    # Production-ready code (protected) 🔒
├── develop            # Integration branch
├── paradigm/*         # Paradigm development [A]
├── feature/*          # Feature development [A]
├── fix/*              # Bug fixes [A]
└── experiment/*       # Experimental work [A]
```

Note: Only `main` is protected 🔒 and requires PR approval. All other branches can be pushed to directly by [A] agents.

### Branch Naming

```bash
# Paradigm branches
paradigm/customer-service-v1
paradigm/research-team-v2
paradigm/medical-triage-beta

# Feature branches
feature/cognitive-trace-enhancement
feature/parallel-agent-execution
feature/mcp-zen-integration

# Fix branches
fix/governance-cycle-timeout
fix/agent-memory-leak
```

## Commit Standards

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Examples

```bash
# Good commits
git commit -m "feat(agents): Add cognitive trace to base agent

- Implement operation recording for attention
- Add insight tracking for understanding  
- Tests verify both surface and meta behavior

Closes #42"

git commit -m "fix(paradigm/customer): Improve anger escalation logic

- Agent now evaluates repetition before escalating
- Added test for de-escalation when issue resolved
- Maintains authentic behavior patterns"

git commit -m "refactor(core): Simplify governance cycle

- Governance now uses single reflection prompt
- Reduced complexity while maintaining effectiveness
- All paradigm tests still passing"
```

## Pull Request Process

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] New paradigm
- [ ] Feature enhancement
- [ ] Bug fix
- [ ] Refactoring
- [ ] Documentation

## Paradigm Impact
- [ ] No impact on existing paradigms
- [ ] Updates existing paradigm: ___
- [ ] Adds new paradigm: ___

## Testing
- [ ] Unit tests pass ✓
- [ ] Integration tests pass ✓
- [ ] Paradigm tests pass ✓
- [ ] New tests added ✓

## Checklist
- [ ] Code follows project style ✓
- [ ] Self-review completed ✓
- [ ] Documentation updated ✓
- [ ] No breaking changes ✓
```

### PR Workflow

```bash
# 1. Create feature branch [A]
git checkout -b feature/enhanced-reflection

# 2. Make changes with frequent commits [A]
git add -A
git commit -m "feat(governance): Add peer review capability"

# 3. Push and create PR [A]→[Review]
git push origin feature/enhanced-reflection
gh pr create --title "Add peer review to governance cycle" \
             --body "$(cat .github/pull_request_template.md)"

# 4. Address review feedback [A]
git commit -m "fix(governance): Address PR feedback on timing"

# 5. Merge via GitHub [Review]→[Approved]→[Merged]
# PRs to main require at least one approval from [H] or [O]
```

## GitHub Actions CI/CD

### CI/CD Flow
[Push]→[Build]→[Test]→[Validate]→✓ ∨ ✗

### Main Test Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: '*'  # Run on push to any branch
  pull_request:
    branches: [main]  # Run on PRs to main only

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run unit tests    # [Test]→✓ ∨ ✗
      run: pytest tests/unit -v --cov=src
    
    - name: Run integration tests    # [Test]→✓ ∨ ✗
      run: pytest tests/integration -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Paradigm Validation Pipeline

```yaml
# .github/workflows/paradigm-validation.yml
name: Paradigm Validation

on:
  pull_request:
    paths:
      - 'src/paradigms/**'
      - 'tests/paradigm/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Detect changed paradigms
      id: paradigms
      run: |
        echo "::set-output name=list::$(git diff --name-only origin/main | grep paradigms/ | cut -d'/' -f3 | sort -u)"
    
    - name: Validate each paradigm    # ■(∀paradigms: valid)
      run: |
        for paradigm in ${{ steps.paradigms.outputs.list }}; do
          echo "Validating paradigm: $paradigm"
          python scripts/validate_paradigm.py $paradigm    # [Validate]→✓ ∨ ✗
        done
    
    - name: Run paradigm tests    # ■(∀tests: pass)
      run: |
        for paradigm in ${{ steps.paradigms.outputs.list }}; do
          pytest tests/paradigm/test_${paradigm}.py -v    # [Test]→✓ ∨ ✗
        done
```

### Automated PR Comments

```yaml
# .github/workflows/pr-assistant.yml
name: PR Assistant

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  analyze:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Analyze code changes
      id: analysis
      run: |
        python scripts/analyze_pr.py > analysis.md
    
    - name: Comment on PR
      uses: actions/github-script@v6
      with:
        script: |
          const analysis = require('fs').readFileSync('analysis.md', 'utf8');
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: analysis
          });
```

## Issue Management

### Issue Templates

```markdown
# .github/ISSUE_TEMPLATE/paradigm-request.md
---
name: Paradigm Request
about: Request a new paradigm implementation
labels: paradigm, enhancement
---

## Use Case
Describe the business problem this paradigm solves.

## Agents Required
List the types of agents and their roles.

## Success Criteria
What measurable outcomes indicate success?

## Timeline
When is this needed?
```

### Labeling System

- `paradigm`: Related to paradigm implementation
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `performance`: Performance-related issues
- `priority-high`: Blocking client work
- `good-first-issue`: Good for newcomers

## Git Hooks

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run quick tests
pytest tests/unit -x -q

# Check code style
black src/ --check
isort src/ --check

# Verify no broken paradigms
python scripts/quick_paradigm_check.py

# Check for test files
if ! git diff --cached --name-only | grep -q "test_"; then
  echo "Warning: No test files modified. Did you forget tests?"
fi
```

## Daily GitHub Workflow

### Morning Routine

```bash
# 1. Sync with latest [A]
git checkout develop
git pull origin develop

# 2. Check PR status [A]
gh pr status    # [WIP] ∨ [Review] ∨ [Approved]

# 3. Review assigned issues [A]
gh issue list --assignee @me

# 4. Start work on highest priority [A]→[WIP]
git checkout -b feature/issue-123
```

### During Development

```bash
# Frequent commits [A]
git add -A
git commit -m "feat(agents): Enhance attention mechanism"

# Check CI status [A]
gh workflow view    # [Build]→[Test]→✓ ∨ ✗

# Run local validation [A]
pytest tests/unit/test_current_work.py -v    # [Test]→✓ ∨ ✗
```

### End of Day

```bash
# Push work [A]→[WIP]
git push origin feature/current-branch

# Create/update PR [A]→[Review]
gh pr create --draft --title "WIP: Feature description"

# Update issue status [A]
gh issue comment 123 --body "Progress update: implemented attention and understanding mechanisms"
```

## Collaboration Patterns

### Code Review Focus

When reviewing PRs, check:
1. **Code quality**: Is the code clean and maintainable? ✓ ∨ ✗
2. **Test coverage**: Are all behaviors properly tested? ✓ ∨ ✗
3. **Paradigm impact**: Will this break existing paradigms? ✓ ∨ ✗
4. **Documentation**: Are changes documented? ✓ ∨ ✗
5. **Imperatives**: ■(∀agents: P1∧P2∧P3∧P4)

### Async Collaboration

```bash
# Leave detailed PR comments
gh pr review 456 --comment --body "
The decision logic here could be strengthened by...
"

# Reference issues and PRs
git commit -m "feat(paradigm): Address feedback from #456

Implements suggestions for better state transitions
See discussion in PR #456"
```

## Release Process

```bash
# 1. Create release branch [O]
git checkout -b release/v1.2.0 develop

# 2. Run full test suite [O]
pytest tests/ -v --cov=src --cov-report=html    # ■(∀tests: pass)

# 3. Update version and changelog [O]
python scripts/bump_version.py 1.2.0

# 4. Create PR to main [O]→[Review]
gh pr create --base main --title "Release v1.2.0"

# 5. After merge, tag release [Approved]→[Merged]
git tag -a v1.2.0 -m "Release v1.2.0: New customer service paradigm"
git push origin v1.2.0

# 6. Create GitHub release [O]
gh release create v1.2.0 --generate-notes
```

### Release Flow
[O]:Prepare→[Review]→[Approved]→[Merged]→[Tagged]→[Released]✓

## Remember

- Commit early and often [A]
- Keep PRs focused and reviewable [A]
- Automate everything you can [O]→[A]
- Document paradigm changes thoroughly [A]
- ■(∀workflows: maintainable ∧ efficient)