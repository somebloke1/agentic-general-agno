# Transcendental Agent Orchestration Framework

[![CI/CD Pipeline](https://github.com/somebloke1/agno-app/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/somebloke1/agno-app/actions/workflows/ci-cd.yml)
[![Test Suite](https://github.com/somebloke1/agno-app/workflows/Test%20Suite/badge.svg)](https://github.com/somebloke1/agno-app/actions/workflows/test.yml)
[![Claude Review](https://github.com/somebloke1/agno-app/workflows/Claude%20Code%20Review/badge.svg)](https://github.com/somebloke1/agno-app/actions/workflows/claude-code-review.yml)
[![Security](https://github.com/somebloke1/agno-app/workflows/Security%20Scan/badge.svg)](https://github.com/somebloke1/agno-app/actions/workflows/security.yml)

An autonomous agent orchestration framework based on Bernard Lonergan's transcendental method, implementing recursive self-improvement through P1→P2→P3→P4→R cycles.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/somebloke1/agno-app.git
cd agno-app

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Load a paradigm
python -m src.cli load paradigms/configs/question_answer.yaml
```

## 🏗️ Architecture

The framework implements Lonergan's transcendental imperatives:

- **◉ P1: Be Attentive** - Perceive and gather data
- **◈ P2: Be Intelligent** - Understand and form insights  
- **■ P3: Be Reasonable** - Judge and verify truth
- **★ P4: Be Responsible** - Decide and act ethically
- **↻ R: Recursion** - Self-improvement through governance

## 🔧 Core Components

1. **TranscendentalAgent**: Base agent with cognitive tracking
2. **ConsciousAgnoAgent**: Agno integration with imperative awareness
3. **Paradigms**: Configurable multi-agent interaction patterns
4. **Governance**: Recursive self-monitoring and improvement
5. **State Management**: Shared state across agent interactions

## 📝 Creating Paradigms

Define paradigms in YAML:

```yaml
name: ResearchTeam
description: Multi-agent research team
interaction_pattern: sequential
agents:
  - name: DataGatherer
    role: gatherer
    imperative_focus: ◉
    instructions: Gather comprehensive data
  - name: Analyst
    role: analyst
    imperative_focus: ◈
    instructions: Analyze and form insights
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/
pytest tests/paradigm/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch from `develop`
3. Write tests following TDD principles
4. Ensure all tests pass
5. Submit a PR with clear description

### Branch Protection

- `main`: Protected, requires PR + approval + all checks
- `develop`: Protected, requires PR + tests passing
- Feature branches: `feature/*` naming convention

### Auto-merge

PRs with the `auto-merge` label will automatically merge when:
- All tests pass
- Code review approved
- No `do-not-merge` label
- Not in draft state

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Built on the philosophical foundations of Bernard Lonergan's transcendental method.