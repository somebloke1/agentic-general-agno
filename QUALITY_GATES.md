# QUALITY_GATES.md - Automated Quality Checks and Standards

## Quality Flow
[Code]→[Style]→[Tests]→[Coverage]→[Delegation]→[Consultation]→✓ ∨ ✗

## Overview

Quality gates ensure code maintains high standards and paradigms preserve imperative fidelity. These checks run automatically in CI/CD and can be run locally.

## Code Quality Standards

### Python Code Style

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
        args: [--line-length=88]
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Install Pre-commit

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Task Delegation Checks

### Automated Orchestration Validation O:20%→A:80%

```python
# scripts/check_delegation.py
"""Verify agents maintain 80/20 orchestration ratio"""

import ast
import sys
from pathlib import Path
from typing import List, Dict

class TaskDelegationChecker(ast.NodeVisitor):
    """Check agent classes for proper task delegation"""
    
    def __init__(self):
        self.issues = []
        self.current_class = None
        self.delegation_stats = {}
        
    def visit_ClassDef(self, node):
        """Check each agent class"""
        if "Agent" in node.name:
            self.current_class = node.name
            self.check_delegation_pattern(node)
        self.generic_visit(node)
    
    def check_delegation_pattern(self, node):
        """Verify 80/20 delegation pattern"""
        
        delegation_calls = 0
        direct_work_calls = 0
        
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                # Count delegation vs direct work
                if isinstance(item.func, ast.Attribute):
                    method_name = item.func.attr
                    
                    # Delegation patterns
                    if method_name in ['delegate', 'assign_task', 'orchestrate', 
                                     'coordinate', 'consult', 'request_from']:
                        delegation_calls += 1
                    # Direct work patterns  
                    elif method_name in ['process', 'calculate', 'transform',
                                       'analyze', 'generate', 'execute']:
                        direct_work_calls += 1
        
        total_calls = delegation_calls + direct_work_calls
        if total_calls > 0:
            delegation_ratio = delegation_calls / total_calls
            
            self.delegation_stats[self.current_class] = {
                'delegation': delegation_calls,
                'direct': direct_work_calls,
                'ratio': delegation_ratio
            }
            
            # Check 80/20 rule (allow some flexibility) ■(delegation ≥ 70%)
            if delegation_ratio < 0.7:
                self.issues.append(
                    f"{self.current_class} delegation ratio too low: "
                    f"{delegation_ratio:.1%} (target: 80%)"
                )

def check_all_agents():
    """Check all agent files"""
    checker = TaskDelegationChecker()
    
    for agent_file in Path("src/agents").glob("**/*.py"):
        with open(agent_file) as f:
            tree = ast.parse(f.read())
            checker.visit(tree)
    
    # Report stats
    print("\n📊 Delegation Statistics:")
    for agent, stats in checker.delegation_stats.items():
        print(f"  {agent}: {stats['ratio']:.1%} delegation "
              f"({stats['delegation']} delegated, {stats['direct']} direct)")
    
    if checker.issues:
        print("\n❌ Delegation issues found:")    # ✗
        for issue in checker.issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("\n✅ All agents maintain proper delegation ratios")    # ✓

if __name__ == "__main__":
    check_all_agents()
```

## Consultation Tracking

### Automated Consultation Verification ■(∀failures: consulted)

```python
# scripts/check_consultations.py
"""Verify mandatory consultations are happening"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

class ConsultationTracker:
    """Track and verify consultation patterns"""
    
    def __init__(self, log_dir: Path = Path("logs/consultations")):
        self.log_dir = log_dir
        self.issues = []
        self.consultations = []
        
    def load_consultations(self, hours: int = 24):
        """Load recent consultation logs"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        for log_file in self.log_dir.glob("*.json"):
            try:
                with open(log_file) as f:
                    data = json.load(f)
                    
                timestamp = datetime.fromisoformat(data['timestamp'])
                if timestamp > cutoff:
                    self.consultations.append(data)
                    
            except Exception as e:
                print(f"Warning: Could not load {log_file}: {e}")
    
    def check_test_failure_consultations(self):
        """Verify consultations happen on test failures"""
        
        # Look for test failure events
        test_failures = [c for c in self.consultations 
                        if c.get('trigger') == 'test_failure']
        
        if not test_failures:
            print("⚠️  No test failure consultations found")
            return
            
        # Check each failure had proper consultation
        for failure in test_failures:
            if 'models_consulted' not in failure:
                self.issues.append(
                    f"Test failure at {failure['timestamp']} "
                    "had no model consultation"
                )
            elif len(failure['models_consulted']) < 2:
                self.issues.append(
                    f"Test failure at {failure['timestamp']} "
                    f"only consulted {len(failure['models_consulted'])} models"
                )
    
    def check_architecture_consultations(self):
        """Verify architecture decisions involve consultation"""
        
        arch_decisions = [c for c in self.consultations
                         if c.get('type') == 'architecture_decision']
        
        for decision in arch_decisions:
            # Architecture decisions should consult multiple models
            models = decision.get('models_consulted', [])
            
            if len(models) < 3:
                self.issues.append(
                    f"Architecture decision '{decision.get('topic')}' "
                    f"only consulted {len(models)} models (minimum: 3)"
                )
            
            # Should include diverse model types
            model_types = set(m.get('type') for m in models)
            if len(model_types) < 2:
                self.issues.append(
                    f"Architecture decision '{decision.get('topic')}' "
                    "lacks model diversity"
                )
    
    def generate_report(self):
        """Generate consultation summary report"""
        
        print("\n📊 Consultation Summary (last 24h):")
        print(f"  Total consultations: {len(self.consultations)}")
        
        # Group by type
        by_type = {}
        for c in self.consultations:
            c_type = c.get('type', 'unknown')
            by_type[c_type] = by_type.get(c_type, 0) + 1
        
        print("\n  By Type:")
        for c_type, count in sorted(by_type.items()):
            print(f"    {c_type}: {count}")
        
        # Model usage
        model_usage = {}
        for c in self.consultations:
            for model in c.get('models_consulted', []):
                model_name = model.get('name', 'unknown')
                model_usage[model_name] = model_usage.get(model_name, 0) + 1
        
        print("\n  Model Usage:")
        for model, count in sorted(model_usage.items(), 
                                  key=lambda x: x[1], reverse=True):
            print(f"    {model}: {count}")

def log_consultation(consultation_data: Dict):
    """Helper to log consultations"""
    
    log_dir = Path("logs/consultations")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    consultation_data['timestamp'] = datetime.now().isoformat()
    
    # Generate filename
    filename = f"consultation_{datetime.now():%Y%m%d_%H%M%S}.json"
    
    with open(log_dir / filename, 'w') as f:
        json.dump(consultation_data, f, indent=2)

def check_consultations():
    """Main consultation check"""
    tracker = ConsultationTracker()
    
    # Load recent consultations
    tracker.load_consultations()
    
    # Run checks
    tracker.check_test_failure_consultations()
    tracker.check_architecture_consultations()
    
    # Generate report
    tracker.generate_report()
    
    # Report issues
    if tracker.issues:
        print("\n❌ Consultation issues found:")    # ✗
        for issue in tracker.issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("\n✅ Consultation requirements met")    # ✓

if __name__ == "__main__":
    check_consultations()
```

### Example Consultation Logging

```python
# Example: Log a test failure consultation
from scripts.check_consultations import log_consultation

# When a test fails and consultation happens
log_consultation({
    'type': 'test_failure',
    'trigger': 'test_failure', 
    'test_name': 'test_agent_delegation',
    'topic': 'Agent delegation pattern failure',
    'models_consulted': [
        {'name': 'gemini-2.5-pro', 'type': 'analytical'},
        {'name': 'o3', 'type': 'reasoning'},
        {'name': 'grok-3', 'type': 'systems'}
    ],
    'outcome': 'Fixed delegation ratio calculation',
    'consensus': True
})

# When making architecture decisions
log_consultation({
    'type': 'architecture_decision',
    'topic': 'Agent communication protocol',
    'models_consulted': [
        {'name': 'gemini-2.5-pro', 'type': 'systems'},
        {'name': 'o3', 'type': 'analytical'}, 
        {'name': 'gpt-4.1', 'type': 'creative'},
        {'name': 'grok-3', 'type': 'implementation'}
    ],
    'decision': 'Async message passing with event sourcing',
    'rationale': 'Scalability and audit trail requirements'
})
```

## Test Quality Gates

### Test Coverage Requirements ■(coverage ≥ 80%)

```ini
# .coveragerc
[run]
source = src
omit = 
    */tests/*
    */migrations/*
    */venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

### Coverage Enforcement

```python
# scripts/check_coverage.py
"""Enforce minimum test coverage"""

import sys
import xml.etree.ElementTree as ET

def check_coverage(min_coverage=80):
    """Check if coverage meets minimum"""
    tree = ET.parse('coverage.xml')
    root = tree.getroot()
    
    # Get overall coverage
    coverage = float(root.attrib['line-rate']) * 100
    
    print(f"Overall coverage: {coverage:.1f}%")
    
    # Check specific modules
    for package in root.findall('.//package'):
        name = package.attrib['name']
        if 'paradigms' in name:
            # Paradigms need higher coverage
            pkg_coverage = float(package.attrib['line-rate']) * 100
            if pkg_coverage < 90:
                print(f"❌ {name} coverage too low: {pkg_coverage:.1f}%")
                sys.exit(1)
    
    if coverage < min_coverage:
        print(f"❌ Coverage {coverage:.1f}% below minimum {min_coverage}%")    # ✗
        sys.exit(1)
    
    print("✅ Coverage requirements met")    # ✓

if __name__ == "__main__":
    check_coverage()
```

## Paradigm Validation

### Paradigm Structure Validation

```python
# scripts/validate_paradigm.py
"""Validate paradigm configurations"""

import json
from pathlib import Path
from typing import Dict, List

class ParadigmValidator:
    """Validate paradigm meets requirements"""
    
    def __init__(self, paradigm_path: Path):
        self.paradigm_path = paradigm_path
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """Run all validations"""
        
        # Load paradigm config
        config = self.load_config()
        if not config:
            return False
        
        # Run validators
        self.validate_agents(config)
        self.validate_interactions(config)
        self.validate_success_metrics(config)
        self.validate_tests_exist()
        
        # Report results
        if self.errors:
            print(f"❌ Paradigm validation failed:")    # ✗
            for error in self.errors:
                print(f"  ERROR: {error}")
                
        if self.warnings:
            print(f"⚠️  Warnings:")    # ⚠
            for warning in self.warnings:
                print(f"  WARN: {warning}")
                
        return len(self.errors) == 0    # ✓ ∨ ✗
    
    def validate_agents(self, config: Dict):
        """Validate agent definitions"""
        
        if 'agents' not in config:
            self.errors.append("No agents defined")
            return
            
        for name, agent in config['agents'].items():
            # Check required fields
            required = ['role', 'instructions']
            for field in required:
                if field not in agent:
                    self.errors.append(
                        f"Agent {name} missing {field}"
                    )
            
            # Validate agent has clear role
            if 'role' in agent and len(agent['role']) < 10:
                self.warnings.append(
                    f"Agent {name} role description too brief"
                )
    
    def validate_tests_exist(self):
        """Ensure paradigm has tests"""
        
        paradigm_name = self.paradigm_path.stem
        test_file = Path(f"tests/paradigm/test_{paradigm_name}.py")
        
        if not test_file.exists():
            self.errors.append(f"No tests found at {test_file}")
        else:
            # Check test content
            with open(test_file) as f:
                content = f.read()
                
            required_tests = [
                "test_basic_functionality",
                "test_agent_delegation",
                "test_agent_interactions"
            ]
            
            for test in required_tests:
                if test not in content:
                    self.warnings.append(
                        f"Missing recommended test: {test}"
                    )
```

## Performance Gates

### Performance Benchmarks ◇(response<100ms)

```python
# tests/performance/test_paradigm_performance.py
import pytest
import time
import asyncio
from memory_profiler import profile

class TestParadigmPerformance:
    """Performance requirements for paradigms"""
    
    @pytest.mark.benchmark
    def test_agent_startup_time(self, benchmark):
        """Agent creation should be fast"""
        
        def create_agent():
            return AngryCustomerAgent()
        
        result = benchmark(create_agent)
        assert benchmark.stats['mean'] < 0.1  # Under 100ms ◇(t<100ms)
    
    @pytest.mark.asyncio
    async def test_parallel_execution_time(self):
        """Parallel execution should scale well"""
        
        # Single agent baseline
        start = time.time()
        agent = ResearchAgent()
        await agent.research("test topic")
        single_time = time.time() - start
        
        # Ten agents parallel
        start = time.time()
        agents = [ResearchAgent(f"r{i}") for i in range(10)]
        await asyncio.gather(*[
            agent.research("test topic") for agent in agents
        ])
        parallel_time = time.time() - start
        
        # Should be less than 2x single time ■(parallel_time < 2×single_time)
        assert parallel_time < single_time * 2
    
    @profile
    def test_memory_usage(self):
        """Check memory usage stays reasonable"""
        
        agents = []
        for i in range(100):
            agents.append(CustomerServiceAgent(f"agent_{i}"))
        
        # Memory usage tracked by @profile decorator
        # CI can parse output and fail if too high
```

## Security Gates

### Security Scanning

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Bandit
      run: |
        pip install bandit
        bandit -r src/ -ll -i
    
    - name: Check for secrets
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        
    - name: Dependency check
      run: |
        pip install safety
        safety check --json
```

### Paradigm Security Validation

```python
# scripts/check_paradigm_security.py
"""Ensure paradigms don't leak sensitive data"""

def check_paradigm_security(paradigm_config):
    """Security checks for paradigms"""
    
    issues = []
    
    # Check for hardcoded credentials
    config_str = str(paradigm_config)
    patterns = [
        r'api_key.*=.*["\'][^"\']+["\']',
        r'password.*=.*["\'][^"\']+["\']',
        r'secret.*=.*["\'][^"\']+["\']'
    ]
    
    for pattern in patterns:
        if re.search(pattern, config_str, re.I):
            issues.append("Possible hardcoded credentials")
    
    # Check agent instructions
    for agent in paradigm_config.get('agents', {}).values():
        instructions = ' '.join(agent.get('instructions', []))
        
        # No PII in instructions
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', instructions):
            issues.append("SSN pattern in instructions")
            
        # No real endpoints
        if re.search(r'https?://(?!localhost|example)', instructions):
            issues.append("Real URL in instructions")
    
    return issues
```

## Documentation Gates

### Documentation Requirements ■(∀paradigms: documented)

```python
# scripts/check_documentation.py
"""Ensure proper documentation"""

def check_paradigm_docs():
    """Each paradigm needs documentation"""
    
    for paradigm_file in Path("src/paradigms").glob("*.py"):
        paradigm_name = paradigm_file.stem
        doc_file = Path(f"docs/paradigms/{paradigm_name}.md")
        
        if not doc_file.exists():
            print(f"❌ Missing docs for {paradigm_name}")    # ✗
            return False
            
        # Check documentation content
        with open(doc_file) as f:
            content = f.read()
            
        required_sections = [
            "## Overview",
            "## Use Case", 
            "## Agents",
            "## Success Criteria",
            "## Example Usage"
        ]
        
        for section in required_sections:
            if section not in content:
                print(f"❌ {paradigm_name} docs missing {section}")    # ✗
                return False
    
    print("✅ All paradigms documented")    # ✓
    return True
```

## Local Quality Checks

### Run All Checks Locally

```bash
#!/bin/bash
# scripts/quality_check.sh

echo "Running quality checks..."

# Code style
echo "1. Code style..."
black src/ --check
isort src/ --check
flake8 src/

# Tests
echo "2. Running tests..."
pytest tests/unit -v

# Coverage
echo "3. Checking coverage..."
pytest tests/ --cov=src --cov-report=term-missing

# Delegation patterns
echo "4. Checking delegation patterns..."
python scripts/check_delegation.py

# Consultations
echo "5. Checking consultations..."
python scripts/check_consultations.py

# Documentation
echo "6. Checking documentation..."
python scripts/check_documentation.py

echo "✅ All quality checks passed!"    # ✓
```

## Git Hooks Integration

### Pre-commit Hook (Essential Checks Only)

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running pre-commit checks..."

# Format check only (fast) [Style]→✓ ∨ ✗
black src/ --check --quiet
if [ $? -ne 0 ]; then
    echo "❌ Code formatting issues. Run 'black src/' to fix."    # ✗
    exit 1
fi

# Quick unit tests only [Test]→✓ ∨ ✗
pytest tests/unit -x --quiet
if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed."    # ✗
    exit 1
fi

echo "✅ Pre-commit checks passed"    # ✓
```

### Pre-push Hook (Minimal)

```bash
# .git/hooks/pre-push
#!/bin/bash

echo "Running pre-push validation..."

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing --quiet

if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Push aborted."    # ✗→[Push]
    exit 1
fi

echo "✅ Ready to push"    # ✓→[Push]
```

## CI/CD Integration

All quality gates run automatically in GitHub Actions. See `.github/workflows/quality.yml` for full configuration.

## Remember

- Quality gates catch problems early ?→✗→fix
- Automate everything you can [O]→[A]
- Keep checks fast (<5 minutes total) ◇(t<5min)
- Fix warnings before they become errors ⚠→fix→✓
- Quality enables speed, not hinders it quality⇒velocity
- Track consultations for accountability ■(∀consultations: tracked)
- Enforce 80/20 delegation patterns O:20%→A:80%
- Simple hooks for developer productivity [A]→productivity↑