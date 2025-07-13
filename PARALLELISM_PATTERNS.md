# PARALLELISM_PATTERNS.md - Parallel Agent Execution Patterns

## Overview

Parallel execution enables multiple agents to operate simultaneously while maintaining imperative fidelity and coordination. This is crucial for performance and realistic multi-agent simulations.

## Python Async Patterns

### Basic Parallel Agent Execution

```python
import asyncio
from typing import List, Dict, Any
import aiohttp

class ParallelOrchestrator:
    """Manages parallel agent execution"""
    
    async def execute_agents_parallel(self, agents: List[ConsciousAgent], 
                                    input_data: Any) -> Dict[str, Any]:
        """Execute multiple agents in parallel"""
        
        # Create tasks for each agent
        tasks = []
        for agent in agents:
            task = asyncio.create_task(
                self._execute_with_monitoring(agent, input_data)
            )
            tasks.append((agent.name, task))
        
        # Wait for all to complete
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                results[name] = {"error": str(e)}
                
        return results
    
    async def _execute_with_monitoring(self, agent: ConsciousAgent, 
                                     input_data: Any) -> Any:
        """Execute agent with imperative monitoring"""
        start_time = asyncio.get_event_loop().time()
        
        # Execute agent
        result = await agent.async_execute(input_data)
        
        # Verify imperatives maintained
        if not agent.cognitive_trace.validates_imperatives():
            raise ValueError(f"{agent.name} violated imperatives during execution")
            
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return {
            "result": result,
            "execution_time": execution_time,
            "cognitive_trace": agent.cognitive_trace.get_summary()
        }
```

### Parallel Paradigm Patterns

#### Pattern 1: Independent Parallel

```python
class IndependentParallel:
    """Agents work independently on same input"""
    
    async def execute(self, paradigm_config: ParadigmConfig, 
                     test_input: Any) -> Dict:
        """Customer service stress test pattern"""
        
        # All agents attack the service simultaneously
        agents = [
            AngryCustomerAgent(),
            ConfusedElderlyAgent(),
            SocialEngineerAgent()
        ]
        
        # Execute in parallel
        async with asyncio.TaskGroup() as tg:
            tasks = {
                agent.name: tg.create_task(agent.test_service(test_input))
                for agent in agents
            }
        
        # Collect results
        results = {
            name: task.result() 
            for name, task in tasks.items()
        }
        
        # Analyze collective impact
        return self.analyze_stress_test(results)
```

#### Pattern 2: Pipeline Parallel

```python
class PipelineParallel:
    """Agents work in parallel stages"""
    
    async def execute(self, research_topic: str) -> Dict:
        """Research team with parallel data gathering"""
        
        # Stage 1: Parallel data gathering
        empiricists = [
            EmpiricalResearcher("academic_sources"),
            EmpiricalResearcher("industry_reports"),
            EmpiricalResearcher("government_data")
        ]
        
        data_tasks = [
            asyncio.create_task(emp.gather_data(research_topic))
            for emp in empiricists
        ]
        
        all_data = await asyncio.gather(*data_tasks)
        
        # Stage 2: Parallel theory formation
        theorists = [
            Theorist("systems_thinking"),
            Theorist("statistical_analysis"),
            Theorist("causal_reasoning")
        ]
        
        theory_tasks = [
            asyncio.create_task(theorist.form_insights(all_data))
            for theorist in theorists
        ]
        
        theories = await asyncio.gather(*theory_tasks)
        
        # Stage 3: Unified critique
        critic = CriticalEvaluator()
        final_judgment = await critic.evaluate_theories(theories)
        
        return final_judgment
```

#### Pattern 3: Competitive Parallel

```python
class CompetitiveParallel:
    """Agents race to find best solution"""
    
    async def execute(self, problem: str) -> Dict:
        """Multiple agents compete to solve problem fastest/best"""
        
        solvers = [
            ProblemSolver("algorithmic", model="o3"),
            ProblemSolver("heuristic", model="gemini-pro"),
            ProblemSolver("creative", model="claude-opus")
        ]
        
        # First valid solution wins
        for future in asyncio.as_completed(
            [solver.solve(problem) for solver in solvers]
        ):
            solution = await future
            if self.validate_solution(solution):
                # Cancel remaining tasks
                for solver in solvers:
                    solver.cancel()
                return solution
                
        raise ValueError("No valid solution found")
```

## Coordination Mechanisms

### Shared State Management

```python
class SharedState:
    """Thread-safe shared state for parallel agents"""
    
    def __init__(self):
        self._state = {}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        
    async def update(self, agent_name: str, key: str, value: Any):
        """Atomic state update"""
        async with self._lock:
            if agent_name not in self._state:
                self._state[agent_name] = {}
            self._state[agent_name][key] = value
            
        # Notify waiters
        async with self._condition:
            self._condition.notify_all()
    
    async def wait_for(self, condition_func):
        """Wait for condition to be met"""
        async with self._condition:
            while not condition_func(self._state):
                await self._condition.wait()
```

### Message Passing

```python
class MessageBus:
    """Async message passing between agents"""
    
    def __init__(self):
        self.queues = {}
        
    def register_agent(self, agent_name: str):
        """Create dedicated queue for agent"""
        self.queues[agent_name] = asyncio.Queue()
    
    async def send(self, from_agent: str, to_agent: str, message: Any):
        """Send message to specific agent"""
        if to_agent in self.queues:
            await self.queues[to_agent].put({
                "from": from_agent,
                "to": to_agent,
                "message": message,
                "timestamp": asyncio.get_event_loop().time()
            })
    
    async def broadcast(self, from_agent: str, message: Any):
        """Broadcast to all agents"""
        tasks = [
            self.send(from_agent, agent, message)
            for agent in self.queues.keys()
            if agent != from_agent
        ]
        await asyncio.gather(*tasks)
    
    async def receive(self, agent_name: str, timeout: float = None):
        """Receive message for agent"""
        try:
            return await asyncio.wait_for(
                self.queues[agent_name].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
```

## Performance Patterns

### Batching for Efficiency

```python
class BatchProcessor:
    """Batch multiple agent requests for efficiency"""
    
    def __init__(self, batch_size: int = 10, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending = []
        self.results = {}
        
    async def process_request(self, agent_name: str, request: Any):
        """Add request to batch"""
        future = asyncio.Future()
        self.pending.append((agent_name, request, future))
        
        # Process batch if full
        if len(self.pending) >= self.batch_size:
            await self._process_batch()
            
        # Schedule timeout processing
        asyncio.create_task(self._timeout_process())
        
        return await future
    
    async def _process_batch(self):
        """Process all pending requests together"""
        if not self.pending:
            return
            
        batch = self.pending[:self.batch_size]
        self.pending = self.pending[self.batch_size:]
        
        # Process in parallel
        results = await self._batch_api_call(
            [(name, req) for name, req, _ in batch]
        )
        
        # Resolve futures
        for i, (_, _, future) in enumerate(batch):
            future.set_result(results[i])
```

### Resource Pooling

```python
class AgentPool:
    """Reusable agent pool for efficiency"""
    
    def __init__(self, agent_class, pool_size: int = 5):
        self.agent_class = agent_class
        self.pool = asyncio.Queue()
        self.all_agents = []
        
        # Pre-create agents
        for _ in range(pool_size):
            agent = agent_class()
            self.all_agents.append(agent)
            self.pool.put_nowait(agent)
    
    async def acquire(self) -> ConsciousAgent:
        """Get agent from pool"""
        return await self.pool.get()
    
    async def release(self, agent: ConsciousAgent):
        """Return agent to pool"""
        # Reset agent state
        agent.reset()
        await self.pool.put(agent)
    
    @asynccontextmanager
    async def agent(self):
        """Context manager for agent usage"""
        agent = await self.acquire()
        try:
            yield agent
        finally:
            await self.release(agent)
```

## Testing Parallel Patterns

### Testing Race Conditions

```python
@pytest.mark.asyncio
async def test_parallel_imperative_fidelity():
    """Ensure parallel execution maintains imperatives"""
    
    # Create multiple agents
    agents = [ResearchAgent(f"researcher_{i}") for i in range(10)]
    
    # Execute same task in parallel
    orchestrator = ParallelOrchestrator()
    results = await orchestrator.execute_agents_parallel(
        agents, 
        "research quantum computing"
    )
    
    # Verify all maintained imperatives
    for agent_name, result in results.items():
        assert "error" not in result
        trace = result["cognitive_trace"]
        assert trace["validates_p1_through_p4"]
        assert trace["recursive_governance_active"]
```

### Testing Coordination

```python
@pytest.mark.asyncio
async def test_message_coordination():
    """Test agents coordinate via messages"""
    
    bus = MessageBus()
    agents = []
    
    # Create interconnected agents
    for i in range(3):
        agent = CoordinatingAgent(f"agent_{i}", bus)
        bus.register_agent(agent.name)
        agents.append(agent)
    
    # Start all agents
    tasks = [agent.run() for agent in agents]
    
    # Wait for coordination
    await asyncio.gather(*tasks)
    
    # Verify coordination occurred
    for agent in agents:
        assert agent.received_from_peers
        assert agent.consensus_reached
```

## Production Patterns

### Graceful Degradation

```python
class ResilientParallelism:
    """Handle failures in parallel execution"""
    
    async def execute_with_fallback(self, primary_agents: List[ConsciousAgent],
                                   fallback_agents: List[ConsciousAgent],
                                   task: Any) -> Dict:
        """Try primary agents, fall back if needed: primary∨fallback"""
        
        try:
            # Try primary agents with timeout
            return await asyncio.wait_for(
                self._execute_primary(primary_agents, task),
                timeout=30.0
            )
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Primary failed: {e}, trying fallback")
            
            # (primary✗)⇒fallback
            return await self._execute_fallback(fallback_agents, task)
```

### Monitoring and Metrics

```python
class ParallelMetrics:
    """Track parallel execution metrics"""
    
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.imperative_violations = defaultdict(int)
        
    async def track_execution(self, agent_name: str, coro):
        """Track execution metrics"""
        start = asyncio.get_event_loop().time()
        
        try:
            result = await coro
            execution_time = asyncio.get_event_loop().time() - start
            self.execution_times[agent_name].append(execution_time)
            return result
            
        except ImperativeViolation:
            self.imperative_violations[agent_name] += 1
            raise
            
        except Exception as e:
            self.error_counts[agent_name] += 1
            raise
```

## Best Practices

### DO:
- Use `asyncio.gather()` for independent tasks
- Implement proper timeout handling
- Monitor resource usage
- Test race conditions thoroughly
- Use connection pooling for external calls

### DON'T:
- Share state without locks
- Ignore backpressure
- Create unbounded queues
- Forget error isolation
- Assume order of completion

## Quick Reference

```python
# Parallel execution patterns
await asyncio.gather(*tasks)  # All must complete
async with asyncio.TaskGroup() as tg:  # Better error handling
    tasks = [tg.create_task(coro) for coro in coros]

# First completion wins
for future in asyncio.as_completed(tasks):
    result = await future
    if valid(result):
        break

# Timeout handling
try:
    await asyncio.wait_for(coro, timeout=30.0)
except asyncio.TimeoutError:
    handle_timeout()

# Semaphore for rate limiting
sem = asyncio.Semaphore(10)
async with sem:
    await make_api_call()
```

## Remember

- Parallelism improves performance, not correctness
- Always maintain imperative fidelity
- Test parallel scenarios explicitly
- Monitor and measure in production
- Design for failure cases