# PLAYWRIGHT_TESTING.md - UI Testing with Playwright MCP

> See also: [SYMBOLIC_NOTATION.md](SYMBOLIC_NOTATION.md) for notation reference

## Overview

Use playwright-mcp-server to validate [A] paradigms through UI testing. This ensures paradigms work not just in code but in actual user interfaces.

**Key**: Test both behavior ∧ imperative visibility

## Setup

```bash
# Add Playwright MCP server
claude mcp add playwright npx @playwright/mcp@latest

# Install dependencies
npm install -D @playwright/test
npx playwright install chromium
```

## When to Use Playwright Testing

1. **Paradigm UI validation** - Ensure [A] display correctly
2. **Interaction testing** - Verify A₁⇄A₂ coordination visually
3. **Client demos** - Test presentation layer
4. **Performance validation** - Measure UI responsiveness
5. **Accessibility testing** - Ensure ∀users: accessible

## Basic UI Test Structure

### Testing Agent Visualization

```python
"""
@playwright action=test

Test customer service paradigm UI:

1. Navigate to http://localhost:3000/paradigms/customer-service
2. Verify [A] cards are displayed:
   - A₁: Angry Customer (red border)
   - A₂: Confused Elderly (blue border)
   - A₃: Social Engineer (orange border)

3. Click "Start Simulation"
4. Verify chat interfaces appear for A₁⊕A₂⊕A₃
5. Verify message flow visualization: A→Service→A
"""
```

### Testing Agent Interactions

```python
"""
@playwright action=record

Record multi-agent interaction:

1. Navigate to paradigm dashboard
2. Start research team paradigm
3. Verify E₁ shows "Gathering data..." status
4. Verify T₁ waits for E₁ data (E₁⊸T₁)
5. When E₁ completes, verify data→T₁
6. Verify C₁ activates after T₁ proposes insights
7. Check final consensus: Σ(E₁,T₁,C₁)→!

Save as: tests/ui/research-team-flow.spec.ts
"""
```

## Paradigm-Specific Test Patterns

### Customer Service UI Tests

```javascript
// tests/ui/customer-service-paradigm.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Customer Service Paradigm UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/paradigms/customer-service');
  });

  test('displays all customer personas', async ({ page }) => {
    // Verify agent cards
    await expect(page.locator('.agent-card')).toHaveCount(3);
    
    // Check angry customer
    const angryCard = page.locator('.agent-card:has-text("Angry Customer")');
    await expect(angryCard).toHaveClass(/border-red/);
    await expect(angryCard.locator('.mood-indicator')).toHaveText('😠');
    
    // Check confused elderly
    const confusedCard = page.locator('.agent-card:has-text("Confused Elderly")');
    await expect(confusedCard).toHaveClass(/border-blue/);
    await expect(confusedCard.locator('.mood-indicator')).toHaveText('😕');
  });

  test('shows cognitive traces on demand', async ({ page }) => {
    // Start simulation
    await page.click('button:has-text("Start Simulation")');
    
    // Wait for first interaction
    await page.waitForSelector('.message-bubble');
    
    // Toggle cognitive trace
    await page.click('button:has-text("Show Cognitive Trace")');
    
    // Verify P1→P2→P3→P4 display
    await expect(page.locator('.trace-p1')).toContainText('P1: Attended to:');
    await expect(page.locator('.trace-p2')).toContainText('P2: Understood:');
    await expect(page.locator('.trace-p3')).toContainText('P3: Judged:');
    await expect(page.locator('.trace-p4')).toContainText('P4: Decided:');
  });

  test('escalation visualization', async ({ page }) => {
    // Start angry customer
    await page.click('.agent-card:has-text("Angry Customer") button');
    
    // Send repeated unhelpful response
    const chat = page.locator('.chat-interface');
    await chat.locator('input').fill('Please hold');
    await chat.locator('button:has-text("Send")').click();
    
    // Repeat
    await chat.locator('input').fill('Please hold');
    await chat.locator('button:has-text("Send")').click();
    
    // Check escalation indicator
    await expect(page.locator('.escalation-meter')).toHaveAttribute(
      'data-level',
      '2'
    );
    
    // Verify visual changes
    await expect(chat).toHaveClass(/escalated/);
  });
});
```

### Research Team UI Tests

```javascript
test.describe('Research Team Paradigm UI', () => {
  test('visualizes data flow between agents', async ({ page }) => {
    await page.goto('/paradigms/research-team');
    await page.click('button:has-text("Start Research")');
    
    // Wait for E₁ to start
    await expect(page.locator('.empiricist-status')).toHaveText('Gathering data...');
    
    // Verify data flow arrow appears: E₁→T₁
    await page.waitForSelector('.data-flow-arrow.empiricist-to-theorist');
    
    // Check T₁ receives data
    await expect(page.locator('.theorist-input-count')).not.toHaveText('0');
    
    // Verify complete flow visualization
    const flowDiagram = page.locator('.paradigm-flow-diagram');
    await expect(flowDiagram).toBeVisible();
    await expect(flowDiagram).toHaveScreenshot('research-flow-active.png');
  });

  test('shows peer review connections', async ({ page }) => {
    // Navigate and start
    await page.goto('/paradigms/research-team');
    await page.click('button:has-text("Enable Peer Review")');
    
    // Verify review connections appear: A₁⇄A₂⇄A₃
    const connections = page.locator('.peer-review-connection');
    await expect(connections).toHaveCount(3); // Each reviews another
    
    // Test review interaction: hover(E₁)⇒show(C₁→E₁)
    await page.hover('.agent-node.empiricist');
    await expect(page.locator('.review-from-critic')).toBeVisible();
  });
});
```

## Visual Testing Patterns

### Screenshot Comparison

```python
"""
@playwright action=screenshot

Capture paradigm states for visual regression:

1. Navigate to each paradigm
2. Capture "idle" state
3. Start paradigm
4. Capture "active" state  
5. Complete one cycle
6. Capture "completed" state

Save to: tests/ui/screenshots/paradigm-states/
"""
```

### Accessibility Testing

```javascript
test.describe('Paradigm Accessibility', () => {
  test('meets WCAG standards', async ({ page }) => {
    await page.goto('/paradigms/customer-service');
    
    // Check color contrast
    const results = await page.evaluate(() => {
      // Run axe-core
      return window.axe.run();
    });
    
    expect(results.violations).toHaveLength(0);
    
    // Keyboard navigation
    await page.keyboard.press('Tab');
    await expect(page.locator(':focus')).toHaveAttribute('role', 'button');
    
    // Screen reader labels
    const agentCards = page.locator('[role="article"]');
    await expect(agentCards.first()).toHaveAttribute(
      'aria-label',
      /Agent: .+ Role: .+/
    );
  });
});
```

## Performance Testing

### UI Responsiveness

```javascript
test('paradigm UI stays responsive under load', async ({ page }) => {
  await page.goto('/paradigms/customer-service');
  
  // Start A₁⊕A₂⊕A₃ simultaneously
  const startTime = Date.now();
  
  await Promise.all([
    page.click('.agent-card:nth-child(1) button'),
    page.click('.agent-card:nth-child(2) button'),
    page.click('.agent-card:nth-child(3) button'),
  ]);
  
  // Measure time to interactive: ◇(responsive)
  await page.waitForLoadState('networkidle');
  const loadTime = Date.now() - startTime;
  
  expect(loadTime).toBeLessThan(1000); // ∃t: t<1s
  
  // Test interaction responsiveness
  const inputTime = Date.now();
  await page.fill('.chat-input', 'Test message');
  const inputComplete = Date.now() - inputTime;
  
  expect(inputComplete).toBeLessThan(100); // Under 100ms
});
```

### Memory Monitoring

```python
"""
@playwright action=monitor

Monitor memory usage during extended paradigm run:

1. Open DevTools Performance Monitor
2. Start research team paradigm
3. Run for 10 minutes
4. Monitor:
   - Heap size growth
   - DOM node count
   - Event listener count
   
Alert if:
- Heap grows >50MB
- DOM nodes >5000
- Listeners >1000
"""
```

## Integration with CI/CD

### GitHub Action for UI Tests

```yaml
# .github/workflows/ui-tests.yml
name: UI Tests

on:
  pull_request:
    paths:
      - 'ui/**'
      - 'tests/ui/**'
      - 'src/paradigms/**'

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node
      uses: actions/setup-node@v3
      with:
        node-version: 18
    
    - name: Install dependencies
      run: |
        npm ci
        npx playwright install --with-deps
    
    - name: Start application
      run: |
        npm run build
        npm start &
        npx wait-on http://localhost:3000
    
    - name: Run Playwright tests
      run: npx playwright test
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: playwright-report
        path: playwright-report/
```

## Common UI Test Scenarios

### Multi-Agent Coordination

```python
"""
@playwright action=test

Test surgical planning team coordination:

1. Start paradigm with test patient data
2. Verify anatomical mapper highlights risk areas in red
3. Verify procedure strategist shows approach options
4. When mapper selects high-risk area:
   - Strategist options should update
   - Risk assessment panel shows details
5. Verify consensus indicator when all agents agree
"""
```

### Error State Testing

```javascript
test('handles agent failures gracefully', async ({ page }) => {
  await page.goto('/paradigms/customer-service');
  
  // Simulate agent timeout
  await page.route('**/api/agents/angry-customer', route => {
    route.abort('timedout');
  });
  
  await page.click('button:has-text("Start Simulation")');
  
  // Should show error state: A₁✗
  await expect(page.locator('.agent-error')).toBeVisible();
  await expect(page.locator('.agent-error')).toContainText('Unable to connect');
  
  // Other agents should continue: A₁✗ ∧ (A₂✓ ∧ A₃✓)
  await expect(page.locator('.agent-card:has-text("Confused Elderly")')).not.toHaveClass(/error/);
});
```

## Debugging UI Tests

### Using Playwright Inspector

```bash
# Run with inspector
npx playwright test --debug

# Run specific test with UI
npx playwright test customer-service --headed
```

### Trace Viewer

```bash
# Run with trace
npx playwright test --trace on

# View trace
npx playwright show-trace trace.zip
```

## Best Practices

### DO:
- Test both happy paths and error states
- Verify cognitive traces are visible
- Check accessibility requirements
- Test responsive design
- Monitor performance metrics

### DON'T:
- Test implementation details
- Rely only on CSS selectors
- Ignore flaky tests
- Skip visual regression tests
- Forget mobile viewports

## Quick Test Commands

```bash
# Run all UI tests
npm run test:ui

# Run specific paradigm tests
npm run test:ui -- customer-service

# Update snapshots
npm run test:ui -- --update-snapshots

# Run in headed mode
npm run test:ui -- --headed

# Generate HTML report
npm run test:ui -- --reporter=html
```

## Remember

- UI tests validate the complete experience
- Visual testing catches CSS regressions
- Performance testing ensures: ◇(usable)
- Accessibility testing ensures: ∀users: accessible
- Keep tests maintainable ∧ fast