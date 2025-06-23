# Quantum Factoring - Shor's Algorithm

This code breaks numbers into their factors using quantum computing instead of regular math.

## What it does

Takes a number like 15 and finds that 15 = 3 × 5. But instead of guessing and checking like normal computers, it uses quantum weirdness to find the answer "faster".

## How to use it

Run it. It will factor the number 15 by default. If you want to try a different number, change the `N = 15` line at the bottom.

## What happens when you run it

```
Factoring N = 15 using Shor's algorithm

Attempt 1: base a = 7
Circuit: 62 moments, 20 qubits
Top phases: {0: 181, 4095: 105, 1: 101}
Modular results: {8: 372}
Quantum period found: 4
x = 7^2 mod 15 = 4
gcd(4±1, 15) = 3, 5
Success! Factors: [3, 5]

Result: 15 = 3 × 5
```

## The basic steps

1. Pick a random number
2. Build a quantum circuit with about 20 qubits
3. Run quantum math to find a hidden pattern
4. Use that pattern to get the factors

## Why this is cool

- Quantum computers can factor huge numbers way faster than regular computers
- This algorithm could break internet encryption someday
- It shows quantum computers can solve certain problems exponentially faster

## What the output means

- **Circuit**: How big the quantum computer needs to be
- **Phases**: Quantum measurements that hide the pattern we need
- **Modular results**: The actual math results from the quantum circuit
- **Period**: The hidden pattern that leads to the factors

## Limitations

Only works on small numbers because we're simulating a quantum computer on a regular computer. Real quantum computers could factor much bigger numbers.

## Requirements

You need Python with cirq and qsimcirq installed. That's Google's quantum computing simulator.
