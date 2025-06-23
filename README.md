# Quantum Factoring - Shor's Algorithm

This code breaks numbers into their factors using quantum computing.

## How to use it

Run it. It will factor the number 15 by default.

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

## How Shor's Algorithm Works

### The Classical Approach (Slow)
Normal computers have to basically guess and check:
- Try dividing by 2, 3, 5, 7, 11, 13...
- Keep going until you find a factor
- For big numbers, this takes forever

### Shor's Quantum Trick (Fast)
Instead of trying to factor directly, Shor's algorithm does something clever:

#### Step 1: Pick a Random Number
Choose some number `a` that's smaller than the number `N` you want to factor.

#### Step 2: Find the Period
This is the quantum magic part. We want to find the "period" of the function f(x) = a^x mod N.

What's a period? It's how often the pattern repeats:
- 7^1 mod 15 = 7
- 7^2 mod 15 = 4  
- 7^3 mod 15 = 13
- 7^4 mod 15 = 1  ← Back to 1, so period = 4
- 7^5 mod 15 = 7  ← Pattern starts over

#### Step 3: The Quantum Circuit
The quantum computer creates a superposition of ALL possible values of x at once, then computes a^x mod N for all of them simultaneously.

The circuit has two parts:
- **Counting register**: Holds all the x values in superposition
- **Work register**: Computes a^x mod N for each x

#### Step 4: Quantum Fourier Transform
This is where the period gets extracted. The QFT finds hidden periodicities in the quantum state.

#### Step 5: Classical Math
Once we have the period r, we use regular math:
- Compute x = a^(r/2) mod N
- Calculate gcd(x-1, N) and gcd(x+1, N)
- One of these will be a factor of N

### Why It's Faster
- **Classical**: Try O(√N) possible factors → exponential time
- **Quantum**: Find period in O(log³ N) time → polynomial time

## What the output means

- **Circuit**: How big the quantum computer needs to be
- **Phases**: Quantum measurements that hide the pattern we need
- **Modular results**: The actual math results from the quantum circuit
- **Period**: The hidden pattern that leads to the factors

## Limitations

Only works on small numbers because we're simulating a quantum computer on a regular computer.

## Requirements

```bash
pip install cirq qsimcirq numpy
```
