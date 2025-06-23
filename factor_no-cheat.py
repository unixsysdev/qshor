#!/usr/bin/env python3

# Attempts to implement most of the quantum operations

import cirq
import qsimcirq
import numpy as np
import math
from fractions import Fraction
import random

class QuantumPeriodFinder:
    def __init__(self, N, a):
        self.N = N
        self.a = a
        self.n = math.ceil(math.log2(N)) if N > 1 else 1
        self.m = max(8, 3 * self.n)
        
        self.counting_qubits = [cirq.GridQubit(0, i) for i in range(self.m)]
        self.work_qubits = [cirq.GridQubit(1, i) for i in range(self.n)]
        self.aux_qubits = [cirq.GridQubit(2, i) for i in range(max(25, self.n + 20))]
        
        self.powers = []
        power = self.a % self.N
        for k in range(self.m):
            self.powers.append(power)
            power = (power * power) % self.N
    
    def quantum_adder(self, a_qubits, b_qubits, carry_out):
        ops = []
        n = len(a_qubits)
        
        # Simple ripple carry adder with proper qubit management
        for i in range(n):
            if i == 0:
                # First bit: sum = a XOR b, carry = a AND b
                ops.append(cirq.CNOT(a_qubits[i], b_qubits[i]))
                if n > 1:  # Only add carry if we have more bits
                    ops.append(cirq.CCNOT(a_qubits[i], b_qubits[i], carry_out))
            else:
                # Get unique auxiliary qubit for this bit's carry
                temp_carry_idx = self.n + 15 + i
                if temp_carry_idx < len(self.aux_qubits):
                    temp_carry = self.aux_qubits[temp_carry_idx]
                else:
                    temp_carry = carry_out  # Fallback
                
                # sum = a XOR b XOR carry_in
                ops.append(cirq.CNOT(carry_out, b_qubits[i]))
                ops.append(cirq.CNOT(a_qubits[i], b_qubits[i]))
                
                # New carry generation - only if not last bit
                if i < n - 1 and temp_carry != carry_out:
                    ops.append(cirq.CCNOT(a_qubits[i], carry_out, temp_carry))
                    ops.append(cirq.CCNOT(a_qubits[i], b_qubits[i], temp_carry))
                    ops.append(cirq.CNOT(temp_carry, carry_out))
        
        return ops
    
    def quantum_subtractor(self, a_qubits, b_qubits, borrow_out):
        ops = []
        n = len(a_qubits)
        
        for i in range(n):
            # Basic subtraction: result = a - b
            ops.append(cirq.CNOT(a_qubits[i], b_qubits[i]))
            
            # Simple borrow handling for first bit only
            if i == 0 and n > 1:
                ops.append(cirq.X(a_qubits[i]))
                ops.append(cirq.CCNOT(a_qubits[i], b_qubits[i], borrow_out))
                ops.append(cirq.X(a_qubits[i]))
        
        return ops
    
    def quantum_compare_greater(self, a_qubits, b_value, result_qubit):
        ops = []
        n = len(a_qubits)
        b_bits = [(b_value >> i) & 1 for i in range(n)]
        
        # Simplified comparison to avoid conflicts
        for i in range(n-1, -1, -1):
            if b_bits[i] == 0:
                # If b[i] = 0 and a[i] = 1, then a > b
                ops.append(cirq.CNOT(a_qubits[i], result_qubit))
        
        return ops
    
    def quantum_modular_reduction(self):
        ops = []
        if len(self.aux_qubits) < self.n + 15:
            return ops  # Skip if not enough qubits
        
        comparison_qubit = self.aux_qubits[self.n + 2]
        
        # Simple modular reduction: if work >= N, subtract N
        ops.extend(self.quantum_compare_greater(self.work_qubits, self.N - 1, comparison_qubit))
        
        # Conditional subtraction of N
        N_bits = [(self.N >> i) & 1 for i in range(self.n)]
        for i in range(self.n):
            if N_bits[i] == 1:
                ops.append(cirq.CNOT(comparison_qubit, self.work_qubits[i]))
        
        # Uncompute comparison (simplified)
        ops.extend(self.quantum_compare_greater(self.work_qubits, self.N - 1, comparison_qubit))
        
        return ops
    
    def conditional_set_value(self, control, temp_qubits, target_val, result_val):
        ops = []
        if len(self.aux_qubits) < self.n + 6:
            return ops
        
        condition_qubit = self.aux_qubits[self.n + 3]
        
        for i in range(self.n):
            if not ((target_val >> i) & 1):
                ops.append(cirq.X(temp_qubits[i]))
        
        if self.n == 1:
            ops.append(cirq.CNOT(temp_qubits[0], condition_qubit))
        elif self.n == 2:
            ops.append(cirq.CCNOT(temp_qubits[0], temp_qubits[1], condition_qubit))
        elif self.n >= 3:
            temp_and = self.aux_qubits[self.n + 4]
            ops.append(cirq.CCNOT(temp_qubits[0], temp_qubits[1], temp_and))
            if self.n == 3:
                ops.append(cirq.CCNOT(temp_and, temp_qubits[2], condition_qubit))
            else:
                ops.append(cirq.CCNOT(temp_and, temp_qubits[2], condition_qubit))
        
        combined_control = self.aux_qubits[self.n + 5] if len(self.aux_qubits) > self.n + 5 else condition_qubit
        if combined_control != condition_qubit:
            ops.append(cirq.CCNOT(control, condition_qubit, combined_control))
        else:
            combined_control = condition_qubit
        
        for i in range(self.n):
            if (result_val >> i) & 1:
                ops.append(cirq.CNOT(combined_control, self.work_qubits[i]))
        
        if combined_control != condition_qubit:
            ops.append(cirq.CCNOT(control, condition_qubit, combined_control))
        
        if self.n == 1:
            ops.append(cirq.CNOT(temp_qubits[0], condition_qubit))
        elif self.n == 2:
            ops.append(cirq.CCNOT(temp_qubits[0], temp_qubits[1], condition_qubit))
        elif self.n >= 3:
            temp_and = self.aux_qubits[self.n + 4]
            if self.n == 3:
                ops.append(cirq.CCNOT(temp_and, temp_qubits[2], condition_qubit))
            else:
                ops.append(cirq.CCNOT(temp_and, temp_qubits[2], condition_qubit))
            ops.append(cirq.CCNOT(temp_qubits[0], temp_qubits[1], temp_and))
        
        for i in range(self.n):
            if not ((target_val >> i) & 1):
                ops.append(cirq.X(temp_qubits[i]))
        
        return ops
    
    def controlled_modular_mult(self, control, multiplier):
        ops = []
        if multiplier == 1:
            return ops
        
        # Use simpler approach to avoid qubit conflicts
        if len(self.aux_qubits) < self.n + 20:
            return self.controlled_modular_mult_lookup(control, multiplier)
        
        # Simplified quantum multiplication
        multiplicand_qubits = self.aux_qubits[:self.n]
        carry_qubit = self.aux_qubits[self.n + 10]
        
        # Copy work register to multiplicand (controlled)
        for i in range(self.n):
            ops.append(cirq.CCNOT(control, self.work_qubits[i], multiplicand_qubits[i]))
        
        # Clear work register
        for i in range(self.n):
            ops.append(cirq.CNOT(control, self.work_qubits[i]))
        
        # Simple multiplication: add multiplicand 'multiplier' times
        for add_count in range(min(multiplier, 8)):  # Limit to avoid too many gates
            if add_count > 0:  # Skip first iteration (work is already 0)
                ops.extend(self.quantum_adder(multiplicand_qubits, self.work_qubits, carry_qubit))
                ops.extend(self.quantum_modular_reduction())
        
        # Uncompute multiplicand
        for i in range(self.n):
            ops.append(cirq.CNOT(control, multiplicand_qubits[i]))
            ops.append(cirq.CNOT(control, multiplicand_qubits[i]))
        
        return ops
    
    def controlled_modular_mult_lookup(self, control, multiplier):
        # Fallback lookup table method for when we don't have enough qubits
        ops = []
        temp_qubits = self.aux_qubits[:self.n]
        
        for i in range(self.n):
            ops.append(cirq.CCNOT(control, self.work_qubits[i], temp_qubits[i]))
        
        for i in range(self.n):
            ops.append(cirq.CNOT(control, self.work_qubits[i]))
        
        if multiplier < 16:
            for val in range(min(8, 2**self.n)):
                result = (val * multiplier) % self.N
                if result != val:
                    ops.extend(self.conditional_set_value(control, temp_qubits, val, result))
        
        for i in range(self.n):
            ops.append(cirq.CSWAP(control, self.work_qubits[i], temp_qubits[i]))
        
        for i in range(self.n):
            ops.append(cirq.CNOT(control, temp_qubits[i]))
            ops.append(cirq.CNOT(control, temp_qubits[i]))
        
        return ops
    
    def qft(self, qubits):
        ops = []
        n = len(qubits)
        
        for i in range(n):
            ops.append(cirq.H(qubits[i]))
            
            for j in range(i + 1, n):
                angle = 2 * np.pi / (2 ** (j - i + 1))
                ops.append(cirq.CZ(qubits[j], qubits[i]) ** (angle / np.pi))
        
        for i in range(n // 2):
            ops.append(cirq.SWAP(qubits[i], qubits[n - 1 - i]))
        
        return ops
    
    def build_circuit(self):
        circuit = cirq.Circuit()
        
        circuit.append(cirq.X(self.work_qubits[0]))
        
        for q in self.counting_qubits:
            circuit.append(cirq.H(q))
        
        for i, control_qubit in enumerate(self.counting_qubits):
            multiplier = self.powers[i]
            if multiplier != 1:
                circuit.append(self.controlled_modular_mult(control_qubit, multiplier))
        
        qft_ops = self.qft(self.counting_qubits)
        circuit.append(cirq.inverse(qft_ops))
        
        circuit.append(cirq.measure(*self.counting_qubits, key='phase'))
        circuit.append(cirq.measure(*self.work_qubits, key='modular_result'))
        
        return circuit

class ShorFactoring:
    def __init__(self, N):
        self.N = N
    
    def gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a
    
    def get_random_base(self):
        if self.N <= 3:
            return 2
        
        candidates = [2, 3, 5, 7, 11, 13]
        valid_bases = []
        
        for a in candidates:
            if a < self.N and self.gcd(a, self.N) == 1:
                valid_bases.append(a)
        
        if valid_bases:
            return random.choice(valid_bases)
        
        for _ in range(20):
            a = random.randint(2, max(2, self.N - 1))
            if self.gcd(a, self.N) == 1:
                return a
        
        return 2
    
    def quantum_period_finding(self, a):
        try:
            period_finder = QuantumPeriodFinder(self.N, a)
            circuit = period_finder.build_circuit()
            
            print(f"Circuit: {len(circuit)} moments, {len(circuit.all_qubits())} qubits")
            
            simulator = qsimcirq.QSimSimulator()
            result = simulator.run(circuit, repetitions=500)
            phase_measurements = result.measurements['phase']
            modular_measurements = result.measurements['modular_result']
            
            phase_counts = {}
            for measurement in phase_measurements:
                phase_int = int(''.join(map(str, measurement)), 2)
                phase_counts[phase_int] = phase_counts.get(phase_int, 0) + 1
            
            modular_counts = {}
            for measurement in modular_measurements:
                mod_result = int(''.join(map(str, measurement)), 2)
                if mod_result < self.N:
                    modular_counts[mod_result] = modular_counts.get(mod_result, 0) + 1
            
            print(f"Top phases: {dict(list(sorted(phase_counts.items(), key=lambda x: x[1], reverse=True))[:3])}")
            print(f"Modular results: {dict(list(sorted(modular_counts.items(), key=lambda x: x[1], reverse=True))[:3])}")
            
            candidate_periods = []
            sorted_phases = sorted(phase_counts.items(), key=lambda x: x[1], reverse=True)
            
            for phase_val, count in sorted_phases[:5]:
                if phase_val == 0:
                    continue
                
                frac = Fraction(phase_val, 2**period_finder.m).limit_denominator(self.N)
                period_candidate = frac.denominator
                
                if 1 < period_candidate <= self.N:
                    if pow(a, period_candidate, self.N) == 1:
                        candidate_periods.append((period_candidate, count))
                
                for multiplier in [2, 4]:
                    test_period = period_candidate * multiplier
                    if 1 < test_period <= self.N and pow(a, test_period, self.N) == 1:
                        candidate_periods.append((test_period, count // multiplier))
            
            if candidate_periods:
                candidate_periods.sort(key=lambda x: x[1], reverse=True)
                best_period = candidate_periods[0][0]
                print(f"Quantum period found: {best_period}")
                return best_period
            
            return None
            
        except Exception as e:
            print(f"Quantum simulation failed: {e}")
            return None
    
    def extract_factors(self, a, r):
        if r is None or r <= 1:
            print(f"Invalid period: {r}")
            return None
        
        if r % 2 != 0:
            print(f"Odd period {r}, trying {2*r}")
            r = 2 * r
            if r > self.N:
                print(f"Period {r} too large")
                return None
        
        if pow(a, r, self.N) != 1:
            print(f"Period verification failed: {a}^{r} mod {self.N} = {pow(a, r, self.N)}")
            return None
        
        x = pow(a, r // 2, self.N)
        print(f"x = {a}^{r//2} mod {self.N} = {x}")
        
        factor1 = self.gcd(x - 1, self.N)
        factor2 = self.gcd(x + 1, self.N)
        
        print(f"gcd({x}-1, {self.N}) = gcd({x-1}, {self.N}) = {factor1}")
        print(f"gcd({x}+1, {self.N}) = gcd({x+1}, {self.N}) = {factor2}")
        
        # Check each factor
        if 1 < factor1 < self.N:
            other = self.N // factor1
            print(f"Factor found: {factor1} × {other} = {factor1 * other}")
            return [factor1, other]
        
        if 1 < factor2 < self.N:
            other = self.N // factor2
            print(f"Factor found: {factor2} × {other} = {factor2 * other}")
            return [factor2, other]
        
        print(f"No non-trivial factors: {factor1}, {factor2}")
        return None
    
    def factor(self):
        print(f"Factoring N = {self.N} using Shor's algorithm")
        
        if self.N <= 1:
            return [self.N] if self.N > 0 else [1]
        
        if self.N == 2 or self.N == 3:
            return [self.N]
        
        sqrt_n = int(self.N ** 0.5)
        if sqrt_n * sqrt_n == self.N:
            return [sqrt_n, sqrt_n]
        
        for attempt in range(3):
            a = self.get_random_base()
            print(f"\nAttempt {attempt + 1}: base a = {a}")
            
            g = self.gcd(a, self.N)
            if g > 1:
                print(f"Found factor via gcd: {g}")
                return [g, self.N // g]
            
            r = self.quantum_period_finding(a)
            if r is None:
                continue
                
            factors = self.extract_factors(a, r)
            if factors:
                print(f"Success! Factors: {factors}")
                return sorted(factors)
        
        print("No factors found")
        return [self.N]

def main():
    N = 6  
    
    shor = ShorFactoring(N)
    factors = shor.factor()
    
    print(f"\nResult: {N} = ", end="")
    if len(factors) == 1:
        print(f"{factors[0]} (prime)")
    else:
        print(f"{factors[0]} × {factors[1]}")

if __name__ == "__main__":
    main()
