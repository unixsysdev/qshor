#!/usr/bin/env python3

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
        self.aux_qubits = [cirq.GridQubit(2, i) for i in range(max(8, self.n + 4))]
        
        self.powers = []
        power = self.a % self.N
        for k in range(self.m):
            self.powers.append(power)
            power = (power * power) % self.N
    
    def quantum_compare_geq(self, a_qubits, b_bits, result_qubit):
        ops = []
        n = len(a_qubits)
        
        if len(self.aux_qubits) < self.n + 7:
            return ops
            
        gt_qubits = self.aux_qubits[self.n + 5:self.n + 7]
        
        for i in range(n-1, -1, -1):
            if b_bits[i] == 0:
                ops.append(cirq.CNOT(a_qubits[i], result_qubit))
            else:
                if len(gt_qubits) > 0:
                    ops.append(cirq.X(a_qubits[i]))
                    ops.append(cirq.CNOT(a_qubits[i], gt_qubits[0]))
                    ops.append(cirq.X(a_qubits[i]))
        
        return ops
    
    def quantum_modular_reduction(self):
        ops = []
        if len(self.aux_qubits) < self.n + 6:
            return ops
            
        comparison_qubit = self.aux_qubits[self.n + 2]
        temp_qubit = self.aux_qubits[self.n + 3]
        
        N_bits = [(self.N >> i) & 1 for i in range(self.n)]
        
        ops.extend(self.quantum_compare_geq(self.work_qubits, N_bits, comparison_qubit))
        
        for i in range(self.n):
            if N_bits[i] == 1:
                ops.append(cirq.CNOT(comparison_qubit, self.work_qubits[i]))
        
        borrow_qubit = self.aux_qubits[self.n + 4] if len(self.aux_qubits) > self.n + 4 else temp_qubit
        for i in range(1, self.n):
            if N_bits[i-1] == 1:
                ops.append(cirq.X(self.work_qubits[i]))
                ops.append(cirq.CCNOT(comparison_qubit, self.work_qubits[i], borrow_qubit))
                ops.append(cirq.X(self.work_qubits[i]))
                
                ops.append(cirq.CNOT(borrow_qubit, self.work_qubits[i]))
                ops.append(cirq.CNOT(borrow_qubit, borrow_qubit))
        
        ops.extend(self.quantum_compare_geq(self.work_qubits, N_bits, comparison_qubit))
        
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
            return None
        
        if r % 2 != 0:
            r = 2 * r
            if r > self.N:
                return None
        
        if pow(a, r, self.N) != 1:
            return None
        
        x = pow(a, r // 2, self.N)
        print(f"x = {a}^{r//2} mod {self.N} = {x}")
        
        if x == 1 or x == self.N - 1:
            return None
        
        factor1 = self.gcd(x - 1, self.N)
        factor2 = self.gcd(x + 1, self.N)
        
        print(f"gcd({x}±1, {self.N}) = {factor1}, {factor2}")
        
        for factor in [factor1, factor2]:
            if 1 < factor < self.N:
                return [factor, self.N // factor]
        
        return None
    
    def factor(self):
        print(f"Factoring N = {self.N} using Shor's algorithm")
        
        if self.N <= 1:
            return [self.N] if self.N > 0 else [1]
        
        if self.N == 2 or self.N == 3:
            return [self.N]
        
        if self.N % 2 == 0:
            return [2, self.N // 2]
        
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
    N = 15
    
    shor = ShorFactoring(N)
    factors = shor.factor()
    
    print(f"\nResult: {N} = ", end="")
    if len(factors) == 1:
        print(f"{factors[0]} (prime)")
    else:
        print(f"{factors[0]} × {factors[1]}")

if __name__ == "__main__":
    main()
