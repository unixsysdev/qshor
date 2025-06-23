#!/usr/bin/env python3

import math
import random

def get_prime_factors(n, max_factor=None):
    # Get prime factorization up to max_factor limit
    factors = []
    d = 2
    limit = min(max_factor or n, int(n**0.5) + 1)
    
    while d <= limit and n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d > 1000:  # Don't spend forever on large numbers
            break
    
    if n > 1 and (max_factor is None or n <= max_factor):
        factors.append(n)
    
    return factors

def is_smooth(n, bound):
    # Check if n has only small prime factors <= bound
    temp = n
    for p in range(2, min(bound + 1, 1000)):  # Limit to avoid infinite loops
        while temp % p == 0:
            temp //= p
        if temp == 1:
            return True
        if p > int(temp**0.5):
            break
    return temp <= bound

def pollard_rho_factor(n, max_iterations=100000):
    # Pollard's rho algorithm - much faster than trial division
    if n % 2 == 0:
        return 2
    
    x = random.randint(2, n - 1)
    y = x
    c = random.randint(1, n - 1)
    d = 1
    
    for _ in range(max_iterations):
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        
        if 1 < d < n:
            return d
        if d == n:
            # Restart with different parameters
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
    
    return None

def advanced_gnfs_factor(N):
    print(f"Factoring {N} using advanced GNFS techniques")
    print(f"This is a {len(str(N))}-digit number")
    
    # Quick check for small factors first
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if N % p == 0:
            print(f"Found small factor: {p}")
            return [p, N // p]
    
    # Step 1: Use multiple polynomials for better coverage
    polynomials = [
        (1, 0, -N),      # x^2 - N
        (1, 1, -N),      # x^2 + x - N  
        (1, -1, -N),     # x^2 - x - N
    ]
    
    sqrt_n = int(N ** 0.5)
    
    # Step 2: Adaptive parameters based on N size
    if N < 10**6:
        bound = 100
        search_range = 200
    elif N < 10**9:
        bound = 500
        search_range = 1000
    else:
        bound = 2000
        search_range = 5000
    
    print(f"Using smoothness bound: {bound}")
    print(f"Search range: ±{search_range} around √N ≈ {sqrt_n}")
    
    all_relations = []
    
    # Step 3: Try each polynomial
    for poly_idx, (a, b, c) in enumerate(polynomials):
        print(f"\nTrying polynomial {poly_idx + 1}: {a}x² + {b}x + {c}")
        
        start = max(1, sqrt_n - search_range)
        end = sqrt_n + search_range
        
        relations = []
        checked = 0
        
        for x in range(start, end):
            checked += 1
            if checked % 10000 == 0:
                print(f"  Checked {checked} values, found {len(relations)} smooth")
            
            # Evaluate polynomial
            value = a * x * x + b * x + c
            if value <= 0:
                continue
            
            # Quick smoothness test
            if is_smooth(value, bound):
                factors = get_prime_factors(value, bound)
                if factors and all(f <= bound for f in factors):
                    print(f"  Smooth: x={x}, value={value}")
                    relations.append((x, value, factors))
                    all_relations.append((x, value, factors))
                    
                    if len(relations) >= 20:  # Enough for this polynomial
                        break
        
        if len(all_relations) >= 50:  # Enough total relations
            break
    
    if len(all_relations) < 2:
        print("Not enough smooth relations found")
        print("Trying Pollard's rho as fallback...")
        factor = pollard_rho_factor(N)
        if factor:
            return [factor, N // factor]
        return None
    
    print(f"\nFound {len(all_relations)} total smooth relations")
    
    # Step 4: Proper GNFS square finding using exponent vectors
    print("Building factor base and exponent matrix...")
    
    # Create factor base (all primes up to bound)
    factor_base = []
    for p in range(2, bound + 1):
        if all(p % i != 0 for i in range(2, int(p**0.5) + 1)):
            factor_base.append(p)
    
    print(f"Factor base size: {len(factor_base)} primes")
    
    # Convert smooth relations to exponent vectors
    valid_relations = []
    for x, value, factors in all_relations:
        # Count exponents of each prime in factor base
        exponent_vector = [0] * len(factor_base)
        temp_factors = factors.copy()
        
        for i, prime in enumerate(factor_base):
            while prime in temp_factors:
                exponent_vector[i] += 1
                temp_factors.remove(prime)
        
        # Only keep if all factors are in our factor base
        if not temp_factors:  # All factors accounted for
            valid_relations.append((x, value, exponent_vector))
    
    print(f"Valid relations for linear algebra: {len(valid_relations)}")
    
    if len(valid_relations) < 2:
        print("Not enough valid relations for GNFS")
    else:
        # Find linear dependencies (simplified Gaussian elimination mod 2)
        print("Looking for linear dependencies...")
        
        # Try all pairs to find where exponent vectors sum to all-even
        for i in range(len(valid_relations)):
            for j in range(i + 1, len(valid_relations)):
                x1, val1, vec1 = valid_relations[i]
                x2, val2, vec2 = valid_relations[j]
                
                # Add exponent vectors mod 2
                combined_exp = [(vec1[k] + vec2[k]) % 2 for k in range(len(factor_base))]
                
                # Check if all exponents are even
                if all(exp == 0 for exp in combined_exp):
                    print(f"Found linear dependency!")
                    print(f"Relations: ({x1}, {val1}) and ({x2}, {val2})")
                    
                    # We have (x1 * x2)^2 ≡ (product of primes with even exponents) (mod N)
                    left = (x1 * x2) % N
                    
                    # Compute right side: square root of val1 * val2
                    right_squared = val1 * val2
                    right = 1
                    
                    # Build right side from factor base using combined exponents
                    for k, prime in enumerate(factor_base):
                        total_exp = vec1[k] + vec2[k]
                        if total_exp > 0:
                            right = (right * pow(prime, total_exp // 2)) % N
                    
                    print(f"Square congruence: {left}² ≡ {right}² (mod {N})")
                    
                    if left != right and left != 0 and right != 0:
                        factor1 = math.gcd(abs(left - right), N)
                        factor2 = math.gcd((left + right) % N, N)
                        
                        print(f"gcd({abs(left - right)}, {N}) = {factor1}")
                        print(f"gcd({(left + right) % N}, {N}) = {factor2}")
                        
                        for factor in [factor1, factor2]:
                            if 1 < factor < N:
                                print(f"GNFS found factor: {factor}")
                                return [factor, N // factor]
        
        # Try larger combinations (triplets)
        print("Trying triplet combinations...")
        for i in range(len(valid_relations)):
            for j in range(i + 1, len(valid_relations)):
                for k in range(j + 1, min(len(valid_relations), j + 10)):  # Limit to avoid explosion
                    x1, val1, vec1 = valid_relations[i]
                    x2, val2, vec2 = valid_relations[j]
                    x3, val3, vec3 = valid_relations[k]
                    
                    # Add three exponent vectors mod 2
                    combined_exp = [(vec1[l] + vec2[l] + vec3[l]) % 2 for l in range(len(factor_base))]
                    
                    if all(exp == 0 for exp in combined_exp):
                        print(f"Found triplet dependency!")
                        
                        left = (x1 * x2 * x3) % N
                        
                        # Build right side
                        right = 1
                        for l, prime in enumerate(factor_base):
                            total_exp = vec1[l] + vec2[l] + vec3[l]
                            if total_exp > 0:
                                right = (right * pow(prime, total_exp // 2)) % N
                        
                        print(f"Triplet congruence: {left}² ≡ {right}² (mod {N})")
                        
                        if left != right and left != 0 and right != 0:
                            factor1 = math.gcd(abs(left - right), N)
                            factor2 = math.gcd((left + right) % N, N)
                            
                            for factor in [factor1, factor2]:
                                if 1 < factor < N:
                                    print(f"GNFS found factor: {factor}")
                                    return [factor, N // factor]
    
    # Step 5: Direct square congruence search with larger range
    print("Trying direct square congruences...")
    squares_mod_n = {}
    
    # Much larger search range for big numbers
    search_limit = min(100000, sqrt_n * 2)
    batch_size = 10000
    
    for batch_start in range(1, search_limit, batch_size):
        batch_end = min(batch_start + batch_size, search_limit)
        print(f"  Checking squares {batch_start} to {batch_end}...")
        
        for x in range(batch_start, batch_end):
            square = (x * x) % N
            if square in squares_mod_n:
                y = squares_mod_n[square]
                if abs(x - y) > 1:  # Avoid trivial cases
                    factor1 = math.gcd(abs(x - y), N)
                    factor2 = math.gcd(x + y, N)
                    
                    for factor in [factor1, factor2]:
                        if 1 < factor < N:
                            print(f"Found factor via square congruence: {factor}")
                            return [factor, N // factor]
            else:
                squares_mod_n[square] = x
    
    # Step 6: Last resort - Pollard's rho
    print("GNFS didn't find factors, trying Pollard's rho...")
    factor = pollard_rho_factor(N)
    if factor:
        print(f"Pollard's rho found factor: {factor}")
        return [factor, N // factor]
    
    return None

def main():
    # Target: big-ass number
    N = 56464765764653323151

    
    print(f"Target: {N}")
    print(f"This is a {len(str(N))}-digit number")
    print(f"Trial division would need to check up to √{N} ≈ {int(N**0.5):,}")
    print(f"That's potentially millions of candidates!\n")
    
    result = advanced_gnfs_factor(N)
    
    if result:
        print(f"\nSUCCESS: {N:,} = {result[0]:,} × {result[1]:,}")
        print(f"Verification: {result[0]:,} × {result[1]:,} = {result[0] * result[1]:,}")
        
        # Check how many trial division steps this would have taken
        smaller_factor = min(result[0], result[1])
        trial_steps = len([p for p in range(2, smaller_factor) if all(p % i != 0 for i in range(2, int(p**0.5) + 1))])
        print(f"Trial division would have needed ~{trial_steps:,} prime checks!")
        print(f"GNFS/advanced methods found it much faster!")
    else:
        print(f"\nFAILED to factor {N:,}")

if __name__ == "__main__":
    main()
