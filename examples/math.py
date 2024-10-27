from flowery import flowery

@flowery(verbose=True)
def fibonacci(n, memo={}):
    print(f"Calculating fibonacci({n})")
    if n in memo:
        print(f"Returning memoized value for {n}")
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    print(f"Computed fibonacci({n}): {memo[n]}")
    return memo[n]

if __name__ == '__main__':
    print(f"Result: {fibonacci(5)}")

