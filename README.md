# GPU Lock Benchmarking

Evaluates the performance and correctness of GPU locking mechanisms under concurrency stress. See: https://johnwickerson.github.io/papers/gpuconcurrency.pdf

## Overview

We implement five different types of GPU locks and run them across two benchmark modes:
- **Low complexity**: small critical sections
- **High complexity**: large critical sections

Each test is executed 100,000 times on multiple GPU architectures (pre- and post-2016) to measure:
- **Correctness**: whether blocks are accurately counted
- **Performance**: execution time under contention

## Lock Types

The following lock types are implemented:
- `spinlock.h`
- `ticketlock.h`
- `andersonlock.h`
- `mcslock.h`
- `spinspinlock.h`

## Files

| File                        | Description |
|----------------------------|-------------|
| `basicAtomicAddition.cu`   | Baseline using atomic addition (no locks) |
| `lowComplexityCritical.cu` | Runs all locks on small critical sections |
| `highComplexityCritical.cu`| Runs all locks on large critical sections |
| `*.h` files                | Lock implementations |
| `Scalable Lock...pdf`      | Reference paper |
| `README.md`                | Project documentation |

## Compile Instructions

To compile any `.cu` file:
```bash
nvcc -o [filename] [filename].cu
```
Replace `[filename]` with the name of the file, e.g.:
```bash
nvcc -o lowComplexityCritical lowComplexityCritical.cu
```

## Run Instructions

To execute the compiled file:
```bash
./[filename]
```

Example:
```bash
./lowComplexityCritical
```

## Output

Each run prints:
- Lock type used
- Number of blocks counted (correctness)
- Time taken (performance)

## Motivation

- Quantify weak behaviors in older vs. newer GPU architectures
- Compare lock performance and fairness under high contention
- Analyze whether newer GPUs show fewer deviations under lock stress

## Key Findings (Expected)

- Newer GPUs should reduce weak behaviors
- Spin locks may show worse performance due to lack of fairness
- Ticket locks may reduce deviations under high load

## Authors

- Arnav Gattani  
- Akash Anickode  
