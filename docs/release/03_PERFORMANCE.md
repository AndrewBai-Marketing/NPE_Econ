# Performance benchmark

Status: component timing record; **no speed-superiority claim**

This note reports the measurements in
[`benchmark_results.json`](../../benchmarks/benchmark_results.json), generated
by [`benchmark_inference.py`](../../benchmarks/benchmark_inference.py). The
benchmark is deliberately transparent: it uses the synthetic
conjugate-normal example and compares the fitted MDN with a closed-form
posterior calculation.

Recorded `benchmark_results.json` SHA-256:
`14236ee3ba2af2f5cbd6f173555aee82e2bcc5ab10084e6ffe364859ec8daaa6`.

## Environment and workload

| item | recorded value |
| --- | --- |
| operating system | Linux, kernel `5.14.0-687.38.1.el9_8.x86_64`, glibc 2.34 |
| machine / processor | x86_64 / x86_64 |
| logical CPUs | 16 |
| Python | 3.13.11 |
| NumPy | 2.4.2 |
| training / inference device | CPU / CPU |
| estimator | five-component diagonal-Gaussian MDN |
| simulation budget | 2,500 |
| requested epochs | 35 |
| seed | 61001 |
| raw dataset shape | `(20,)` |
| representation dimension | 1 |
| parameter dimension | 1 |
| posterior draws | 10,000 per dataset |
| batch benchmark | 32 observed datasets |
| timing repeats | 7, except 70 for the analytic comparator |

Each raw dataset contains 20 scalar observations. The observation adapter
reduces it to the one-dimensional sample mean. The batch uses 32 separately
generated observed datasets; each receives 10,000 posterior draws.

## Recorded component timings

The following one-off timings have no repeated-run variability estimate:

| component | seconds | definition |
| --- | ---: | --- |
| simulation | 0.043913 | simulator-bank generation reported by training metadata |
| training, excluding simulation | 52.188520 | training time reported separately from simulation |
| fit wall time | 52.243224 | complete measured `fit` call, including simulation and training |
| newly loaded estimator: load | 0.002734 | load a new estimator object in the existing Python process |
| newly loaded estimator: first inference | 0.000854 | first 10,000-draw inference on that object |
| newly loaded estimator: combined | 0.003588 | preceding load plus first inference |

“Newly loaded” is not a cold-process measurement. It means a new estimator
object inside an already running Python interpreter; the OS page cache was not
controlled. Warm inference below excludes estimator loading.

Repeated timings report the median, median absolute deviation (MAD), minimum,
and maximum:

| operation | repeats | median (s) | MAD (s) | min (s) | max (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| estimator load in warm process | 7 | 0.00290323 | 0.00021868 | 0.00268046 | 0.00348848 |
| one-dataset warm inference, 10,000 draws | 7 | 0.00056961 | 0.00000735 | 0.00053738 | 0.00065188 |
| 32-dataset warm batch inference, 10,000 draws each | 7 | 0.01288302 | 0.00008302 | 0.01280000 | 0.01394765 |
| posterior summary table | 7 | 0.00081517 | 0.00003298 | 0.00076825 | 4.84225928 |
| analytic posterior mean and SD | 70 | 0.000002418 | 0.000000037 | 0.000002306 | 0.000032519 |

The 4.8423-second summary maximum is far above its sub-millisecond median.
This pattern is consistent with, and likely explained by, the first summary
call paying the lazy pandas import and initialization cost. The benchmark did
not instrument pandas import separately, so that explanation is an inference
rather than a measured timing decomposition.

The inference result also records preprocessing and posterior-draw components
for the last repetition:

| last repetition | preprocessing (s) | posterior draws (s) |
| --- | ---: | ---: |
| one dataset | 0.00001650 | 0.00045861 |
| batch of 32 | 0.00025738 | 0.01205399 |

These component values are single last-run observations, not medians. The
corresponding end-to-end inference medians also include remaining method and
result-construction overhead.

## Analytic comparator and break-even

The comparator has prior `Normal(0, 1)` and likelihood consisting of 20 iid
`Normal(theta, 1)` observations. It computes only the closed-form posterior
mean and standard deviation; it does not sample a posterior and has no MCMC
effective sample size. For the benchmark observation it returned mean
0.26881091 and SD 0.21821789.

Its median time was approximately 2.42 microseconds, versus approximately
569.61 microseconds for warm MDN inference with 10,000 draws for one dataset.
Consequently the declared break-even formula,

```text
training_seconds /
    (comparator_seconds_per_dataset - amortized_seconds_per_dataset)
```

has a negative denominator. There is **no finite break-even** against this
analytic comparator: the exact formula remains faster per dataset even before
the roughly 52.2-second fit cost is amortized.

This is also not an equal-output speed contest. The analytic path returns two
closed-form moments, while the MDN path generates 10,000 joint posterior
draws. The result supports neither an unconditional speedup claim nor a claim
that amortized neural inference is preferable when a closed-form posterior is
available.

## Reproduction

From the repository root, the recorded configuration is:

```bash
python benchmarks/benchmark_inference.py \
  --simulations 2500 \
  --epochs 35 \
  --draws 10000 \
  --batch-size 32 \
  --repeats 7 \
  --seed 61001 \
  --output benchmarks/benchmark_results.json \
  --quiet
```

## Limitations

- This is one synthetic, scalar-parameter conjugate model on one CPU machine;
  it does not establish scaling to realistic structural simulators,
  higher-dimensional posteriors, accelerators, or other hardware.
- Simulation, training, and the newly loaded first-inference components were
  measured once. Only the explicitly repeated rows have median and MAD
  estimates.
- “Newly loaded” does not launch a fresh interpreter, clear filesystem caches,
  or control other OS activity. The repeated load measurement is likewise
  process-resident.
- The preprocessing and draw splits are from the last repetition only.
- The summary outlier explanation is inferred; lazy-import cost was not
  isolated in a separate benchmark stage.
- The analytic comparator performs less output work than posterior sampling.
  It is useful as an honesty check, not as a representative conventional
  inference workload.
- Wall-clock results can change with package versions, CPU scheduling, thread
  settings, power management, and cache state. No memory-use or controlled
  throughput benchmark was recorded.
- These measurements do not demonstrate posterior accuracy, structural
  identification, model validity, or universal speed superiority.
