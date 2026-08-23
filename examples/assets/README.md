# Bundled demonstration estimator

`demo_estimator/` is used only by `examples/quickstart.py`. It was trained on
synthetic panels from a five-state replacement model and contains no empirical
or private data. In particular, it was not trained on the Rust bus panel.

The artifact contains JSON metadata, numeric NumPy arrays, checksums, and a
completion marker. Loading it does not require Torch or pickle. It is tied to
the exact two-parameter model and ten-coordinate observation representation
recorded in its manifest; it cannot analyze unrelated data.
