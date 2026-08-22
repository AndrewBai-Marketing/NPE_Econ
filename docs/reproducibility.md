# Reproducibility

Every stochastic public operation accepts an explicit seed. Training derives
separate simulation, split, optimization, and support-reference streams from
the supplied seed; inference derives a separate posterior-draw stream.

Save these together:

- the estimator artifact and its checksums;
- training and validation configurations;
- training metadata and loss history;
- simulation and inference seeds;
- package, Python, NumPy, and Torch versions where applicable;
- operating system, CPU/GPU, and device selection;
- observation shape, batch size, and requested draw count;
- validation and benchmark outputs.

The estimator fingerprint covers ordered parameter definitions, transforms,
adapter configuration, architecture, package compatibility line, and declared
prior/simulator identifiers and configurations. It prevents obvious cross-model
mixups; it is not a perfect hash of arbitrary source-code semantics.

Beta estimator and result artifacts use structured JSON and non-object NPZ/NPY
arrays with SHA-256 checksums and a completeness marker. Loaders do not use
pickle. Missing files, checksum failures, unsupported schemas, and shape
mismatches are hard errors.

Built-in adapters load without artifact-directed imports. For a custom adapter,
record the reviewed adapter package and version and prefer loading with a
compatible `StructuralModel`. The model-free `allow_custom_adapter=True` path
may import code selected by the artifact; it is an explicit trusted-artifact
option and emits a warning.

Fixed seeds support repeatable draws within a compatible environment. They do
not guarantee identical trained weights across Torch versions, hardware, thread
settings, or accelerators. Record such differences instead of claiming full
determinism.

The public source export is prepared as a fresh repository rather than
inheriting unrelated research history. Release provenance must use the exact
committed source identifier recorded by the release workflow. The external
release manifest records that commit and the hashes of artifacts built from
it; embedding an archive's own hash inside that archive would be circular.
Do not invent a commit, tag, remote, or release identity.
