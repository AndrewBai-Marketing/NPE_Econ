# Security

Do not upload secrets, access tokens, private data, proprietary scanner data,
or institution-specific paths to this repository.

`structnpe` examples are synthetic and private-data-free. Real projects should
keep raw data outside the repository and pass paths through local config files.

## Public-beta artifacts

The 0.1 beta estimator and result formats are checksummed directories composed
of canonical JSON and non-object NPZ/NPY arrays. Their loaders verify the
completeness marker and file checksums and do not use pickle. Checksums detect
accidental or post-creation changes; they do not authenticate who created an
artifact, make arbitrary content trustworthy, or provide a resource-usage
sandbox. The beta loader does impose finite limits on manifest size, payload
count and bytes, NPZ members and decompression, NPY headers, and array elements;
over-limit artifacts fail closed.

Built-in `ArrayAdapter` and `SummaryAdapter` artifacts can be reconstructed
without importing artifact-selected code. For a custom adapter, the preferred
path is to pass a compatible `StructuralModel`, whose adapter code the caller
has already imported and reviewed. Loading a custom adapter without that model
requires the explicit `allow_custom_adapter=True` trust opt-in. That option may
import and initialize code named by the artifact and must be used only for
artifacts and installed adapter modules you trust.

## Legacy formats

The retained alpha project, simulation-bank, posterior, and estimator NPZ paths
are trusted-local compatibility workflows. Some use NumPy object loading and
therefore may invoke pickle semantics. Never give an untrusted legacy NPZ or
pickle file to those APIs. Legacy files are not beta estimator bundles, and a
checksum-free alpha file does not gain the beta security properties by being
renamed.

If you find a security issue, report it privately to the repository maintainer
rather than opening a public issue with sensitive details. Include the affected
version and a minimal private reproduction, but do not attach secrets, private
data, or an untrusted artifact to a public report.
