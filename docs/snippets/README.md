# Executable README snippets

The Python fragments in this directory are generated from marked regions in
the executable public examples. The renderer verifies that each exact body is
present in `README.md`. It also executes the saved-estimator quickstart with
10,000 draws and seed 123, writes `quickstart_output.txt`, and checks that the
same table is displayed in the README. The fragments are checked documentation
inputs, not standalone programs.

Regenerate and verify them with:

```bash
python scripts/render_readme_examples.py --write
python scripts/render_readme_examples.py --check
```

Edit the source example rather than a generated fragment. CI rejects stale
fragments, which prevents documentation examples from drifting away from the
API exercised by the installed-wheel jobs.
