# Release checklist: v1.0 freeze (roadmap 4.7)

Check: every dissertation and paper number comes from the v1.0 tag.

1. All Phase 4 pull requests are merged into `main`, and CI on `main` is green on Linux and
   Windows (with the 80% coverage floor).
2. `git checkout main && git pull`, then confirm `pyproject.toml` says `version = "1.0.0"`.
3. `make check` passes locally.
4. Tag and push:

   ```bash
   git tag -a v1.0 -m "ShortcutCheck v1.0: frozen for the dissertation and paper"
   git push origin v1.0
   ```

5. Optional: create a GitHub release from the tag and archive it on Zenodo for a DOI (needed for
   the paper's data and code availability statement once the repository is public).
6. From now on, change results only to fix bugs. For every fix: a pull request, an entry under
   "Fixes after v1.0" in `CHANGELOG.md` listing the numbers it changes, and a new tag (v1.0.1, ...).
7. In the dissertation, cite the tag (and its commit) next to every table and figure source.
