# Migrations

SQL migrations live in `src/frigate_learn/migrations/` and are applied by the
`frigate-learn db migrate` command (and automatically by `collect` / `db init`).
They are deliberately kept inside the Python package so the installed CLI can
always find them, no matter where it is run from.

To override (e.g. for a custom deployment), point `data.migrations` in the
config at a directory containing `NNNN_name.sql` files; the runner falls back to
the bundled scripts when the directory is empty/missing.

Manage with:

```bash
frigate-learn db migrate     # apply pending
frigate-learn db status      # show applied/pending + schema version
```