# Reflection

## What steps did you take to complete the project?

1. Read through `King-County.ipynb` end to end and grouped its data cleaning, feature
   engineering, and modeling cells into discrete, named operations (e.g. "drop the
   33-bedroom row", "fill missing `waterfront`", "add distance to the wealth center").
2. Extracted each operation into a small, pure function that takes a DataFrame and
   returns a new one, organized into three modules under `src/house_pipeline/`:
   `cleaning.py`, `features.py`, and `modeling.py`, composed by `pipeline.py`.
3. Wrote unit tests for every function against small synthetic DataFrames
   (`tests/test_cleaning.py`, `tests/test_features.py`), plus an integration test that
   runs the full pipeline against the real dataset (`tests/test_pipeline.py`).
4. Built two CLI entry points (`scripts/build_dataset.py`, `scripts/train_model.py`) so
   the pipeline and training run outside the notebook, and confirmed they reproduce the
   notebook's numbers (21,596 rows after cleaning, ~84% test R²).
5. Rewired the notebook itself to call into `house_pipeline` instead of duplicating the
   logic inline, then executed it top to bottom to confirm the refactor didn't change
   its behavior or its results.
6. Extended the refactor to the modeling step (stretch goal): baseline linear
   regression, the polynomial/ElasticNet pipeline, grid search tuning, and model
   persistence all moved into `house_pipeline.modeling`, with tests covering each piece
   including a save/load round-trip.
7. Built the FastAPI CRUD app (stretch goal) in `api/`, using `bonus_solution/` as a
   reference: five features per house (`bedrooms`, `bathrooms`, `sqft_living`, `grade`,
   `zipcode`), backed by Postgres via SQLAlchemy, with Pydantic request/response
   validation. Added `tests/test_api.py`, which runs the full CRUD flow against SQLite
   so it needs no external database.
8. Added a `Dockerfile` and `docker-compose.yaml` for the API + Postgres, and validated
   the compose configuration resolves correctly (`docker compose config`).

## What challenges did you face?

- **Keeping the notebook narrative intact while removing duplication.** Several cells
  built one result across two or three notebook cells (e.g. the `last_known_change`
  loop, the water-distance loop). Merging those into single pipeline calls meant
  carefully renumbering and deleting cells with `nbformat` rather than editing by hand,
  to avoid breaking the notebook's JSON structure.
- **Preserving the exact cleaning/feature-engineering semantics.** The original
  water-distance calculation used a Python loop with a per-reference-point latitude
  correction (`cos(radians(ref_lat))`). Vectorizing it with NumPy broadcasting needed
  care to apply that correction per reference column, not once globally, so the
  refactored version produces the same distances as the original loop.
- **`skops`'s stricter loading API.** `sio.load(path, trusted=True)` from the notebook
  no longer works on the installed `skops` version (it now requires an explicit list of
  trusted types, per CVE-2024-37065). `house_pipeline.modeling.load_model` now calls
  `get_untrusted_types` and passes that list along, which is also the safer pattern.
- **Making the API testable without Postgres.** `bonus_solution`'s API requires a
  running Postgres instance. For fast, isolated tests, `tests/conftest.py` points
  `DB_CONN` at a temporary SQLite file before `api.database` is imported, so the same
  app code runs against SQLite in tests and Postgres in Docker.
- **No running Docker daemon in this environment**, so the compose stack's config was
  validated with `docker compose config` (env substitution, service wiring) and the
  Dockerfile was checked against `bonus_solution`'s working equivalent, but a live
  container run was not possible here.

## What would you do differently if you had more time?

- Add a data-validation layer (e.g. a `pandera` schema) at the pipeline's input/output
  boundaries, so malformed source data fails fast with a clear error instead of
  producing silently wrong features downstream.
- Wire a `/predict` endpoint into the FastAPI app that loads `model/model.bin` and
  scores a house on request, rather than keeping the CRUD API and the trained model
  fully separate.
- Turn `scripts/build_dataset.py` and `scripts/train_model.py` into a single small CLI
  (e.g. with `typer`) with proper `--help` output and configurable paths/hyperparameters,
  instead of positional `sys.argv` handling.
- Add a CI workflow (the repo already has `.github/pull_request_template.md`, but no
  GitHub Actions) that runs `uv run pytest` and builds the Docker image on every PR.
