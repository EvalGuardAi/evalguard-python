# Releasing the Python SDK

Published to PyPI as `evalguardai`.

## One-time setup

1. Claim the package on PyPI and configure **Trusted Publishing** for this repo:
   - https://pypi.org/manage/account/publishing/
   - Workflow filename: `publish-python-sdk.yml`
   - Environment name: `pypi`
2. Create the `pypi` environment in GitHub repo settings (no secrets needed — OIDC).

## Cutting a release

1. Bump `version` in `packages/python-sdk/pyproject.toml`.
2. Commit + push to `main`.
3. Tag the release:
   ```bash
   git tag python-sdk-v$(python -c "import tomllib,sys; print(tomllib.load(open('packages/python-sdk/pyproject.toml','rb'))['project']['version'])")
   git push origin --tags
   ```
4. The `publish-python-sdk.yml` workflow runs tests → builds sdist/wheel → publishes via OIDC.
5. Verify: `pip install evalguardai==<version>` from a clean venv.

## Dry-run

Use `workflow_dispatch` with `dry_run=true` to build + twine-check without publishing.
