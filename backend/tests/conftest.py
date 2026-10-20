"""Test-wide setup. Runs before any test module imports cortex."""

import os

# Never read the repo's .env in tests: a developer's real InsForge or OpenRouter credentials would otherwise
# make tests talk to the shared database (and fail on its data). Tests that need a provider set it explicitly.
os.environ["CORTEX_DOTENV"] = "0"


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _cluster_tools_present(monkeypatch):
    """Tests fake the cluster (`_sh`), so launches must not depend on ssh/rsync being installed on the test
    machine. test_training_transfer.py checks the real preflight."""
    from cortex.services import training_service

    monkeypatch.setattr(training_service, "preflight_transfer_tools", lambda: None)
