"""Test-wide setup. Runs before any test module imports cortex."""

import os

# Never read the repo's .env in tests: a developer's real InsForge or OpenRouter credentials would otherwise
# make tests talk to the shared database (and fail on its data). Tests that need a provider set it explicitly.
os.environ["CORTEX_DOTENV"] = "0"
