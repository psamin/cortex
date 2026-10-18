from types import SimpleNamespace

import huggingface_hub.constants as hfc

from cortex.services.import_service import hf_token


def ctx(env_token):
    return SimpleNamespace(settings=SimpleNamespace(hf_token=env_token))


def test_local_login_wins_over_env_and_env_is_the_fallback(tmp_path, monkeypatch):
    login = tmp_path / "token"
    monkeypatch.setattr(hfc, "HF_TOKEN_PATH", str(login))
    assert hf_token(ctx("from-dotenv")) == "from-dotenv"  # no login yet
    assert hf_token(ctx(None)) is None
    login.write_text("from-login\n")
    assert hf_token(ctx("from-dotenv")) == "from-login"  # the user's own account, what LeRobot uses
