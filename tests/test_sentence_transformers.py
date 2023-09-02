from click.testing import CliRunner
from llm.cli import cli
import llm
import json


def test_model_installed():
    runner = CliRunner()
    result = runner.invoke(cli, ["embed-models"])
    assert result.exit_code == 0, result.output
    fragment = "sentence-transformers/all-MiniLM-L6-v2"
    assert fragment in result.output


def test_run_embedding():
    model = llm.get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    result = model.embed("hello world")
    assert len(result) == 384
    assert isinstance(result[0], float)


def test_cli_register(user_path):
    path = user_path / "sentence-transformers.json"
    assert not path.exists()
    runner = CliRunner()
    result = runner.invoke(
        cli, ["sentence-transformers", "register", "all-MiniLM-L12-v2", "--lazy"]
    )
    assert result.exit_code == 0, result.output
    assert "all-MiniLM-L12-v2" in json.loads((path).read_text("utf-8"))
