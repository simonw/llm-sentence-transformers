from click.testing import CliRunner
from llm.cli import cli


def test_model_installed():
    runner = CliRunner()
    result = runner.invoke(cli, ["embed-models"])
    assert result.exit_code == 0, result.output
    fragment = "sentence-transformers/all-MiniLM-L6-v2"
    assert fragment in result.output
