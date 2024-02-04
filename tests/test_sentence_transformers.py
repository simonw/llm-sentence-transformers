from click.testing import CliRunner
from llm.cli import cli
import llm
import json
import sqlite_utils


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
    assert not (user_path / "aliases.json").exists()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sentence-transformers",
            "register",
            "all-MiniLM-L12-v2",
            "--lazy",
            "--alias",
            "a1",
            "--alias",
            "a2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "all-MiniLM-L12-v2" in [
        m["name"] for m in json.loads((path).read_text("utf-8"))
    ]
    # And aliases should be set
    assert json.loads((user_path / "aliases.json").read_text("utf-8")) == {
        "a1": "sentence-transformers/all-MiniLM-L12-v2",
        "a2": "sentence-transformers/all-MiniLM-L12-v2",
    }
    for alias in ("a1", "a2"):
        assert (
            llm.get_embedding_model(alias).model_id
            == "sentence-transformers/all-MiniLM-L12-v2"
        )


def test_embed_multi_with_generator():
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection(
        name="test", db=db, model_id="sentence-transformers/all-MiniLM-L6-v2"
    )

    def generate():
        yield (1, "hello world")
        yield (2, "goodbye world")

    assert collection.count() == 0
    collection.embed_multi(generate())
    assert collection.count() == 2
