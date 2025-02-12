import contextlib
import llm
import logging
from sentence_transformers import SentenceTransformer
import textwrap
import click
import json


def models_path():
    return llm.user_dir() / "sentence-transformers.json"


def read_models():
    sentence_transformers_path = models_path()
    if not sentence_transformers_path.exists():
        llm.user_dir().mkdir(exist_ok=True, parents=True)
        sentence_transformers_path.write_text(
            json.dumps(
                [
                    {
                        "name": "all-MiniLM-L6-v2",
                    }
                ],
                indent=2,
            ),
            "utf-8",
        )
    data = json.loads(sentence_transformers_path.read_text("utf-8"))
    fixed = []
    for item in data:
        if isinstance(item, str):
            fixed.append({"name": item})
        else:
            fixed.append(item)
    return fixed


def write_models(models):
    sentence_transformers_path = models_path()
    with sentence_transformers_path.open("w", encoding="utf-8") as f:
        json.dump(models, f, indent=2)


@llm.hookimpl
def register_embedding_models(register):
    for model in read_models():
        name = model["name"]
        register(
            SentenceTransformerModel(
                f"sentence-transformers/{name}",
                name,
                model.get("trust_remote_code", False),
            ),
            aliases=None,
        )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def sentence_transformers():
        "Commands for managing sentence-transformers embedding models"

    @sentence_transformers.command(
        help=textwrap.dedent(
            """
        Download and register a model by name.

        Try one of these names:
        
        \b
            all-mpnet-base-v2 - 420MB
            all-MiniLM-L12-v2 - 120MB
        """
        )
    )
    @click.argument("name")
    @click.option(
        "aliases", "-a", "--alias", multiple=True, help="Alias to use for this model"
    )
    @click.option(
        "--lazy", is_flag=True, help="Don't download the model until it is first used"
    )
    @click.option(
        "--trust-remote-code",
        is_flag=True,
        help="Set trust_remote_code=True for this model",
    )
    def register(name, aliases, lazy, trust_remote_code):
        llm.user_dir().mkdir(exist_ok=True, parents=True)
        current = read_models()
        current_names = [model["name"] for model in current]
        full_name = f"sentence-transformers/{name}"
        if name in current_names:
            # Set the aliases anyway
            for alias in aliases:
                llm.set_alias(alias, full_name)
            raise click.ClickException(f"Model {name} is already registered")
        current.append({"name": name, "trust_remote_code": trust_remote_code})
        write_models(current)
        if not lazy:
            model = llm.get_embedding_model(full_name)
            model.embed("hello world")
        for alias in aliases:
            llm.set_alias(alias, full_name)


class SentenceTransformerModel(llm.EmbeddingModel):
    def __init__(self, model_id, model_name, trust_remote_code):
        self.model_id = model_id
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self._model = None

    def embed_batch(self, texts):
        with disable_logging():
            try:
                if self._model is None:
                    self._model = SentenceTransformer(
                        self.model_name, trust_remote_code=self.trust_remote_code
                    )
                results = self._model.encode(list(texts))
                return [list(map(float, result)) for result in results]
            except ImportError as ex:
                s = str(ex)
                if "Run `pip install" in s:
                    try:
                        package = s.split("Run `pip install ")[1].split("`")[0]
                        raise ImportError(
                            "Install the missing package with `llm install "
                            + package
                            + "`"
                        )
                    except IndexError:
                        # Raise the original
                        raise ex
                else:
                    raise


@contextlib.contextmanager
def disable_logging(level=logging.WARNING):
    # Save the current disable level and the current root logger level.
    prev_disable = logging.root.manager.disable
    root_logger = logging.getLogger()
    prev_level = root_logger.level

    # Disable logging up to the specified level.
    logging.disable(level)
    try:
        yield
    finally:
        # Restore the previous disable level and logger level.
        logging.disable(prev_disable)
        root_logger.setLevel(prev_level)
