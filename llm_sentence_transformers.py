import llm
from sentence_transformers import SentenceTransformer
import textwrap
import click
import json


@llm.hookimpl
def register_embedding_models(register):
    sentence_transformers_path = llm.user_dir() / "sentence-transformers.json"
    if not sentence_transformers_path.exists():
        llm.user_dir().mkdir(exist_ok=True, parents=True)
        sentence_transformers_path.write_text(json.dumps(["all-MiniLM-L6-v2"]), "utf-8")
    models = json.loads(sentence_transformers_path.read_text("utf-8"))
    for model in models:
        register(
            SentenceTransformerModel(f"sentence-transformers/{model}", model),
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
    def register(name, aliases, lazy):
        llm.user_dir().mkdir(exist_ok=True, parents=True)
        sentence_transformers_path = llm.user_dir() / "sentence-transformers.json"
        if not sentence_transformers_path.exists():
            sentence_transformers_path.write_text("[]", "utf-8")
        current = json.loads(sentence_transformers_path.read_text("utf-8"))
        full_name = f"sentence-transformers/{name}"
        if name in current:
            # Set the aliases anyway
            for alias in aliases:
                llm.set_alias(alias, full_name)
            raise click.ClickException(f"Model {name} is already registered")
        current.append(name)
        sentence_transformers_path.write_text(json.dumps(current), "utf-8")
        if not lazy:
            model = llm.get_embedding_model(full_name)
            model.embed("hello world")
        for alias in aliases:
            llm.set_alias(alias, full_name)


class SentenceTransformerModel(llm.EmbeddingModel):
    def __init__(self, model_id, model_name):
        self.model_id = model_id
        self.model_name = model_name
        self._model = None

    def embed_batch(self, texts):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        results = self._model.encode(list(texts))
        return [list(map(float, result)) for result in results]
