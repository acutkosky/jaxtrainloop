import jax
from models.gpt import GPT
from models.resnet import resnet18
from models.alt_resnet import alt_resnet18
import equinox as eqx
from typing import Any, Tuple
from omegaconf import DictConfig


MODEL_REGISTRY = {}

def get_model(config: DictConfig, data_loaders: Any, *, key=jax.random.PRNGKey(0)) -> Tuple[eqx.Module, eqx.nn.State]:
    return MODEL_REGISTRY[config.model.name](config.model, data_loaders, key=key)

def load_gpt(config: DictConfig, data_loaders, *, key) -> Tuple[eqx.Module, eqx.nn.State]:
    return eqx.nn.make_with_state(GPT)(
        config, data_loaders["tokenizer"].vocab_size, key=key
    )


MODEL_REGISTRY["gpt"] = load_gpt


def load_resnet(
    config: DictConfig, dataloaders: Any, *, key
) -> Tuple[eqx.Module, eqx.nn.State]:
    if config.model == "torch_resnet18":
        model_fn = resnet18
    elif config.model == "kuangliu_resnet18" or config.model == "resnet18":
        model_fn = alt_resnet18
    model, state = eqx.nn.make_with_state(model_fn)(
        num_classes=config.model.num_classes, bn_momentum=config.model.bn_mom, key=key
    )
    return model, state
