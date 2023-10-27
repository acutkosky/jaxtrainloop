import jax
from models.gpt import GPT
from models.resnet import resnet18
from models.alt_resnet import alt_resnet18
from models.linear import LinearModel
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
    if config.name == "torch_resnet18":
        model_fn = resnet18
    elif config.name in ["kuangliu_resnet18", "resnet18"]:
        model_fn = alt_resnet18
    model, state = eqx.nn.make_with_state(model_fn)(
        num_classes=config.num_classes, bn_momentum=config.bn_momentum, key=key
    )
    return model, state
for name in ["resnet18", "torch_resnet18", "kuangliu_resnet18"]:
    MODEL_REGISTRY[name] = load_resnet



def load_linear(config: DictConfig, dataloaders: Any, *, key) -> Tuple[eqx.Module, eqx.nn.State]:
    use_bias = config.use_bias
    zero_init = config.zero_init
    feature_dim = dataloaders['feature_dim']
    label_dim = dataloaders['num_classes']
    return eqx.nn.make_with_state(LinearModel)(feature_dim, label_dim, use_bias, zero_init, key=key)
MODEL_REGISTRY["linear"] = load_linear
