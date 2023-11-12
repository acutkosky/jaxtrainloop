from loader.c4_loader import get_c4_loader_next_token
import torchvision
import torchvision.transforms as transforms
import torch
from typing import Any, Optional, Callable
from omegaconf import DictConfig
from jax import tree_util as jtu
from jax import numpy as jnp
import jax
import transformers
import openml
import numpy as np
import os
import util
import duration

DATASET_REGISTRY = {}


def get_loader(config):
    return DATASET_REGISTRY[config.train.dataset](config)


def make_arrays(batch):
    return jtu.tree_map(jnp.asarray, batch)


def load_c4_data(config: DictConfig):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    dataset = {"tokenizer": tokenizer}
    splits = ["train", "validation"]
    for split in splits:
        dataset[split] = get_c4_loader_next_token(
            tokenizer,
            split=split,
            batch_size=config.train.batch_size,
            max_length=config.model.context_length,
            pad_to_multiple_of=config.model.context_length,
            num_workers=config.train.dataloader_workers,
            ds_path=config.train.data_path,
        )

    def batch_size_fn(batch):
        target = batch["labels"]
        input = batch["input_ids"]
        tok = util.count_tokens(batch["attention_mask"])
        ex = target.shape[0]
        it = 1
        return duration.TrainDuration(it=it, ex=ex, tok=tok)

    dataset["batch_size_fn"] = batch_size_fn
    return dataset


DATASET_REGISTRY["c4"] = load_c4_data


def load_cifar_data(config: DictConfig):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.dataloader_workers,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.dataloader_workers,
    )

    def batch_size_fn(batch):
        input = batch[0]
        target = batch[1]
        ex = target.shape[0]
        it = 1
        return duration.TrainDuration(it=it, ex=ex)

    data = {
        "train": trainloader,
        "validation": testloader,
        "batch_size_fn": batch_size_fn,
    }
    return data


DATASET_REGISTRY["cifar10"] = load_cifar_data


def normalize_labels(labels):
    """
    make the set of possible labels be consecutive integers
    starting with 0.
    """
    values, indices = np.unique(labels, return_inverse=True)
    values = np.arange(len(values), dtype=np.int32)
    return values[indices]


def load_libsvm_data(config: DictConfig):
    if config.train.cache_dir is not None:
        openml.config.set_root_cache_directory(config.train.cache_dir)

    dataset_id = LIBSVM_TO_OPENML[config.train.dataset]

    dataset = openml.datasets.get_dataset(dataset_id=dataset_id)
    dataset = dataset.get_data()[0].to_numpy()
    key = jax.random.PRNGKey(123)

    np.random.shuffle(dataset)

    # dataset = np.random.shuffle(dataset.get_data()[0].to_numpy())

    features = dataset[:, :-1].astype(np.float32)
    feature_dim = np.shape(features)[1]
    features = torch.tensor(features)

    labels = dataset[:, -1].astype(np.int32)
    labels = normalize_labels(labels)
    num_classes = np.max(labels) + 1
    labels = torch.tensor(labels)

    length = len(labels)
    # reserve 10% for validation
    train_len = int(length * 0.9)

    train_features = features[:train_len]
    valid_features = features[train_len:]

    train_labels = labels[:train_len]
    valid_labels = labels[train_len:]

    trainset = torch.utils.data.TensorDataset(train_features, train_labels)

    validset = torch.utils.data.TensorDataset(valid_features, valid_labels)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.dataloader_workers,
    )
    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.dataloader_workers,
    )

    def batch_size_fn(batch):
        input = batch[0]
        target = batch[1]
        ex = target.shape[0]
        it = 1
        return duration.TrainDuration(it=it, ex=ex)

    result = {
        "train": trainloader,
        "validation": validloader,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "batch_size_fn": batch_size_fn,
    }
    return result


LIBSVM_TO_OPENML = {
    "aloi": 42396,
    "connect-4": 1591,
    "covertype": 150,
    "dna": 40670,
    "poker": 1595,
    "duke_breast_cancer": 1434,
    "vehicle_sensIT": 357,
    "news20": 1594,
    "Australian": 40981,
    "rcv1_binary": 1577,
    "ijcnn1": 1575,
    "vowel": 1016,
    "glass": 1005,
}
for name in LIBSVM_TO_OPENML:
    DATASET_REGISTRY[name] = load_libsvm_data


def load_constant(config: DictConfig):
    dim = config.train.dimension

    distance = config.train.starting_distance

    target = np.ones((1, dim)) / np.sqrt(dim) * distance

    dataset = torch.utils.data.TensorDataset(torch.tensor(target))
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.train.batch_size
    )

    def batch_size_fn(batch):
        ex = batch.shape[0]
        it = 1
        return duration.TrainDuration(it=it, ex=ex)

    result = {"train": train_loader, "batch_size_fn": batch_size_fn}

    return result


DATASET_REGISTRY["constant"] = load_constant
