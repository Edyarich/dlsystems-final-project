import dill
from typing import Any
from .nn import STATE_DICT_T


def load(filename: str) -> Any:
    with open(filename, 'rb') as fd:
        data = dill.load(fd)

    return data


def save(state_dict: STATE_DICT_T, filename: str):
    with open(filename, 'wb') as fd:
        dill.dump(state_dict, fd)