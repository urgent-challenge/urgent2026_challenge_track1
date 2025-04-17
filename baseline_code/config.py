import json
import os
from typing import Any, Dict, Optional, Type, TypeVar


class Config():

    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
    

