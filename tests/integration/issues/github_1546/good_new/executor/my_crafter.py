from jina import Executor
from .helper import helper_function


class GoodCrafterNew(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(helper_function)
