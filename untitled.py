import os
import shutil
from typing import Any
import torch
from pathlib import Path
import numpy as np
import os
import random
import shutil


if __name__ == "__main__":
    import pdb

    class classA:
        def __init__(self, a):
            self.a = a
        
        def __call__(self, add) -> Any:
            self.a += add
            return self.a


    class MyClass:
        def __init__(self):
            self.my_list = [1, 2, 3]
            self.a = classA(3)

    my_obj = MyClass()
    my_list_copy = getattr(my_obj, "my_list")
    my_list_copy.append(4)

    a = getattr(my_obj, "a")
    # loz = a(1)


    # The original list has been modified
    print(my_obj.my_list)  # Output: [1, 2, 3, 4]
    print(my_obj.a.a)
    # pdb.set_trace()

    

