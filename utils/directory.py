'''directory.py'''

from pathlib import Path


cwd         = Path(__file__).resolve().parent
top         = cwd.parent
vocab       = top/'vocab'
data        = top/'data'
formatted   = data/'formatted'
annotations = formatted/'annotations'
images      = formatted/'images'
embedding   = top/'vocab'/'embedding'
models      = top/'models'
