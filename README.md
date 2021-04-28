# Put panda on your head.

## Requirements

python 3.*, numpy, cv2, 
## Installation

```
pip install panda_producer
pip install panda_producer -i https://pypi.python.org/simple
```

## Usage example:

```

import cv2
from panda_producer import gen_panda_head

front_img = cv2.imread("front.jpg", 0)
res = gen_panda_head(front_img, "text", hyper=128)
```



