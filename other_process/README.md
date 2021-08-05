# Get the Relation Between Objects by Attribute Representation

First, get the attribute representation of each class, don't forget change the variable `TRAIN_PATH`, `TEST_PATH` and `IMAGE_PATH` in the code:

```
python compute_presentation.py
```

Then, compute the relationship matrix:

```
python compute_distance.py --n-clusters <clusters number> --split split1
```