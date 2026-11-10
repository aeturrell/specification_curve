## subsets_by()


Builds a `subsets` argument with one sample per level of a column.


Usage

``` python
subsets_by(
    data,
    column,
    include_all=False,
)
```


## Parameters


`data: pd.DataFrame`  
The data.

`column: str`  
Column whose levels define the samples.

`include_all: bool = ``False`  
Also include the full sample, named "all".


## Returns


`dict[str, Optional[str]]`  
dict\[str, Optional\[str\]\]: Maps a name for each sample to its query.


## Examples


``` python
    >>> import pandas as pd
    >>> subsets_by(pd.DataFrame({"g": ["a", "b", "a"]}), "g")
```

    {'g = a': "`g` == 'a'", 'g = b': "`g` == 'b'"}
