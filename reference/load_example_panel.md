## load_example_panel()


Generates a synthetic firm-year panel.


Usage

``` python
load_example_panel(
    n_firms=300,
    n_years=8,
)
```


The true effect of `treated` on `log(sales)` is 0.3. Treatment is more likely for firms with a high unobserved firm effect, so specifications without firm fixed effects overstate it. `eligible` is a randomly assigned instrument for `treated`.

Columns: `firm`, `year`, `industry`, `state` (identifiers); `treated` (0/1), `eligible` (0/1); `size`, `age`, `leverage`, `rd` (controls; `age` and `leverage` have a few missing values); `sales` (positive), `patents` (count), `exporter` (0/1) (outcomes); `weight`. Note that `age` rises by one each year, so it cannot be separated from firm and year fixed effects used together.


## Parameters


`n_firms: int = ``300`  
Number of firms.

`n_years: int = ``8`  
Number of years, starting in 2010.


## Returns


`pd.DataFrame`  
pd.DataFrame: One row per firm and year.
