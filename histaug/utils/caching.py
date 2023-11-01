import pandas as pd
import functools
from pathlib import Path
import json
from typing import Union


def cached_df(args_to_table_name: callable, cache_dir: Union[Path, str] = Path("/app/results")):
    """Decorator for caching a dataframe to disk, and loading it if it exists."""

    cache_dir = Path(cache_dir)

    @functools.wraps(cached_df)
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            table_name = args_to_table_name(*args, **kwargs)
            if not (cache_dir / f"{table_name}.csv").exists():
                df = func(*args, **kwargs)
                df.to_csv(cache_dir / f"{table_name}.csv")
                with (cache_dir / f"{table_name}.json").open("w") as f:
                    # Write kwargs for reading
                    json.dump(
                        dict(
                            index_col=list(range(len(getattr(df.index, "levels", " ")))),
                            header=list(range(len(getattr(df.columns, "levels", " ")))),
                        ),
                        f,
                    )
            read_kwargs = dict()
            if (cache_dir / f"{table_name}.json").exists():
                with (cache_dir / f"{table_name}.json").open("r") as f:
                    read_kwargs = json.load(f)
            return pd.read_csv(cache_dir / f"{table_name}.csv", **read_kwargs)

        return wrapper

    return decorator
