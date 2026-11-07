from collections.abc import Callable

from rsl_rl.utils import resolve_callable


def resolve_amp_callable(
    path: str,
    *,
    field_name: str,
) -> Callable:
    """Resolve a trusted AMP 'module:object' entry without calling it.

    Importing still executes module-level Python code. Keep the original import
    exception type and traceback, adding configuration context for diagnostics.
    """
    if not isinstance(path, str):
        raise TypeError(f"{field_name} must be a string path to a callable")

    if path.count(":") != 1:
        raise ValueError(f"{field_name} must use 'module:object' format")

    module_path, object_path = (part.strip() for part in path.split(":", 1))

    if not module_path or not object_path:
        raise ValueError(f"{field_name} must specify both module and object")

    try:
        target = resolve_callable(f"{module_path}:{object_path}")
    except (ImportError, AttributeError) as error:
        error.add_note(f"While resolving AMP {field_name}={path!r}")
        raise

    if not callable(target):
        raise TypeError(f"{field_name} must resolve to a callable")

    return target
