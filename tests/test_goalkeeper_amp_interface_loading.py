import math

from booster_mjlab.amp.utils.interface_loading import (
    resolve_amp_callable,
)


resolved = resolve_amp_callable(
    "math:sqrt",
    field_name="expert_sampler_factory",
)

# 找到的是同一个函数对象，不是函数执行结果。
assert resolved is math.sqrt

# 真正调用发生在这里。
assert resolved(9.0) == 3.0

print("可调用对象解析检查通过")


cases = [
    (None, TypeError),
    ("", ValueError),
    ("math.sqrt", ValueError),
    ("math:", ValueError),
    (":sqrt", ValueError),
    ("math:pi", TypeError),
    ("math:not_an_existing_function", AttributeError),
    ("nonexistent_goalkeeper_module:factory", ImportError),
]

for path, expected_error in cases:
    try:
        resolve_amp_callable(
            path,
            field_name="motion_id_reader",
        )
    except expected_error:
        pass
    else:
        raise AssertionError(f"Invalid entry was accepted: {path!r}")

print("入口格式、对象类型和不存在入口检查通过")
