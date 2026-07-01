"""``evalguardai`` -- import alias that matches the PyPI distribution name.

``pip install evalguardai`` exposes the SDK as **both** ``evalguard`` (the
historical module name) **and** ``evalguardai`` (this module), so the import
name can match the install name::

    pip install evalguardai
    from evalguardai import EvalGuard          # matches the install name

``import evalguard`` keeps working unchanged for existing code. This is a full
module alias: the package **and every already-loaded submodule** are bound to
the *same* objects as ``evalguard.*``, so attributes, submodules, and class
identities are shared::

    from evalguardai import EvalGuard, EvalGuardError
    from evalguardai.tracing import traceable, trace
    # isinstance(err, evalguardai.EvalGuardError) == isinstance(err, evalguard.EvalGuardError)

Sharing the identical module objects (rather than re-importing under a second
name) is what keeps ``except evalguard.EvalGuardError`` catching errors raised
through the ``evalguardai`` import path, and vice-versa.

Why ``evalguardai`` (not a bare ``evalguard`` dist): the bare ``evalguard``
name on PyPI is owned by an unaffiliated third party, so shipping the real SDK
as ``evalguardai`` keeps install-name == import-name and avoids any confusion
with that package.
"""

import sys as _sys
from importlib import import_module as _import_module

# Load the real package (and its eagerly-imported submodules: client,
# guardrails, tracing, types, pydantic_integration), then bind `evalguardai`
# and each `evalguard.<sub>` to the SAME module object under the `evalguardai`
# namespace. `list(...)` snapshots sys.modules so we can mutate it while
# iterating. The `from evalguardai import X` that triggered this import reads X
# off the aliased top-level module registered below.
_import_module("evalguard")
for _name in list(_sys.modules):
    if _name == "evalguard" or _name.startswith("evalguard."):
        _sys.modules["evalguardai" + _name[len("evalguard"):]] = _sys.modules[_name]
