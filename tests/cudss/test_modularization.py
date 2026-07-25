"""Architecture guards for the modular cuDSS Python implementation."""

import ast
import sys
from pathlib import Path

import spineax.cudss as cudss
import spineax.cudss.solver as solver

_api = sys.modules["spineax.cudss._api"]
_core = sys.modules["spineax.cudss._core"]
_ffi = sys.modules["spineax.cudss._ffi"]
_lineax = sys.modules["spineax.cudss._lineax"]
_transpose = sys.modules["spineax.cudss._transpose"]


def test_public_surface_and_canonical_identities():
    public = {
        "CSROperator",
        "CuDSS",
        "FactorToken",
        "analyze",
        "cache_capacity",
        "factorize",
        "inertia",
        "query",
        "rebuild_count",
        "refactorize",
        "registry_size",
        "release",
        "solve",
    }
    assert set(solver.__all__) == public
    assert set(cudss.__all__) == public
    for name in public:
        assert getattr(cudss, name) is getattr(solver, name)
    assert solver.FactorToken is _core.FactorToken
    assert solver.CSROperator is _lineax.CSROperator
    assert solver.CuDSS is _lineax.CuDSS
    for name in public - {"CSROperator", "CuDSS", "FactorToken"}:
        assert getattr(solver, name) is getattr(_api, name)
    assert solver.__dict__["_transpose_csr"] is _transpose._transpose_csr
    assert _ffi._ffi_p.name == "spineax_ffi"


def test_public_classes_keep_compatibility_paths():
    assert solver.FactorToken.__module__ == "spineax.cudss.solver"
    assert solver.CSROperator.__module__ == "spineax.cudss.solver"
    assert solver.CuDSS.__module__ == "spineax.cudss.solver"


def test_private_module_dependency_dag_and_single_registration_owner():
    package = Path(solver.__file__).parent
    modules = ("_core", "_ffi", "_batching", "_transpose", "_api", "_lineax")
    allowed = {
        "_core": set(),
        "_ffi": {"_core"},
        "_batching": {"_core", "_ffi"},
        "_transpose": {"_core", "_ffi"},
        "_api": {"_batching", "_core", "_ffi", "_transpose"},
        "_lineax": {"_api", "_transpose"},
    }

    def local_dependencies(tree):
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("spineax.cudss."):
                        dependencies.add(alias.name.rsplit(".", 1)[-1])
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0 and node.module is None:
                    dependencies.update(alias.name for alias in node.names)
                elif node.module is not None and (
                    node.level > 0 or node.module.startswith("spineax.cudss.")
                ):
                    dependencies.add(node.module.rsplit(".", 1)[-1])
        return dependencies & set(modules)

    registration_owners = {
        "native_import": set(),
        "primitive": set(),
        "ffi_target": set(),
    }
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        module = path.stem
        if module in allowed:
            assert local_dependencies(tree) <= allowed[module]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "spineax":
                if any(alias.name == "pbatch_solve" for alias in node.names):
                    registration_owners["native_import"].add(path.name)
            elif isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None)
                if name == "register_ffi_target":
                    registration_owners["ffi_target"].add(path.name)
                elif (
                    name == "Primitive"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "spineax_ffi"
                ):
                    registration_owners["primitive"].add(path.name)

    assert registration_owners == {
        "native_import": {"_ffi.py"},
        "primitive": {"_ffi.py"},
        "ffi_target": {"_ffi.py"},
    }
