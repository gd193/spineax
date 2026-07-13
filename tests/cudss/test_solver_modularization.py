import ast
import importlib
from pathlib import Path


solver = importlib.import_module("spineax.cudss.solver")
_primitives = importlib.import_module("spineax.cudss._primitives")
_batching = importlib.import_module("spineax.cudss._batching")
jax_batching = importlib.import_module("jax._src.interpreters.batching")
primitive_batchers = getattr(
    jax_batching, "fancy_primitive_batchers", jax_batching.primitive_batchers
)


def test_public_surface_and_primitive_identity():
    for name in (
        "CuDSSSolver",
        "ConstantCSRCuDSSSolver",
        "solve",
        "batch_solve",
        "pbatch_solve",
        "PBATCH_AVAILABLE",
        "vmap_using_pseudo_batch",
        "compute_inertia_from_diag_perm",
        "register_ffi",
    ):
        assert hasattr(solver, name)

    assert len(_primitives._PRIMITIVES) == 32
    assert len({id(primitive) for primitive in _primitives._PRIMITIVES.values()}) == 32
    for name, primitive in _primitives._PRIMITIVES.items():
        assert getattr(solver, f"{name}_p") is primitive
        assert primitive in primitive_batchers


def test_public_classes_remain_in_facade():
    assert solver.CuDSSSolver.__module__ == "spineax.cudss.solver"
    assert solver.ConstantCSRCuDSSSolver.__module__ == "spineax.cudss.solver"


def test_private_module_dependency_boundaries():
    solver_file = solver.__file__
    assert solver_file is not None
    package = Path(solver_file).parent
    imports = {}
    for module in ("_validation", "_primitives", "_dispatch", "_batching"):
        tree = ast.parse((package / f"{module}.py").read_text())
        imports[module] = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }

    assert not any(
        name.endswith("solver") for names in imports.values() for name in names
    )
    reverse_primitive_import = any(
        name.endswith(("_dispatch", "_batching")) for name in imports["_primitives"]
    )
    assert not reverse_primitive_import
    assert not any(name.endswith("_batching") for name in imports["_dispatch"])

    native_importers = []
    for path in package.glob("*.py"):
        text = path.read_text()
        if any(
            name in text
            for name in ("single_solve as", "batch_solve as", "pbatch_solve as")
        ):
            native_importers.append(path.name)
    assert native_importers == ["_primitives.py"]
