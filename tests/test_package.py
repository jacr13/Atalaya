import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _python(code):
    return subprocess.run(
        [sys.executable, "-c", code], env={"PYTHONPATH": str(REPO)}, capture_output=True, text=True
    )


def test_import_as_atl_exposes_the_api():
    import atalaya as atl
    from atalaya.timer import Timer
    from atalaya.writer import Writer

    assert atl.Writer is Writer and atl.Timer is Timer
    for name in ("read_scalars", "read_info", "read_config", "read_timings", "terminal", "format_time"):
        assert callable(getattr(atl, name)) or name == "terminal"
    assert set(atl.__all__) <= set(dir(atl)) | {"__version__"}
    assert isinstance(atl.__version__, str)
    with pytest.raises(AttributeError):
        atl.does_not_exist


def test_from_import_still_works():
    from atalaya import Timer, Writer  # noqa: F401


def test_import_is_lazy():
    proc = _python(
        "import sys, atalaya as atl\n"
        "assert 'tensorboardX' not in sys.modules, 'import atalaya loaded tensorboardX'\n"
        "atl.Timer\n"
        "assert 'tensorboardX' not in sys.modules, 'atl.Timer loaded tensorboardX'\n"
        "atl.Writer\n"
        "assert 'tensorboardX' in sys.modules\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_no_export_shadows_a_submodule():
    import pkgutil

    import atalaya

    submodules = {m.name for m in pkgutil.iter_modules(atalaya.__path__)}
    for name, (module, attr) in atalaya._EXPORTS.items():
        if name in submodules:
            assert attr is None and module == name, name


def test_terminal_works_however_it_is_imported(capsys):
    import atalaya as atl
    import atalaya.terminal  # would replace an instance export with the module
    from atalaya.terminal import terminal

    atl.terminal.print_ok("via module")
    terminal.print_ok("via instance")
    out = capsys.readouterr().out
    assert "via module" in out and "via instance" in out
