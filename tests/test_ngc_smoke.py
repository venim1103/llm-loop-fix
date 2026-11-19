from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("installed", "imported", "expected_pass"),
    [
        ("0.20.1+7124b12a.nv26.5.50625635", "0.20.1+7124b12a.dev", True),
        ("0.20.1+7124b12a.nv26.5.50625635", "0.20.1+deadbeef.dev", False),
        ("0.20.1", "0.20.1", False),
    ],
)
def test_ngc_smoke_bundled_versions(monkeypatch, capsys, installed, imported, expected_pass):
    script = Path(__file__).resolve().parents[1] / ".devcontainer/gpu-ngc/smoke.py"
    main = runpy.run_path(str(script))["main"]
    gpu_calls = []
    native_calls = []

    def tensor(values, *, device):
        gpu_calls.append(device)
        return SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: sum(values)))

    def import_native(name):
        native_calls.append(name)

    monkeypatch.setitem(main.__globals__, "sys", SimpleNamespace(version_info=(3, 12)))
    monkeypatch.setitem(main.__globals__, "version", lambda _: installed)
    monkeypatch.setitem(main.__globals__, "import_module", import_native)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        version=SimpleNamespace(cuda="13.2"), tensor=tensor, __version__="fixture",
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "test GPU"),
    ))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        __version__=imported, SamplingParams=SimpleNamespace,
    ))
    monkeypatch.setitem(sys.modules, "vllm.platforms", SimpleNamespace(
        current_platform=SimpleNamespace(is_cuda=lambda: True),
    ))

    if expected_pass:
        main()
        assert native_calls == ["vllm._C"]
        assert gpu_calls == ["cuda"]
        assert capsys.readouterr().out.startswith("NGC model-free smoke passed:")
    else:
        with pytest.raises(SystemExit, match="Expected bundled vLLM"):
            main()
        assert not native_calls and not gpu_calls
