# SPDX-FileCopyrightText: Copyright (c) 2025 kvpress-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the vllm-ascend latent circular-import bug.

vllm-ascend v0.23.0 has an order-sensitive module cycle::

    vllm_ascend/ops/__init__.py:21   import vllm_ascend.ops.fused_moe.fused_moe
    fused_moe.py:41                  from ...experts_selector import select_experts
    experts_selector.py:25           from vllm_ascend.device.device_op import DeviceOperator
    device_op.py:32                  from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import ...

The natural order (vllm CLI startup) completes the cycle.  Importing it from
the wrong entry (``device_op`` first — exactly what this package's pre-import
of ``attention_v1`` used to do) fails mid-cycle; the swallowed failure leaves
partial modules in ``sys.modules`` and the next import chain dies with:

    ImportError: cannot import name 'select_experts' from partially
    initialized module 'vllm_ascend.ops.fused_moe.experts_selector'

This test replays the cycle with a faithful stub tree in a subprocess
(import-order bugs are pure Python semantics and need no NPU / vllm), and
verifies that the engine's defuse entry (``import
vllm_ascend.ops.fused_moe.fused_moe`` first) makes every later order safe.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap

import pytest

# ensure the REPO source (not a possibly-stale installed copy) is inspected
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STUBS = {
    "vllm_ascend/__init__.py": "",
    "vllm_ascend/device/__init__.py": "",
    "vllm_ascend/device/device_op.py": textwrap.dedent(
        """\
        from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel

        class DeviceOperator:
            pass
        """
    ),
    "vllm_ascend/ops/__init__.py": textwrap.dedent(
        """\
        import vllm_ascend.ops.fused_moe.fused_moe  # noqa
        import vllm_ascend.ops.layernorm  # noqa
        import vllm_ascend.ops.register_custom_ops  # noqa
        """
    ),
    "vllm_ascend/ops/layernorm.py": "",
    "vllm_ascend/ops/register_custom_ops.py": "",
    "vllm_ascend/ops/fused_moe/__init__.py": "",
    "vllm_ascend/ops/fused_moe/fused_moe.py": textwrap.dedent(
        """\
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute

        def fused_experts():
            return "ok"
        """
    ),
    "vllm_ascend/ops/fused_moe/experts_selector.py": textwrap.dedent(
        """\
        from vllm_ascend.device.device_op import DeviceOperator


        def select_experts():
            return "ok"


        def zero_experts_compute():
            return "ok"
        """
    ),
    "vllm_ascend/ops/triton/__init__.py": "",
    "vllm_ascend/ops/triton/fla/__init__.py": "",
    "vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py": textwrap.dedent(
        """\
        def chunk_scaled_dot_kkt_fwd_kernel():
            return "kernel"
        """
    ),
}

# The exact error observed on the user's machine (vllm-ascend v0.23.0).
UPSTREAM_ERROR = (
    "cannot import name 'select_experts' from partially initialized module "
    "'vllm_ascend.ops.fused_moe.experts_selector'"
)

SCENARIO_SCRIPT = textwrap.dedent(
    """\
    import sys
    scenario = sys.argv[1]

    def cli_chain():
        # mirrors vllm CLI: w4a8_mxfp4 -> experts_selector (first ops import)
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts
        assert select_experts() == "ok"
        print("CLI_CHAIN_OK")

    if scenario == "natural":
        cli_chain()
    elif scenario == "preimport-perturbed":
        # mirrors kvpress-ascend's .pth pre-import (attention_v1 -> device_op)
        try:
            from vllm_ascend.device.device_op import DeviceOperator
        except ImportError as exc:
            print("PREIMPORT_FAILED:", exc)
        cli_chain()
    elif scenario == "defused":
        # the engine's fix: import the cycle via its canonical safe entry first
        import vllm_ascend.ops.fused_moe.fused_moe
        print("DEFUSE_OK")
        from vllm_ascend.device.device_op import DeviceOperator  # noqa: F401
        cli_chain()
    """
)


@pytest.fixture()
def stub_root(tmp_path):
    for rel, content in STUBS.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def _run(stub_root, scenario: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(stub_root)
    proc = subprocess.run(
        [sys.executable, "-c", SCENARIO_SCRIPT, scenario],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(stub_root),
        timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_natural_cli_order_completes(stub_root):
    code, out, err = _run(stub_root, "natural")
    assert code == 0, f"natural order must work; stderr={err}"
    assert "CLI_CHAIN_OK" in out


def test_preimport_perturbation_reproduces_upstream_error(stub_root):
    """The exact crash from the user's machine, reproduced offline."""
    code, out, err = _run(stub_root, "preimport-perturbed")
    assert code != 0, "perturbed order must crash like the upstream bug"
    assert "PREIMPORT_FAILED" in out
    assert UPSTREAM_ERROR in err, f"expected upstream error, got: {err}"


def test_defuse_makes_every_later_order_safe(stub_root):
    code, out, err = _run(stub_root, "defused")
    assert code == 0, f"defused order must work; stderr={err}"
    assert "DEFUSE_OK" in out and "CLI_CHAIN_OK" in out


def test_engine_defuse_entry_matches_safe_order():
    """The engine's defuse must import the same canonical safe entry.

    Reads the repo source by path: the .pth auto-import may have already
    loaded an installed (possibly stale) copy into sys.modules, so
    ``engine.__file__`` is not trustworthy here.
    """
    from pathlib import Path

    repo_engine = Path(__file__).resolve().parent.parent / "kvpress_ascend" / "engine.py"
    src = repo_engine.read_text(encoding="utf-8")
    assert re.search(r"import vllm_ascend\.ops\.fused_moe\.fused_moe", src), (
        "engine defuse must import vllm_ascend.ops.fused_moe.fused_moe first"
    )
    assert "_defuse_vllm_ascend_imports" in src
