"""
provenance.py -- run provenance + timing capture for producers that call it.

This helper records the available GPU, driver, CUDA, library versions, git state,
environment, RNG configuration, command line, per-phase wall-clock, and peak GPU
memory. It does not by itself make a run reproducible, and not every historical or
analysis script calls it; release checkers separately require the concrete input,
source and artifact hashes needed by each paper-facing result.

Usage:
    import provenance as prov
    P = prov.collect(config={...})          # snapshot at start
    t = prov.Timer()
    with t.phase("adam"): ...               # times a block
    with t.phase("lm"):   ...
    P["timings"] = t.dump()
    P["peak_gpu_mem_mb"] = prov.peak_gpu_mem_mb()
    json.dump({..., "provenance": P}, f)
"""
import os, sys, json, time, platform, socket, subprocess, getpass, hashlib
from datetime import datetime, timezone
from contextlib import contextmanager


def _run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() or None
    except Exception:
        return None


def _pkg_versions():
    out = {}
    for mod in ("jax", "jaxlib", "numpy", "scipy", "optax", "flax",
                "numpyro", "blackjax", "matplotlib"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    return out


def _jax_devices():
    d = {}
    try:
        import jax
        devs = jax.devices()
        d["backend"] = devs[0].platform
        d["device_kind"] = getattr(devs[0], "device_kind", str(devs[0]))
        d["n_devices"] = len(devs)
        try:
            d["x64"] = bool(jax.config.read("jax_enable_x64"))
        except Exception:
            d["x64"] = None
    except Exception as e:
        d["error"] = str(e)
    return d


def _gpu():
    g = {}
    q = _run(["nvidia-smi",
              "--query-gpu=name,compute_cap,memory.total,memory.used,driver_version",
              "--format=csv,noheader"])
    g["nvidia_smi_query"] = q
    g["nvidia_smi_full"] = _run(["nvidia-smi"])
    # CUDA toolkit / runtime
    g["nvcc"] = _run(["nvcc", "--version"])
    return g


def _git():
    here = os.path.dirname(os.path.abspath(__file__))
    def gat(*a):
        try:
            result = subprocess.run(["git", *a], cwd=here, capture_output=True,
                                    text=True, timeout=10)
            if result.returncode:
                return None, (result.stderr.strip() or f"git exited {result.returncode}")
            return result.stdout.strip(), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    commit, error = gat("rev-parse", "HEAD")
    if commit is None:
        return {"available": False, "commit": None, "short": None, "branch": None,
                "dirty": None, "error": error}
    short, short_error = gat("rev-parse", "--short", "HEAD")
    branch, branch_error = gat("rev-parse", "--abbrev-ref", "HEAD")
    status, status_error = gat("status", "--porcelain")
    errors = [x for x in (short_error, branch_error, status_error) if x]
    return {"available": not errors, "commit": commit, "short": short, "branch": branch,
            "dirty": None if status_error else bool(status),
            "error": "; ".join(errors) if errors else None}


def _pip_freeze():
    try:
        from importlib import metadata
        return sorted("%s==%s" % (d.metadata["Name"], d.version) for d in metadata.distributions())
    except Exception:
        return None


def collect(config=None):
    """Full provenance snapshot. `config` = the run's hyperparameters dict."""
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "user": (getpass.getuser() if hasattr(getpass, "getuser") else None),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": _pkg_versions(),
        "jax_runtime": _jax_devices(),
        "gpu": _gpu(),
        "git": _git(),
        "env": {k: os.environ.get(k) for k in
                ("RAW_METRIC", "WHITEN_TRAIN", "ALPHA_MULT",
                 "XLA_FLAGS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                 "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_PLATFORMS", "OMP_NUM_THREADS",
                 "TF_CUDNN_DETERMINISTIC", "TF_DETERMINISTIC_OPS", "DETERMINISM")},
        "config": config or {},
    }
    fr = _pip_freeze()
    out["pip_freeze"] = fr
    out["pip_freeze_sha1"] = (hashlib.sha1("\n".join(fr).encode()).hexdigest() if fr else None)
    return out


def peak_gpu_mem_mb():
    """Best-effort peak GPU memory in MB (jax live buffers, else nvidia-smi)."""
    try:
        import jax
        dev = jax.devices()[0]
        st = dev.memory_stats() if hasattr(dev, "memory_stats") else None
        if st and "peak_bytes_in_use" in st:
            return round(st["peak_bytes_in_use"] / 1e6, 1)
    except Exception:
        pass
    q = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return float(q.splitlines()[0])
    except Exception:
        return None


class Timer:
    """Phase timer. `with t.phase('name'): ...` records wall-clock seconds."""
    def __init__(self):
        self.t0 = time.time()
        self.times = {}

    @contextmanager
    def phase(self, name):
        s = time.time()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + (time.time() - s)

    def mark(self, name, seconds):
        self.times[name] = float(seconds)

    def dump(self):
        out = dict(self.times)
        out["total_seconds"] = time.time() - self.t0
        return {k: round(v, 3) for k, v in out.items()}
