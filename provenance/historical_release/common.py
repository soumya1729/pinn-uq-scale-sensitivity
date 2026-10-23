"""
common.py -- single source of truth for the unified suite.

Architecture, PDE definitions (Burgers + Allen-Cahn), exact reference solutions,
deterministic data sampling, and MAP I/O. Every suite script calls
`build_context(get_config())` so the network, collocation points, and reference
are byte-identical across train_map.py and every method/diagnostic. This is what
guarantees the shared-MAP consistency.

Faithfully factored from the validated phase2/train_pinn.py (which reproduced the
proven baseline: Burgers BLL last-layer sigma-ratio 1.50, bulk rel-L2 0.10%).
"""
import os, math
if os.environ.get("DETERMINISM", "0") == "1":   # opt-in; flags recorded in provenance either way
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap
from jax.flatten_util import ravel_pytree
from numpy.polynomial.hermite import hermgauss

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------
def get_config(overrides=None):
    e = os.environ
    ff = lambda k, d: float(e.get(k, d))
    ii = lambda k, d: int(e.get(k, d))
    pde = e.get("PDE", "burgers").lower()
    c = SimpleNamespace(
        pde=pde, seed=ii("SEED", 0), arch=e.get("ARCH", "mlp").lower(),
        nu=ff("NU", 0.01 / math.pi), eps2=ff("EPS2", 1e-4), reac=ff("REAC", 5.0),
        kpp_D=ff("KPP_D", 2.5e-4), kpp_r=ff("KPP_R", 25.0), kpp_w0=ff("KPP_W0", 0.05),
        nag_D=ff("NAG_D", 2.5e-3), nag_r=ff("NAG_R", 25.0), nag_a=ff("NAG_A", 0.25),
        nag_w0=ff("NAG_W0", 0.3), nag_x0=ff("NAG_X0", 0.35), nag_w=ff("NAG_W", 0.05),
        nag_ic=e.get("NAG_IC", "plateau"),   # plateau (round 2) | gauss (round 1, collapsed)
        fourier_scale=ff("FOURIER_SCALE", 2.0), fourier_dim=ii("FOURIER_DIM", 16),
        Mx=20, Mt=20, hidden=64, fusion=128,
        N_f=5000, N_u=1000, N_b=200,
        adam_lr=1e-3,
        adam_iters=ii("ADAM_ITERS", 15000 if pde in ("allen_cahn","kuramoto_sivashinsky","ks")
                      else (20000 if pde == "nagumo"
                            else (10000 if pde == "fisher_kpp" else 5000))),
        lm_iters=ii("LM_ITERS", 200 if pde in ("allen_cahn","kuramoto_sivashinsky","ks")
                    else (100 if pde in ("fisher_kpp", "nagumo") else 50)),
        ic_weight=10.0, bc_weight=10.0, bcu_weight=10.0, bcx_weight=1.0,
        # KS domain/horizon are env-tunable (added 2026-07-27 for the horizon-vs-operator
        # discriminating study, PROJECT_LOG sec 3f): the residual carries Tp and Lx as
        # explicit scale factors, so shortening Tp is a clean, single-knob change.
        ks_L=ff("KS_L", 32.0*math.pi), ks_T=ff("KS_T", 100.0),
        ks_t0=ff("KS_T0", 0.0),   # ARM C: spin-up time before the window starts
        cd_a=ff("CD_A", 1.0), cd_delta=ff("CD_DELTA", 0.13),   # convection-diffusion (NEGATIVE CONTROL): solvable pulse width
        cd_nu=ff("CD_NU", 0.01/math.pi),                       # CD diffusion: solvable regime => target~1, full-network sigma-ratio ~1.1-1.4 (no meaningful contrast; corrected 2026-07-17, audit sec 8). delta<0.07 is unsolvable for both MLP & Fourier.
        gh_n=80, lm_cg=100, lm_mu_init=1e-4, lm_mu_min=1e-12,
        lm_rho_accept=0.25, lm_rho_good=0.75, lm_mu_dec=1.0/3.0,
        lm_mu_inc=4.0, lm_max_rej=5,
        eval_nx=400, eval_nt=200, smoke=ii("SMOKE", 0),
    )
    c.d = c.fusion; c.d_aug = c.fusion + 1
    if c.smoke:
        c.N_f, c.N_u, c.N_b = 400, 200, 40
        c.adam_iters, c.lm_iters = 60, 3
        c.eval_nx, c.eval_nt = 80, 40
        c.Mx, c.Mt, c.hidden, c.fusion = 3, 3, 8, 16   # tiny arch -> fast compile, small matrices
        c.d, c.d_aug = c.fusion, c.fusion + 1
    if overrides:
        for k, v in overrides.items():
            setattr(c, k, v)
    return c


def config_dict(c):
    return {k: getattr(c, k) for k in vars(c)}


# ----------------------------------------------------------------------
def build_context(c):
    """Reconstruct the EXACT model + data + reference for config c.
    Deterministic in c.seed, so calling this in train_map and in any method
    yields identical collocation points and architecture."""
    key = jax.random.PRNGKey(c.seed)

    # --- Fourier features (only consume RNG in fourier mode) ---
    if c.arch == "fourier":
        # The key split must happen UNCONDITIONALLY (downstream data sampling depends on
        # the post-split key), but B itself is loaded from a sidecar when one exists:
        # JAX's threefry stream changed across versions (jax_threefry_partitionable), so
        # re-deriving B from the seed on a different JAX silently yields a DIFFERENT
        # network for the same params.pkl. Sidecar search order: env FOURIER_B_NPZ, then
        # <OUTDIR>/<tag>_fourier_B.npz, then results/<tag>/<tag>_fourier_B.npz.
        # train_map writes the sidecar at training time; dump_fourier_B.py backfills it
        # for existing MAPs (run on the SAME JAX that trained them).
        key, kx, kt = jax.random.split(key, 3)
        _tag = f"{c.pde}_{c.arch}_seed{c.seed}"
        _benv = os.environ.get("FOURIER_B_NPZ", "")
        if _benv == "derive":                      # explicit escape hatch: force PRNG derivation
            _bcands = []
        else:
            _bcands = [_benv,
                       os.path.join(os.environ.get("OUTDIR", ""), f"{_tag}_fourier_B.npz"),
                       os.path.join("results", _tag, f"{_tag}_fourier_B.npz")]
        _bfile = next((p for p in _bcands if p and os.path.exists(p)), None)
        if _bfile is not None:
            _bz = np.load(_bfile)
            B_X = jnp.asarray(_bz["B_X"]); B_T = jnp.asarray(_bz["B_T"])
            print(f"[common] fourier B loaded from sidecar {_bfile}")
        else:
            B_X = jax.random.normal(kx, (1, c.fourier_dim)) * c.fourier_scale
            B_T = jax.random.normal(kt, (1, c.fourier_dim)) * c.fourier_scale
            print("[common] fourier B derived from PRNG seed (no sidecar found) -- "
                  "version-sensitive: pin the training JAX or provide FOURIER_B_NPZ")
    else:
        B_X = B_T = None
    sub_in = 2 * c.fourier_dim if c.arch == "fourier" else 1

    def rff(v, B):
        proj = 2.0 * jnp.pi * (v @ B)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)
    embed_x = (lambda x: rff(x, B_X)) if c.arch == "fourier" else (lambda x: x)
    embed_t = (lambda t: rff(t, B_T)) if c.arch == "fourier" else (lambda t: t)

    # --- architecture ---
    def init_mlp(k, layers):
        ps, ks = [], jax.random.split(k, len(layers))
        for kk, (di, do) in zip(ks, zip(layers[:-1], layers[1:])):
            ps.append({"w": jax.random.normal(kk, (di, do)) * jnp.sqrt(2.0/di),
                       "b": jnp.zeros((do,))})
        return ps

    def mlp_forward(p, x):
        for layer in p[:-1]:
            x = jnp.tanh(x @ layer["w"] + layer["b"])
        return x @ p[-1]["w"] + p[-1]["b"]

    def init_pinn(k):
        ks = jax.random.split(k, c.Mx + c.Mt + 1)
        sp = [init_mlp(ks[i], [sub_in, c.hidden, c.hidden, 1]) for i in range(c.Mx)]
        tp = [init_mlp(ks[c.Mx + i], [sub_in, c.hidden, c.hidden, 1]) for i in range(c.Mt)]
        fu = init_mlp(ks[-1], [c.Mx + c.Mt, c.fusion, c.fusion, 1])
        return {"spatial": sp, "temporal": tp, "fusion": fu}

    def forward(p, xt):
        x, t = xt[:, :1], xt[:, 1:]
        feats = [mlp_forward(n, embed_x(x)) for n in p["spatial"]]
        feats += [mlp_forward(n, embed_t(t)) for n in p["temporal"]]
        return mlp_forward(p["fusion"], jnp.concatenate(feats, axis=1))

    def extract_features(p, xt):
        x, t = xt[:, :1], xt[:, 1:]
        feats = [mlp_forward(n, embed_x(x)) for n in p["spatial"]]
        feats += [mlp_forward(n, embed_t(t)) for n in p["temporal"]]
        h = jnp.concatenate(feats, axis=1)
        for layer in p["fusion"][:-1]:
            h = jnp.tanh(h @ layer["w"] + layer["b"])
        return h

    # --- PDE-specific: IC, reference, residual ---
    if c.pde == "burgers":
        def u_ic(x): return -jnp.sin(jnp.pi * x)
        GHn, GHw = hermgauss(c.gh_n)
        GHn, GHw = np.asarray(GHn), np.asarray(GHw)
        def reference(xq, tq):
            x = np.asarray(xq, np.float64); t = np.asarray(tq, np.float64)
            out = np.zeros_like(x); z = t < 1e-14; out[z] = -np.sin(np.pi * x[z])
            m = ~z
            if not np.any(m): return out
            xm, tm = x[m], t[m]; s = 2.0 * np.sqrt(c.nu * tm)
            eta = xm[:, None] + s[:, None] * GHn[None, :]
            lp = -(1.0 + np.cos(np.pi * eta)) / (2.0 * np.pi * c.nu)
            lpm = lp.max(1, keepdims=True); w = GHw[None, :] * np.exp(lp - lpm)
            out[m] = np.sum(w * (-np.sin(np.pi * eta)), 1) / np.sum(w, 1)
            return out
        def grid_reference(xs, ts):
            XX, TT = np.meshgrid(xs, ts)
            return reference(XX.ravel(), TT.ravel()).reshape(len(ts), len(xs))
        def residual_point(p, x, t):
            def u(xx, tt): return forward(p, jnp.array([[xx, tt]]))[0, 0]
            return grad(u, 1)(x, t) + u(x, t)*grad(u, 0)(x, t) - c.nu*grad(grad(u, 0), 0)(x, t)
    elif c.pde == "allen_cahn":
        def u_ic(x): return x**2 * jnp.cos(jnp.pi * x)
        def u_ic_np(x): return x**2 * np.cos(np.pi * x)
        def etdrk4(eps2, reac, Nx, T, Nt):
            L = 2.0; x = np.linspace(-L/2, L/2, Nx, endpoint=False); dt = T/(Nt-1)
            k = 2.0*np.pi*np.fft.fftfreq(Nx, d=L/Nx); Lop = -eps2*k**2 + reac
            E = np.exp(dt*Lop); E2 = np.exp(dt*Lop/2.0); Mc = 32
            r = np.exp(1j*np.pi*(np.arange(1, Mc+1)-0.5)/Mc); LR = dt*Lop[:, None]+r[None, :]
            Q = dt*np.real(np.mean((np.exp(LR/2)-1.0)/LR, 1))
            f1 = dt*np.real(np.mean((-4.0-LR+np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3, 1))
            f2 = dt*np.real(np.mean((2.0+LR+np.exp(LR)*(-2.0+LR))/LR**3, 1))
            f3 = dt*np.real(np.mean((-4.0-3.0*LR-LR**2+np.exp(LR)*(4.0-LR))/LR**3, 1))
            Nh = lambda uh: np.fft.fft(-reac*np.real(np.fft.ifft(uh))**3)
            u = u_ic_np(x); uh = np.fft.fft(u); snaps = [u.copy()]
            for _ in range(Nt-1):
                Nu = Nh(uh); a = E2*uh+Q*Nu; Na = Nh(a); b = E2*uh+Q*Na; Nb = Nh(b)
                cc = E2*a+Q*(2.0*Nb-Nu); Nc = Nh(cc)
                uh = E*uh+Nu*f1+2.0*(Na+Nb)*f2+Nc*f3; snaps.append(np.real(np.fft.ifft(uh)))
            return np.array(snaps), x, np.linspace(0.0, T, Nt)
        # Reference resolution + interpolation are configurable for a validity check; DEFAULTS
        # (512 modes, bilinear) reproduce the original behaviour exactly when the env is unset.
        _acnx = int(os.environ.get("AC_REF_MODES", 512)); _acnt = int(os.environ.get("AC_REF_STEPS", 1001))
        UREF, XREF, TREF = etdrk4(c.eps2, c.reac, _acnx, 1.0, _acnt)
        if os.environ.get("AC_REF_SPECTRAL", "0") == "1":
            # Trigonometric (spectral) interpolation in x (the periodic dimension), linear in t.
            # Resolves the sharp interface far better than bilinear -- pins down whether the 512/
            # bilinear reference under-resolves the AC shock metrics.
            _kk = 2.0*np.pi*np.fft.fftfreq(len(XREF), d=2.0/len(XREF)); _UH = np.fft.fft(UREF, axis=1)
            def _interp(xq, tq):
                xq = np.asarray(xq); tq = np.asarray(tq)
                it = np.clip(np.searchsorted(TREF, tq), 1, len(TREF)-1)
                t0, t1 = TREF[it-1], TREF[it]; wt = (tq-t0)/(t1-t0+1e-30)
                uh = (1-wt)[:, None]*_UH[it-1] + wt[:, None]*_UH[it]        # linear-in-t spectrum
                return np.real(np.sum(uh*np.exp(1j*np.outer(xq-XREF[0], _kk)), axis=1))/len(XREF)
        else:
            def _interp(xq, tq):
                xq = np.asarray(xq); tq = np.asarray(tq)
                ix = np.clip(np.searchsorted(XREF, xq), 1, len(XREF)-1)
                it = np.clip(np.searchsorted(TREF, tq), 1, len(TREF)-1)
                x0, x1 = XREF[ix-1], XREF[ix]; t0, t1 = TREF[it-1], TREF[it]
                wx = (xq-x0)/(x1-x0+1e-30); wt = (tq-t0)/(t1-t0+1e-30)
                return ((1-wt)*(1-wx)*UREF[it-1, ix-1]+(1-wt)*wx*UREF[it-1, ix]
                        + wt*(1-wx)*UREF[it, ix-1]+wt*wx*UREF[it, ix])
        def grid_reference(xs, ts):
            XX, TT = np.meshgrid(xs, ts)
            return _interp(XX.ravel(), TT.ravel()).reshape(len(ts), len(xs))
        def residual_point(p, x, t):
            def u(xx, tt): return forward(p, jnp.array([[xx, tt]]))[0, 0]
            uv = u(x, t)
            return grad(u, 1)(x, t) - c.eps2*grad(grad(u, 0), 0)(x, t) - c.reac*uv + c.reac*uv**3
    elif c.pde == "fisher_kpp":
        # Fisher-KPP: u_t = D u_xx + r u (1 - u), periodic [-1,1] (THIRD positive PDE).
        # A Gaussian seed (width kpp_w0) saturates to u=1 and spreads as two pulled
        # fronts whose profile and speed are set by the reaction-diffusion balance,
        # not the IC (speed -> 2*sqrt(D r), width ~ sqrt(D/r)): the front SHARPENS
        # itself from w0=0.05 to ~0.03 in-window -- nonlinear self-organisation, the
        # property the convection-diffusion control lacks. Baseline D=2.5e-4, r=25:
        # speed ~0.20, fronts reach +/-0.29 by t=1, front ~17 reference cells.
        # Predictions pre-registered in KPP_PREREG.md BEFORE any training run.
        # NOTE: raising r much beyond ~25 destabilises the un-dealiased ETDRK4 at the
        # default step count (band of linearly growing modes up to k=sqrt(r/D));
        # escalate stiffness only with KPP_REF_STEPS >= 4x and re-check convergence.
        def u_ic(x): return jnp.exp(-(x / c.kpp_w0) ** 2)
        def u_ic_np(x): return np.exp(-(x / c.kpp_w0) ** 2)
        def etdrk4_kpp(D, reac, Nx, T, Nt):
            # same Kassam--Trefethen ETDRK4 skeleton as allen_cahn; Lop = -D k^2 + r,
            # nonlinear term N(u) = -r u^2.
            L = 2.0; x = np.linspace(-L/2, L/2, Nx, endpoint=False); dt = T/(Nt-1)
            k = 2.0*np.fft.fftfreq(Nx, d=L/Nx)*np.pi; Lop = -D*k**2 + reac
            E = np.exp(dt*Lop); E2 = np.exp(dt*Lop/2.0); Mc = 32
            r_ = np.exp(1j*np.pi*(np.arange(1, Mc+1)-0.5)/Mc); LR = dt*Lop[:, None]+r_[None, :]
            Q = dt*np.real(np.mean((np.exp(LR/2)-1.0)/LR, 1))
            f1 = dt*np.real(np.mean((-4.0-LR+np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3, 1))
            f2 = dt*np.real(np.mean((2.0+LR+np.exp(LR)*(-2.0+LR))/LR**3, 1))
            f3 = dt*np.real(np.mean((-4.0-3.0*LR-LR**2+np.exp(LR)*(4.0-LR))/LR**3, 1))
            Nh = lambda uh: np.fft.fft(-reac*np.real(np.fft.ifft(uh))**2)
            u = u_ic_np(x); uh = np.fft.fft(u); snaps = [u.copy()]
            for _ in range(Nt-1):
                Nu = Nh(uh); a = E2*uh+Q*Nu; Na = Nh(a); b = E2*uh+Q*Na; Nb = Nh(b)
                cc = E2*a+Q*(2.0*Nb-Nu); Nc = Nh(cc)
                uh = E*uh+Nu*f1+2.0*(Na+Nb)*f2+Nc*f3; snaps.append(np.real(np.fft.ifft(uh)))
            return np.array(snaps), x, np.linspace(0.0, T, Nt)
        _knx = int(os.environ.get("KPP_REF_MODES", 1024)); _knt = int(os.environ.get("KPP_REF_STEPS", 1001))
        UREF, XREF, TREF = etdrk4_kpp(c.kpp_D, c.kpp_r, _knx, 1.0, _knt)
        def _interp(xq, tq):
            xq = np.asarray(xq); tq = np.asarray(tq)
            ix = np.clip(np.searchsorted(XREF, xq), 1, len(XREF)-1)
            it = np.clip(np.searchsorted(TREF, tq), 1, len(TREF)-1)
            x0, x1 = XREF[ix-1], XREF[ix]; t0, t1 = TREF[it-1], TREF[it]
            wx = (xq-x0)/(x1-x0+1e-30); wt = (tq-t0)/(t1-t0+1e-30)
            return ((1-wt)*(1-wx)*UREF[it-1, ix-1]+(1-wt)*wx*UREF[it-1, ix]
                    + wt*(1-wx)*UREF[it, ix-1]+wt*wx*UREF[it, ix])
        def grid_reference(xs, ts):
            XX, TT = np.meshgrid(xs, ts)
            return _interp(XX.ravel(), TT.ravel()).reshape(len(ts), len(xs))
        def residual_point(p, x, t):
            def u(xx, tt): return forward(p, jnp.array([[xx, tt]]))[0, 0]
            uv = u(x, t)
            return grad(u, 1)(x, t) - c.kpp_D*grad(grad(u, 0), 0)(x, t) - c.kpp_r*uv*(1.0 - uv)
    elif c.pde == "nagumo":
        # Bistable Nagumo: u_t = D u_xx + r u (u - a)(1 - u), periodic [-1,1], a in (0,1/2).
        # THE solvable reaction-diffusion front: BOTH equilibria are stable (decay rates
        # r*a and r*(1-a)), so the residual-to-error amplification that killed Fisher-KPP
        # (unstable u=0, gain e^{rT}; PROJECT_LOG 2026-07-18) becomes a CONTRACTION --
        # large r is training-friendly here. The front is nonlinear and self-organized
        # with a CLOSED-FORM traveling wave: U(xi)=1/(1+exp(xi/delta)), delta=sqrt(2D/r),
        # speed c=sqrt(2Dr)(1/2-a). Baseline D=2.5e-3, r=50, a=0.25: delta=0.01,
        # c=0.125; a wide Gaussian bump (w0=0.3) self-sharpens to the 0.043-wide wave
        # (validated standalone: speed 0.127 vs 0.125, width 0.0430 vs 0.0439, profile
        # within 1.5e-2 of the exact wave, 1024-vs-2048 spectral agreement 1e-9).
        # Predictions pre-registered in PROJECT_LOG sec 3b BEFORE any training run.
        # ROUND 2 (2026-07-18): the round-1 Gaussian IC collapsed to the stable u=0
        # attractor in training (Adam smoothing drags threshold-ambiguous bump heights
        # under a; the learned state's own dynamics then legitimately decay it -- thin
        # transient-residual cost, then residual-free). The plateau IC removes the cheap
        # exit: interior sits 0.75 above threshold, so shoulder smoothing cannot drag it
        # under a, and collapse would cost residual over a large space-time volume.
        # r 50->25 (delta 0.0141, c 0.0884) and 20k Adam for trainability. NAG_IC=gauss
        # reproduces round 1 exactly.
        if c.nag_ic == "plateau":
            def u_ic(x):
                return (1.0/(1.0+jnp.exp(-(x + c.nag_x0)/c.nag_w))
                        * 1.0/(1.0+jnp.exp(-(c.nag_x0 - x)/c.nag_w)))
            def u_ic_np(x):
                return (1.0/(1.0+np.exp(-(x + c.nag_x0)/c.nag_w))
                        * 1.0/(1.0+np.exp(-(c.nag_x0 - x)/c.nag_w)))
        else:
            def u_ic(x): return jnp.exp(-(x / c.nag_w0) ** 2)
            def u_ic_np(x): return np.exp(-(x / c.nag_w0) ** 2)
        def etdrk4_nag(D, reac, aa, Nx, T, Nt):
            # Kassam--Trefethen ETDRK4; Lop = -D k^2 (pure diffusion), reaction fully in N.
            L = 2.0; x = np.linspace(-L/2, L/2, Nx, endpoint=False); dt = T/(Nt-1)
            k = 2.0*np.fft.fftfreq(Nx, d=L/Nx)*np.pi; Lop = -D*k**2
            E = np.exp(dt*Lop); E2 = np.exp(dt*Lop/2.0); Mc = 32
            r_ = np.exp(1j*np.pi*(np.arange(1, Mc+1)-0.5)/Mc); LR = dt*Lop[:, None]+r_[None, :]
            Q = dt*np.real(np.mean((np.exp(LR/2)-1.0)/LR, 1))
            f1 = dt*np.real(np.mean((-4.0-LR+np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3, 1))
            f2 = dt*np.real(np.mean((2.0+LR+np.exp(LR)*(-2.0+LR))/LR**3, 1))
            f3 = dt*np.real(np.mean((-4.0-3.0*LR-LR**2+np.exp(LR)*(4.0-LR))/LR**3, 1))
            def Nh(uh):
                u = np.real(np.fft.ifft(uh))
                return np.fft.fft(reac*u*(u-aa)*(1.0-u))
            u = u_ic_np(x); uh = np.fft.fft(u); snaps = [u.copy()]
            for _ in range(Nt-1):
                Nu = Nh(uh); av = E2*uh+Q*Nu; Na = Nh(av); b = E2*uh+Q*Na; Nb = Nh(b)
                cc = E2*av+Q*(2.0*Nb-Nu); Nc = Nh(cc)
                uh = E*uh+Nu*f1+2.0*(Na+Nb)*f2+Nc*f3; snaps.append(np.real(np.fft.ifft(uh)))
            return np.array(snaps), x, np.linspace(0.0, T, Nt)
        _nnx = int(os.environ.get("NAG_REF_MODES", 1024)); _nnt = int(os.environ.get("NAG_REF_STEPS", 1001))
        UREF, XREF, TREF = etdrk4_nag(c.nag_D, c.nag_r, c.nag_a, _nnx, 1.0, _nnt)
        def _interp(xq, tq):
            xq = np.asarray(xq); tq = np.asarray(tq)
            ix = np.clip(np.searchsorted(XREF, xq), 1, len(XREF)-1)
            it = np.clip(np.searchsorted(TREF, tq), 1, len(TREF)-1)
            x0, x1 = XREF[ix-1], XREF[ix]; t0, t1 = TREF[it-1], TREF[it]
            wx = (xq-x0)/(x1-x0+1e-30); wt = (tq-t0)/(t1-t0+1e-30)
            return ((1-wt)*(1-wx)*UREF[it-1, ix-1]+(1-wt)*wx*UREF[it-1, ix]
                    + wt*(1-wx)*UREF[it, ix-1]+wt*wx*UREF[it, ix])
        def grid_reference(xs, ts):
            XX, TT = np.meshgrid(xs, ts)
            return _interp(XX.ravel(), TT.ravel()).reshape(len(ts), len(xs))
        def residual_point(p, x, t):
            def u(xx, tt): return forward(p, jnp.array([[xx, tt]]))[0, 0]
            uv = u(x, t)
            return (grad(u, 1)(x, t) - c.nag_D*grad(grad(u, 0), 0)(x, t)
                    - c.nag_r*uv*(uv - c.nag_a)*(1.0 - uv))
    elif c.pde in ("kuramoto_sivashinsky", "ks"):
        # KS on physical [0,Lx]x[0,Tp], NORMALISED to the suite's [-1,1]x[0,1] grid
        # (network input stays [-1,1]; the residual carries the scale factors).
        Lx, Tp, T0 = c.ks_L, c.ks_T, c.ks_t0
        # ARM C (KS_T0 > 0): start from an ALREADY-DEVELOPED state.
        #
        # Arms A (T=100) and B (T=10) differ in TWO variables at once -- horizon AND
        # how developed the solution is (the T=10 reference carries 35x less energy in
        # the [0.5,1) band, where lambda(k)=k^2-k^4 peaks). So Arm B's near-success
        # cannot be attributed to the short horizon alone. Arm C is the missing cell of
        # the 2x2: a SHORT window (T=Tp, like B) begun from a DEVELOPED state (spun up
        # for T0, like A). The spin-up uses the identical integrator, so the initial
        # condition carries only the reference's own ~1e-10 error.
        _KS_U0 = None
        def _ks_march(u0, Nx, T, Nt, keep):
            """ETDRK4 for KS. keep=False returns only the final state (spin-up)."""
            x = Lx*np.arange(Nx)/Nx; dt = T/(Nt-1)
            k = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx); Lop = k**2 - k**4
            E = np.exp(dt*Lop); E2 = np.exp(dt*Lop/2.0); Mc = 32
            r = np.exp(1j*np.pi*(np.arange(1, Mc+1)-0.5)/Mc); LR = dt*Lop[:, None]+r[None, :]
            Q = dt*np.real(np.mean((np.exp(LR/2)-1.0)/LR, 1))
            f1 = dt*np.real(np.mean((-4.0-LR+np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3, 1))
            f2 = dt*np.real(np.mean((2.0+LR+np.exp(LR)*(-2.0+LR))/LR**3, 1))
            f3 = dt*np.real(np.mean((-4.0-3.0*LR-LR**2+np.exp(LR)*(4.0-LR))/LR**3, 1))
            gg = -0.5j*k; Nh = lambda v: gg*np.fft.fft(np.real(np.fft.ifft(v))**2)
            v = np.fft.fft(u0); snaps = [np.asarray(u0).copy()]
            for _ in range(Nt-1):
                Nv = Nh(v); a = E2*v+Q*Nv; Na = Nh(a); b = E2*v+Q*Na; Nb = Nh(b)
                cc = E2*a+Q*(2.0*Nb-Nv); Nc = Nh(cc)
                v = E*v+Nv*f1+2.0*(Na+Nb)*f2+Nc*f3
                if keep:
                    snaps.append(np.real(np.fft.ifft(v)))
            return (np.array(snaps), x, np.linspace(0.0, T, Nt)) if keep \
                   else np.real(np.fft.ifft(v))

        _xgrid = Lx*np.arange(256)/256
        _u_start = np.cos(_xgrid/16.0)*(1.0+np.sin(_xgrid/16.0))   # the catalogued IC
        if T0 > 0.0:
            # spin up to a developed state; step count matched to the arm-A resolution
            _nspin = max(2, int(round(2000*T0/100.0)) + 1)
            _u_start = _ks_march(_u_start, 256, T0, _nspin, keep=False)
            print(f"[common] KS ARM C: spun up {T0:g} time units ({_nspin-1} steps); "
                  f"IC sd {_u_start.std():.4f} (vs 0.7911 for the catalogued IC)")
        _KS_U0 = _u_start
        _U0_J = jnp.asarray(_KS_U0)
        _XG_J = jnp.asarray(_xgrid)

        def u_ic(xn):
            xp = (xn + 1.0) * 0.5 * Lx
            if T0 > 0.0:
                # periodic linear interpolation of the spun-up state; the network's
                # collocation x are arbitrary, the spin-up lives on a 256-point grid
                return jnp.interp(xp, _XG_J, _U0_J, period=Lx)
            return jnp.cos(xp/16.0) * (1.0 + jnp.sin(xp/16.0))

        def _ks_etdrk4(Nx, Nt):
            u0 = _KS_U0 if Nx == 256 else np.interp(
                Lx*np.arange(Nx)/Nx, _xgrid, _KS_U0, period=Lx)
            return _ks_march(u0, Nx, Tp, Nt, keep=True)

        UREF, XREF, TREF = _ks_etdrk4(256, 2001)
        def _interp(xq, tq):
            xq = np.asarray(xq); tq = np.asarray(tq)
            ix = np.clip(np.searchsorted(XREF, xq), 1, len(XREF)-1)
            it = np.clip(np.searchsorted(TREF, tq), 1, len(TREF)-1)
            x0, x1 = XREF[ix-1], XREF[ix]; t0, t1 = TREF[it-1], TREF[it]
            wx = (xq-x0)/(x1-x0+1e-30); wt = (tq-t0)/(t1-t0+1e-30)
            return ((1-wt)*(1-wx)*UREF[it-1, ix-1]+(1-wt)*wx*UREF[it-1, ix]
                    + wt*(1-wx)*UREF[it, ix-1]+wt*wx*UREF[it, ix])
        def grid_reference(xs, ts):
            XX, TT = np.meshgrid(xs, ts)
            xp = (XX.ravel()+1.0)*0.5*Lx; tp = TT.ravel()*Tp
            return _interp(xp, tp).reshape(len(ts), len(xs))
        _a1 = Tp*(2.0/Lx); _a2 = Tp*(2.0/Lx)**2; _a4 = Tp*(2.0/Lx)**4
        def residual_point(p, x, t):
            def u(xx, tt): return forward(p, jnp.array([[xx, tt]]))[0, 0]
            uv = u(x, t); ux = grad(u, 0)(x, t); ut = grad(u, 1)(x, t)
            uxx = grad(grad(u, 0), 0)(x, t)
            uxxxx = grad(grad(grad(grad(u, 0), 0), 0), 0)(x, t)
            return ut + _a1*uv*ux + _a2*uxx + _a4*uxxxx
    elif c.pde == "convection_diffusion":
        # Linear convection-diffusion on periodic [-1,1]: u_t + a u_x = nu u_xx.
        # A sharp Gaussian pulse is advected (speed a) and slowly diffused (nu = stiffness);
        # small nu keeps a sharp transported front -- a 3rd sharp-feature type (advected layer),
        # qualitatively distinct from Burgers' self-steepening shock and AC's reaction interface.
        # Exact spectral (Fourier) reference: each mode evolves analytically, no time-stepping error.
        a_cd, nu_cd, d0 = c.cd_a, c.cd_nu, c.cd_delta
        def u_ic(x): return jnp.exp(-(x / d0) ** 2)
        _NXR = 1024
        _xref = np.linspace(-1.0, 1.0, _NXR, endpoint=False)
        _kn = 2.0 * np.pi * np.fft.fftfreq(_NXR, d=2.0 / _NXR)       # physical wavenumbers (domain length 2)
        _uh0 = np.fft.fft(np.exp(-(_xref / d0) ** 2))
        def reference(xq, tq):
            xq = np.asarray(xq, np.float64); tq = np.asarray(tq, np.float64)
            out = np.zeros_like(xq)
            for t in np.unique(tq):
                ut = np.real(np.fft.ifft(_uh0 * np.exp(-(nu_cd * _kn ** 2 + 1j * a_cd * _kn) * t)))
                m = tq == t
                out[m] = np.interp(xq[m], _xref, ut, period=2.0)
            return out
        def grid_reference(xs, ts):
            XX, TT = np.meshgrid(xs, ts)
            return reference(XX.ravel(), TT.ravel()).reshape(len(ts), len(xs))
        def residual_point(p, x, t):
            def u(xx, tt): return forward(p, jnp.array([[xx, tt]]))[0, 0]
            return grad(u, 1)(x, t) + a_cd * grad(u, 0)(x, t) - nu_cd * grad(grad(u, 0), 0)(x, t)
    else:
        raise ValueError("unknown PDE: %s" % c.pde)

    # ------------------------------------------------------------------------------
    # RES_CHUNK (added 2026-07-27, DEFAULT OFF): opt-in memory relief for KS.
    #
    # Kuramoto-Sivashinsky is the only cell whose residual needs u_xxxx -- FOUR nested
    # autodiff levels, against two everywhere else -- and the resulting loss_fn wants
    # 10.82 GiB. On a 16 GB tesla that fits during Adam but not once Levenberg-
    # Marquardt's Jacobian (6400 x 196,009 x 4 B = 5.0 GB) is also live, so every KS
    # seed died at the first LM step (job 437076, all five identical). Chunked lax.map
    # + jax.checkpoint makes the backward pass recompute each chunk's forward, so peak
    # memory is one chunk rather than all N_f points at once. Same fix as the 2-D tree
    # (STAGE 1, 2026-07-20), ported to the 1-D path.
    #
    # DEFAULT IS 0 = OFF = the original plain vmap, BIT-FOR-BIT. This matters: the 17
    # gate-passing configurations behind Table 1 were produced by that path, and
    # checkpointing is identical in exact arithmetic but reassociates floats across
    # chunk boundaries. Only slurm/22_ks_train.slurm sets RES_CHUNK; nothing else in
    # the suite changes, and no already-shipped result is affected.
    # ------------------------------------------------------------------------------
    _RCH = int(os.environ.get("RES_CHUNK", "0"))
    _rp_vm = vmap(residual_point, in_axes=(None, 0, 0))
    if _RCH > 0:
        print(f"[common] RES_CHUNK={_RCH}: residual evaluated in chunks "
              f"(memory only; values/grads unchanged)")
        _rp_ck = jax.checkpoint(lambda p, a: _rp_vm(p, a[0], a[1]))
        def _residual_chunked(p, x, t):
            n = x.shape[0]
            if n <= _RCH:                       # small ad-hoc calls: plain path
                return _rp_vm(p, x, t)
            nb = -(-n // _RCH); pad = nb*_RCH - n
            if pad:                             # pad with a repeat of the last point;
                x = jnp.concatenate([x, jnp.broadcast_to(x[-1:], (pad,))])
                t = jnp.concatenate([t, jnp.broadcast_to(t[-1:], (pad,))])
            out = jax.lax.map(lambda a: _rp_ck(p, a),
                              (x.reshape(nb, _RCH), t.reshape(nb, _RCH)))
            return out.reshape(-1)[:n]          # ... then discard the padding
        residual = jax.jit(_residual_chunked)
    else:
        residual = jax.jit(_rp_vm)

    # --- data (deterministic in seed) ---
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    x_f = jax.random.uniform(k1, (c.N_f,), minval=-1.0, maxval=1.0)
    t_f = jax.random.uniform(k2, (c.N_f,), minval=0.0, maxval=1.0)
    x_u = jax.random.uniform(k3, (c.N_u,), minval=-1.0, maxval=1.0)
    t_u = jnp.zeros_like(x_u); u_u = u_ic(x_u)
    t_b = jax.random.uniform(k4, (c.N_b,), minval=0.0, maxval=1.0)
    XT_IC = jnp.stack([x_u, t_u], 1); XT_F = jnp.stack([x_f, t_f], 1)
    data = SimpleNamespace(x_f=x_f, t_f=t_f, x_u=x_u, t_u=t_u, u_u=u_u, t_b=t_b,
                           XT_IC=XT_IC, XT_F=XT_F)

    if c.pde == "burgers":
        x_b1 = -jnp.ones_like(t_b); x_b2 = jnp.ones_like(t_b)
        XT_BC1 = jnp.stack([x_b1, t_b], 1); XT_BC2 = jnp.stack([x_b2, t_b], 1)
        data.XT_BC1, data.XT_BC2 = XT_BC1, XT_BC2

        def residual_fn(p):
            r_ic = c.ic_weight*(forward(p, XT_IC)[:, 0] - u_u)
            r_b1 = c.bc_weight*(forward(p, XT_BC1)[:, 0])
            r_b2 = c.bc_weight*(forward(p, XT_BC2)[:, 0])
            return jnp.concatenate([r_ic, r_b1, r_b2, residual(p, x_f, t_f)])
        idx = SimpleNamespace(ic=c.N_u, bc1=c.N_u+c.N_b, bc2=c.N_u+2*c.N_b,
                              f=c.N_u+2*c.N_b+c.N_f)
    else:
        def u_pt(p, x, t): return forward(p, jnp.array([[x, t]]))[0, 0]
        def ux_pt(p, x, t): return grad(u_pt, 1)(p, x, t)
        bc_u = jax.jit(vmap(lambda p, t: u_pt(p, -1.0, t) - u_pt(p, 1.0, t), in_axes=(None, 0)))
        bc_ux = jax.jit(vmap(lambda p, t: ux_pt(p, -1.0, t) - ux_pt(p, 1.0, t), in_axes=(None, 0)))
        data.bc_u, data.bc_ux = bc_u, bc_ux

        def residual_fn(p):
            r_ic = c.ic_weight*(forward(p, XT_IC)[:, 0] - u_u)
            return jnp.concatenate([r_ic, c.bcu_weight*bc_u(p, t_b),
                                    c.bcx_weight*bc_ux(p, t_b), residual(p, x_f, t_f)])
        idx = SimpleNamespace(ic=c.N_u, bcu=c.N_u+c.N_b, bcx=c.N_u+2*c.N_b,
                              f=c.N_u+2*c.N_b+c.N_f)

    N_TOTAL = int(c.N_u + 2*c.N_b + c.N_f)

    def loss_fn(p):
        r = residual_fn(p); return 0.5*jnp.mean(r**2)

    # --- per-block residual whitening (for the self-consistent "whitened" arm) ---
    # residual_fn already carries the fixed block weights (ic/bc/bcx). The whitened
    # arm additionally divides each block by its residual RMS -- exactly the weighting
    # laplace_core uses to build the whitened covariance. The point of training with
    # it (rather than bolting it on post-hoc) is that the MAP then IS a stationary
    # point of the whitened loss, so the whitened Laplace posterior is well-posed.
    _wseg = ([(0, idx.ic), (idx.ic, idx.bc1), (idx.bc1, idx.bc2), (idx.bc2, idx.f)]
             if hasattr(idx, "bc1") else
             [(0, idx.ic), (idx.ic, idx.bcu), (idx.bcu, idx.bcx), (idx.bcx, idx.f)])

    def block_sigma(r):
        """Per-row vector (len N_TOTAL) of per-block RMS -- the whitening weights."""
        sig = jnp.ones_like(r)
        for a, b in _wseg:
            sig = sig.at[a:b].set(jnp.sqrt(jnp.maximum(jnp.mean(r[a:b] ** 2), 1e-14)))
        return sig

    def residual_fn_white(p, sig=None):
        """Whitened residual r / sigma_block. sig=None -> adaptive (stop-grad RMS,
        recomputed from the current residual); sig=array -> frozen (fixed weights)."""
        r = residual_fn(p)
        s = jax.lax.stop_gradient(block_sigma(r)) if sig is None else sig
        return r / s

    def loss_fn_white(p, sig=None):
        rw = residual_fn_white(p, sig); return 0.5 * jnp.mean(rw ** 2)

    return SimpleNamespace(
        cfg=c, key=key, B_X=B_X, B_T=B_T, sub_in=sub_in,
        init_pinn=init_pinn, forward=forward, extract_features=extract_features,
        residual_point=residual_point, residual=residual, residual_fn=residual_fn,
        loss_fn=loss_fn, grid_reference=grid_reference, u_ic=u_ic,
        block_sigma=block_sigma, residual_fn_white=residual_fn_white, loss_fn_white=loss_fn_white,
        data=data, idx=idx, N_TOTAL=N_TOTAL,
    )


# ----------------------------------------------------------------------
# MAP I/O
# ----------------------------------------------------------------------
def save_map(params, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    return path


def load_map(path):
    import pickle
    with open(path, "rb") as f:
        return jax.tree.map(jnp.asarray, pickle.load(f))


def sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
