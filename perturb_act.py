"""
What happens to the hardened aberration (and modulation) estimate when the real
sky differs from the sims our response matrix K is built from: in b_nu (the
modulation factor), globally or as a function of l, and in the CMB power
spectrum, globally or as a function of l.

The setting is the one where the mean field (and the scatter) come from sims
that match the data (ACT's own), and only K comes from our sims.  So the
"truth" (the data sims, and the mean-field sims) carries the perturbation, and
K is the fiducial one from boost_act.py.  Two more combinations are reported
for context, and a closure test:

  ref             data, mean field and K all fiducial (the standard run)
  kwrong          data and mean field perturbed, K fiducial      <- the case
  kwrong_mfwrong  data perturbed, mean field and K fiducial (power only: the
                  mean field does not depend on b_nu)
  kmatched        data, mean field and K all perturbed: should be unbiased

Every perturbed sim uses the same seeds as the fiducial sim of the same index
(the same CMB, lensing and noise), so each comparison is a paired difference
with very little scatter.  All sims go through boost_act.py's own functions
(cmb, boost, mat_legs, reconstruct, the response and estimator code); this
script only inserts the perturbation, at one marked place each.

Each shift is also predicted from the response and mean-field sims alone,
K_fid^-1 (K_sky - K_fid) u (+ K_fid^-1 (ybar_sky - ybar_fid) when the mean
field is fiducial).  That is linear in K's separate columns; the data sims have
aberration and modulation at once (the modulation acts on the aberrated sky),
and that cross term moves ~0.5% of any change in the modulation part into the
aberration estimate, which shows up as measured - predicted for b_nu.

Scenarios (--scenarios, comma separated; default
bglobal:0.8,bcoadd,pglobal:0.02,ptilt:0.02):

  bglobal:S        the sky's b_nu is S times the b_nu that K assumes (b at the
                   --noise preset's frequency, 2.047 at 150 GHz), at every l,
                   in T and P.  Expected: no change in the hardened
                   aberration at all, since it only rescales K's modulation
                   columns; the modulation estimate scales by S.
  bcoadd[:A:B]     the sky's b_nu depends on l as in an inverse-noise coadd of
                   two array-bands (noise presets A and B, default f090 and
                   f150): b(l) = sum_nu w_nu(l) b_nu, w_nu ~ 1 / N_nu(l), the
                   beam-deconvolved noise of each preset, separately for T and
                   P.  K still assumes the single b of the --noise preset.
                   Implemented exactly as such a coadd: the modulation part of
                   the boosted sky, b_K beta.n X, is filtered in harmonic space
                   by b(l) / b_K (T and E, B separately).
  bramp:R1:R2      b(l) / b_K linear from R1 at lmin to R2 at lmax.
  bfile:PATH       b(l) from a text file: columns l, b_T, b_P.
  pglobal:EPS      the sky's CMB power is C_l (1 + EPS) at every l.
  ptilt:EPS        C_l (1 + EPS (l - lp) / dl), lp = (lmin + lmax) / 2,
                   dl = (lmax - lmin) / 2: 1 - EPS at lmin, 1 + EPS at lmax
                   (clipped outside), e.g. a beam or transfer-function error.
  pbeam:DFWHM      C_l B_l^2(FWHM + DFWHM) / B_l^2(FWHM), FWHM the preset's
                   beam (arcmin): a beam-width error.
  pfile:PATH       C_l f(l) from a text file: columns l, f_T, f_P.
  The p* scenarios take two optional fields after the value: which spectra
  (TEB, the default; T; or P = E and B), and where the change is made:
    obs   (default) on the boosted sky, before the noise: a calibration,
          beam or transfer-function difference between the data and the sims
    sky   on the unlensed CMB, before lensing and the boost: a different
          cosmology.  For a global change the two are identical.
  e.g. ptilt:0.02:T:obs.  Each scenario's sims are cached under their own
  folder, named after the spec, so different values never mix.

What is simulated, per scenario (the fiducial sims are boost_act.py's own,
read from its cache, and made there if missing, exactly as boost_act.py would):
  response   the legs that change (b: modulation x, y, z; power: all seven),
             --n-response sims
  data       'data mat+' (the half-sum leg; the half-difference leg has no
             CMB and is shared), --n-data sims
  mean field 'mf mat+' (power only), --n-meanfield sims

Run (the boost_act.py options go after --, and must be the ones of the
standard run whose cache holds the fiducial sims):
  python perturb_act.py --nshards 28 --shard $k -- --mask dr6 --noise act \\
      --cmb lensed --lout 10 --n-lens-response 0 --mat-noise
  python perturb_act.py --report -- <the same boost_act.py options>
  python perturb_act.py --selftest -- <the same>     (checks that the
      unperturbed path reproduces boost_act.py's sims; ~10 reconstructions)

Output: <cache>/perturb/<scenario>/sims/*.npy, and from --report
  <cache>/perturb/<scenario>/<combination>/summary.npz (+ plots/, boost_act's
      standard figures, for the combinations in --plots)
  <cache>/perturb/<scenario>/figs/perturbation.png, paired_shift.png,
      amplitude_ref_vs_kwrong.png
  <cache>/perturb/summary/summary.png, summary.md, numbers.npz
"""
import argparse
import os
import sys
import time
import gc
import textwrap

import numpy as np

# ------------------------------------------------------------------ options
_argv = sys.argv[1:]
if "--" in _argv:
    _k = _argv.index("--")
    _own, _base = _argv[:_k], _argv[_k + 1:]
else:
    _own, _base = _argv, []
parser = argparse.ArgumentParser(
    description="b_nu and power-spectrum mismatch between the sky and K's sims",
    epilog="boost_act.py's own options go after --")
parser.add_argument("--scenarios", default="bglobal:0.8,bcoadd,pglobal:0.02,ptilt:0.02")
parser.add_argument("--n-response", type=int, default=100)
parser.add_argument("--n-data", type=int, default=300)
parser.add_argument("--n-meanfield", type=int, default=300)
parser.add_argument("--nshards", type=int, default=1)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--no-fid", action="store_true",
                    help="never make a missing fiducial sim; use only those cached")
parser.add_argument("--report", action="store_true",
                    help="analyse whatever is cached; make no sims")
parser.add_argument("--selftest", action="store_true",
                    help="check the unperturbed path against boost_act.py and stop")
parser.add_argument("--plots", default="ref,kwrong",
                    help="combinations that get boost_act's full figure set "
                         "(all: every one); the rest get summary.npz only")
opt = parser.parse_args(_own)
if opt.report and opt.selftest:
    raise SystemExit("give --report or --selftest, not both")

# boost_act.py reads its options at import; its own sim counts are not used.
sys.argv = ["boost_act.py"] + _base
import boost_act as B                                   # noqa: E402

if not B.MAT:
    raise SystemExit("perturb_act.py works with the mat-noise estimator: add --mat-noise")
if not (B.ABERRATE and B.MODULATE):
    raise SystemExit("perturb_act.py needs --boost full (the default)")

# The harmonic filters below run to the grid's exact limit (fejer1: rings - 1,
# 3599 on the 3' grid), above MLMAX, so a filtered map loses nothing the sims hold.
HLMAX = B.shape[-2] - 1
ELL = np.arange(HLMAX + 1.0)
X6 = np.r_[B.U_TRUE, B.U_TRUE]              # true (u_aberration, u_modulation)
DHAT = B.D_TRUE
C = B.C_KMS
NAMES = (B.ABER_CASE, "MOD")
ROOT = os.path.join(B.CACHE, "perturb")
B_K = float(B.B_NU)                         # the b that K (and the MOD normalisation) assume


def b_of(freq_ghz):
    x = B.utils.h * freq_ghz * 1e9 / (B.utils.k * B.utils.T_cmb)
    return x / np.tanh(x / 2) - 1


def clip_band(a):
    """Hold a per-l array at its lmin value below lmin and its lmax value above."""
    a = np.array(a, float)
    a[:B.LMIN] = a[B.LMIN]
    a[B.LMAX + 1:] = a[B.LMAX]
    return a


def preset_noise(name, pol):
    """Beam-deconvolved noise power of a --noise preset (T: pol 0, P: pol 1),
    as boost_act.act_noise makes it, with that preset's own beam."""
    P = B.NOISE_PRESETS[name]
    white = P["t"] if pol == 0 else (P["p"] if P["p"] is not None else P["t"] * np.sqrt(2))
    knee, alpha = P["knee"][pol], P["alpha"][pol]
    bl2 = B.hp.gauss_beam(P["beam"] * B.utils.arcmin, lmax=HLMAX) ** 2
    red = (np.maximum(ELL, 1.0) / knee) ** alpha if knee > 0 else 0.0
    return (white * B.utils.arcmin / B.TCMB) ** 2 * (1.0 + red) / bl2


class Scenario:
    """One perturbation of the sky.  rT, rP: b(l) / b_K for T and for E, B
    (scalars or per-l arrays), or None.  gT, gP: amplitude factors sqrt(f) of
    T and of E, B (power ratio f), or None.  stage: where a power change is
    made ('obs' or 'sky')."""

    def __init__(self, spec):
        self.spec = spec.strip()
        kind, *par = self.spec.split(":")
        self.kind = kind
        self.rT = self.rP = self.gT = self.gP = None
        self.stage = "obs"
        self.note = ""
        if kind == "bglobal":
            s = float(par[0]) if par else 0.8
            self.rT = self.rP = s
            self.note = f"b_nu of the sky = {s:g} x the b_nu K assumes ({B_K:.3f}), all l"
        elif kind == "bcoadd":
            a, b = (par + ["f090", "f150"][len(par):])[:2]
            r = []
            for pol in (0, 1):
                w = [1 / clip_band(preset_noise(n, pol)) for n in (a, b)]
                bl = (w[0] * b_of(B.NOISE_PRESETS[a]["freq"])
                      + w[1] * b_of(B.NOISE_PRESETS[b]["freq"])) / (w[0] + w[1])
                r.append(bl / B_K)
            self.rT, self.rP = r
            self.note = (f"b_nu(l) of an inverse-noise {a}+{b} coadd "
                         f"(T {B_K * self.rT[B.LMIN]:.3f} at l={B.LMIN} to "
                         f"{B_K * self.rT[B.LMAX]:.3f} at l={B.LMAX}); K assumes {B_K:.3f}")
        elif kind == "bramp":
            r1, r2 = float(par[0]), float(par[1])
            r = clip_band(r1 + (r2 - r1) * (ELL - B.LMIN) / (B.LMAX - B.LMIN))
            self.rT = self.rP = r
            self.note = f"b_nu(l) / b_K linear from {r1:g} at l={B.LMIN} to {r2:g} at l={B.LMAX}"
        elif kind == "bfile":
            t = np.loadtxt(par[0])
            self.rT = clip_band(np.interp(ELL, t[:, 0], t[:, 1])) / B_K
            self.rP = clip_band(np.interp(ELL, t[:, 0], t[:, 2])) / B_K
            self.note = f"b_nu(l) from {par[0]}"
        elif kind in ("pglobal", "ptilt", "pbeam", "pfile"):
            val = par[0]
            fields = par[1] if len(par) > 1 else "TEB"
            self.stage = par[2] if len(par) > 2 else "obs"
            if fields not in ("TEB", "T", "P") or self.stage not in ("obs", "sky"):
                raise SystemExit(f"{spec}: fields TEB, T or P; stage obs or sky")
            if kind == "pglobal":
                f = 1 + float(val)
                fT = fP = f
                self.note = f"CMB power x {f:g} at all l"
            elif kind == "ptilt":
                eps = float(val)
                lp, dl = (B.LMIN + B.LMAX) / 2, (B.LMAX - B.LMIN) / 2
                fT = fP = clip_band(1 + eps * (ELL - lp) / dl)
                self.note = f"CMB power x (1 {-eps:+g}) at l={B.LMIN} to (1 {eps:+g}) at l={B.LMAX}, linear"
            elif kind == "pbeam":
                d = float(val)
                fw = B.BEAM
                fT = fP = clip_band(B.hp.gauss_beam((fw + d) * B.utils.arcmin, lmax=HLMAX) ** 2
                                    / B.hp.gauss_beam(fw * B.utils.arcmin, lmax=HLMAX) ** 2)
                self.note = f"beam FWHM {fw:g}' -> {fw + d:g}' on the sky's CMB"
            else:
                t = np.loadtxt(val)
                fT = clip_band(np.interp(ELL, t[:, 0], t[:, 1]))
                fP = clip_band(np.interp(ELL, t[:, 0], t[:, 2]))
                self.note = f"CMB power x f(l) from {val}"
            one = 1.0 if np.isscalar(fT) else np.ones_like(fT)
            self.gT = np.sqrt(fT) if fields in ("TEB", "T") else one
            self.gP = np.sqrt(fP) if fields in ("TEB", "P") else one
            self.note += f" ({'T, E and B' if fields == 'TEB' else 'T only' if fields == 'T' else 'E and B only'}"
            self.note += f", on the {'observed sky, before the noise' if self.stage == 'obs' else 'unlensed CMB'})"
        else:
            raise SystemExit(f"unknown scenario {spec!r}")
        self.bnu = self.rT is not None
        self.power = self.gT is not None
        self.tag = "".join(c if c.isalnum() or c in "-." else "_" for c in self.spec)
        # the legs this perturbation changes; the others are the fiducial ones
        self.changed = {"response": [f"modulation {a}" for a in "xyz"] if self.bnu
                        else [B.BASE] + B.RESP_LABELS,
                        "data": ["data mat+"],
                        "mf": ["mf mat+"] if self.power else []}

    def dir(self, *p):
        return os.path.join(ROOT, self.tag, *p)

    def is_global(self):
        """True for one constant factor (no l dependence)."""
        a = (self.rT, self.rP) if self.bnu else (self.gT, self.gP)
        return all(np.isscalar(x) for x in a)


SCEN = [Scenario(s) for s in opt.scenarios.split(",") if s.strip()]
if len({s.tag for s in SCEN}) != len(SCEN):
    raise SystemExit("two scenarios with the same spec")


# ------------------------------------------------------------------ the perturbations
def harmonic(tqu, fT, fP):
    """The TQU map filtered in harmonic space: T alm x fT(l), E and B alm x fP(l)."""
    alm = B.curvedsky.map2alm(tqu, lmax=HLMAX)
    fT = np.broadcast_to(np.asarray(fT, float), ELL.shape)
    fP = np.broadcast_to(np.asarray(fP, float), ELL.shape)
    alm[0] = B.curvedsky.almxfl(alm[0], fT)
    alm[1] = B.curvedsky.almxfl(alm[1], fP)
    alm[2] = B.curvedsky.almxfl(alm[2], fP)
    out = B.enmap.zeros(tqu.shape, tqu.wcs, tqu.dtype)
    B.curvedsky.alm2map(alm, out)
    return out


def scale_alm(alm, gT, gP):
    """TEB alm of lmax MLMAX (the sims' draws) with T x gT(l), E and B x gP(l)."""
    out = alm.copy()
    gT = np.broadcast_to(np.asarray(gT, float), ELL.shape)[:B.MLMAX + 1]
    gP = np.broadcast_to(np.asarray(gP, float), ELL.shape)[:B.MLMAX + 1]
    out[0] = B.curvedsky.almxfl(alm[0], gT)
    out[1] = B.curvedsky.almxfl(alm[1], gP)
    out[2] = B.curvedsky.almxfl(alm[2], gP)
    return out


def cmb_p(sc, seed):
    """boost_act.cmb, with a sky-stage power change (on the unlensed CMB)."""
    if sc is None or not sc.power or sc.stage != "sky":
        return B.cmb(seed)
    alm = B.unlensed_alm(seed)
    if B.LENSED:                       # boost_act.cmb's lensed path, alm rescaled
        phi = B.curvedsky.rand_alm(B.CLPP, lmax=B.MLMAX, seed=B.PHI_SEED + seed)
        return B.lens(scale_alm(alm, sc.gT, sc.gP), phi)
    X = B.cmb(seed)                    # unlensed: add the change to rand_map's draw
    d = B.enmap.zeros(X.shape, X.wcs, X.dtype)
    B.curvedsky.alm2map(scale_alm(alm, np.asarray(sc.gT) - 1, np.asarray(sc.gP) - 1), d)
    X += d
    return X


def boost_p(sc, tqu, direction, beta, aberrate, modulate):
    """boost_act.boost, with the sky's b_nu.  pixell's modulated map M is
    split into N, the same map put through pixell's thermodynamic -> linear
    conversion with no boost (beta = 0; its small nonlinear term, ~6% of the
    modulation part, is in every modulated leg of boost_act.py, and is not a
    modulation), and D = M - N = b_K beta.n X, the modulation itself.  The
    sky is then N + D b(l) / b_K: in harmonic space for an l-dependent b,
    which is exactly what a coadd of maps at several frequencies does to it.
    With b(l) = b_K this is M again (to rounding; checked by --selftest)."""
    if sc is None or not sc.bnu or not modulate:
        return B.boost(tqu, direction, beta, aberrate=aberrate, modulate=modulate)
    X = B.boost(tqu, direction, beta, aberrate=True, modulate=False) if aberrate else tqu
    D = B.boost(X.copy(), direction, beta, aberrate=False, modulate=True)   # M, in place on the copy
    N = B.boost(X.copy(), direction, 0.0, aberrate=False, modulate=True)    # no boost, conversion only
    del X
    D -= N                                                                  # the modulation part
    if np.isscalar(sc.rT) and np.isscalar(sc.rP):
        N[0] += sc.rT * D[0]
        N[1:] += sc.rP * D[1:]
        return N
    N += harmonic(D, sc.rT, sc.rP)
    return N


def obs_p(sc, X):
    """An observed-sky power change: the boosted CMB filtered by sqrt(f(l))."""
    if sc is None or not sc.power or sc.stage != "obs":
        return X
    if np.isscalar(sc.gT) and np.isscalar(sc.gP):
        X[0] *= sc.gT
        X[1:] *= sc.gP
        return X
    X += harmonic(X, np.asarray(sc.gT) - 1, np.asarray(sc.gP) - 1)
    return X


def make(sc, label, i):
    """boost_act.one_sim for the labels used here, with sc's perturbation
    (sc None: unperturbed, which must equal boost_act.one_sim)."""
    if label == "mf mat+":
        X = cmb_p(sc, 1_000_000 + 2 * i)
    elif label == "data mat+":
        X = boost_p(sc, cmb_p(sc, 3_000_000 + 2 * i), B.BDIR, B.BETA,
                    B.ABERRATE, B.MODULATE)
    elif label == B.BASE:
        X = cmb_p(sc, 2_000_000 + i)
    else:
        effect, axis = label.split()
        X = boost_p(sc, cmb_p(sc, 2_000_000 + i), B.AXES["xyz".index(axis)], B.BETA,
                    effect == "aberration", effect == "modulation")
    X = obs_p(sc, X)
    if label.endswith("mat+"):
        n1, n2 = B.mat_legs(i, "mf" if label.startswith("mf") else "data")
        X = X + 0.5 * (n1 + n2)        # as one_sim: sky + 0.5 * (n1 + n2)
        del n1, n2
    return B.reconstruct(X)


# ------------------------------------------------------------------ files
def pert_path(sc, label, i):
    return sc.dir("sims", f"{label.replace(' ', '_')}_{i:04d}.npy")


def path(sc, label, i):
    """The file of one leg: perturbed if sc changes it, else the fiducial one."""
    if sc is not None and any(label in v for v in sc.changed.values()):
        return pert_path(sc, label, i)
    return B.cache_path(label, i)


def save(p, arr):
    tmp = f"{p}.{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, p)


FID = {"response": [B.BASE] + B.RESP_LABELS, "data": ["data mat+", "data mat-"],
       "mf": ["mf mat+", "mf mat-"]}
COUNT = {"response": opt.n_response, "data": opt.n_data, "mf": opt.n_meanfield}


def all_jobs():
    """Every (scenario or None, label, i), stage by stage (K first), sim by sim,
    so that a run cut short has complete pairs for every scenario."""
    jobs = []
    for stage in ("response", "data", "mf"):
        for i in range(COUNT[stage]):
            if not opt.no_fid:
                jobs += [(None, lab, i) for lab in FID[stage]]
            for sc in SCEN:
                jobs += [(sc, lab, i) for lab in sc.changed[stage]]
    return jobs


def run_sims():
    for sc in SCEN:
        os.makedirs(sc.dir("sims"), exist_ok=True)
        print(f"scenario {sc.spec}: {sc.note}")
    os.makedirs(B.SIMS, exist_ok=True)
    mine = [j for k, j in enumerate(all_jobs()) if k % opt.nshards == opt.shard]
    todo = [j for j in mine if not os.path.exists(
        B.cache_path(j[1], j[2]) if j[0] is None else pert_path(*j))]
    print(f"shard {opt.shard} of {opt.nshards}: {len(todo)} of its {len(mine)} sims to make "
          f"(fiducial ones go to {B.SIMS}/, identical to boost_act.py's)", flush=True)
    t0 = time.time()
    for n, (sc, lab, i) in enumerate(todo, 1):
        t = time.time()
        if sc is None:
            if not os.path.exists(B.cache_path(lab, i)):
                B.make_and_cache((lab, i))
        else:
            save(pert_path(sc, lab, i), make(sc, lab, i))
        gc.collect()
        rate = (time.time() - t0) / n
        print(f"   {'fiducial' if sc is None else sc.spec:>16s}  {lab:14s} {i:4d}  "
              f"{time.time() - t:5.0f}s   {n}/{len(todo)}, "
              f"{rate * (len(todo) - n) / 3600:.1f} h left", flush=True)
    print("shard done", flush=True)


def selftest():
    """The unperturbed path, and the perturbation code at zero perturbation,
    against boost_act.one_sim (identical, or equal to rounding)."""
    cases = [(None, "identity"), (Scenario("bglobal:1"), "b x 1 (modulation split off and put back)"),
             (Scenario("bramp:1:1"), "b(l) = b_K (harmonic filter path)"),
             (Scenario("ptilt:0"), "power x 1 (harmonic filter path)"),
             (Scenario("pglobal:0:TEB:sky"), "power x 1 on the unlensed CMB")]
    worst = 0.0
    for label in ("modulation x", "data mat+", B.BASE):
        ref = (np.load(B.cache_path(label, 0)) if os.path.exists(B.cache_path(label, 0))
               else B.one_sim(label, 0))
        for sc, what in cases:
            if label == B.BASE and sc is not None and sc.bnu:
                continue
            a = make(sc, label, 0)
            rel = np.abs(a - ref).max() / np.abs(ref).max()
            kms = np.abs(yvec(a[None]) - yvec(ref[None])).max() * C
            worst = max(worst, kms)
            print(f"   {label:13s} {what:45s} max |diff| / max |ref| = {rel:.1e};  "
                  f"largest change in the six L=1 components {kms:.1e} km/s", flush=True)
            gc.collect()
    # The harmonic paths are exact to ~1e-6 of the reconstruction (a few 1e-3 km/s):
    # the SHT of the modulation part of a pixell-modulated map.
    print(f"selftest {'PASSED' if worst < 0.01 else 'FAILED'} (largest change "
          f"{worst:.1e} km/s per sim; the test is < 0.01 km/s)")


# ------------------------------------------------------------------ report
def have(sc, labels, i):
    return all(os.path.exists(path(sc, l, i)) for l in labels)


def load(sc, labels, idx):
    return np.array([[np.load(path(sc, l, i)) for i in idx] for l in labels])


def per_sim_K(diff):
    """Each response sim's own K (sim, 6, 6), as boost_act.system builds the mean."""
    V = np.concatenate([B.l1_vector(B.est_alm(diff, n)) for n in NAMES], -1) / B.BETA
    n = V.shape[2]
    return np.moveaxis(V, 2, 0).reshape(n, 6, 6).transpose(0, 2, 1)


def yvec(x):
    """The six-component estimator vector (aberration T+P xyz, MOD xyz) of each sim."""
    return np.concatenate([B.l1_vector(B.est_alm(x, n)) for n in NAMES], -1)


def mean_err(a):
    a = np.asarray(a)
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros(a.shape[1:])


SRC = open(B.__file__).read()
# The end of boost_act.main(), from the analysis on: its lines up to the first
# one back at the left margin (the end of main).
_lines = SRC[SRC.index("    results = analyse(mf, diff, dat)"):].split("\n")
_end = next(k for k, ln in enumerate(_lines) if ln.strip() and not ln[0].isspace())
TAIL = "\n".join(_lines[:_end]) + "\n"
TAIL = TAIL.replace("    if SPLIT or MAT:", "    if False:", 1)
_mk = "    import boost_act_plots\n    boost_act_plots.make_all(path)"
assert TAIL.count(_mk) == 1
TAIL = textwrap.dedent(TAIL.replace(_mk, "    if MAKE_PLOTS:\n" + textwrap.indent(_mk, "    ")))


def standard_summary(out, mf, mf_noise, diff, dat, plots):
    """boost_act.py's own summary.npz (and figures) for one combination, by
    running the end of its main() on these arrays."""
    g = vars(B)
    g.update(mf=mf, mf_noise=mf_noise, mf_std=mf, diff=diff, dat=dat, n_lens=0, n_band=0,
             OUT=out, MAKE_PLOTS=plots)
    B.args.n_meanfield, B.args.n_response, B.args.n_data = len(mf), diff.shape[2], len(dat)
    os.makedirs(out, exist_ok=True)
    exec(TAIL, g)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 150, "savefig.bbox": "tight"})
    INK, GREY = "#222222", "#8a8a8a"
    COL = {"kwrong": "#2a6fdb", "kwrong_mfwrong": "#d1495b", "kmatched": "#c98a1c"}
    WHAT = {"ref": "data, mean field and K fiducial",
            "kwrong": "data and mean field perturbed, K fiducial",
            "kwrong_mfwrong": "data perturbed, mean field and K fiducial",
            "kmatched": "data, mean field and K perturbed (closure)"}
    plots = set(WHAT) if opt.plots == "all" else set(opt.plots.split(","))
    os.makedirs(os.path.join(ROOT, "summary"), exist_ok=True)
    rows, store = [], {}
    for sc in SCEN:
        print(f"\n================ {sc.spec}: {sc.note}", flush=True)
        ri = [i for i in range(opt.n_response)
              if have(None, FID["response"], i) and have(sc, FID["response"], i)]
        di = [i for i in range(opt.n_data)
              if have(None, FID["data"], i) and have(sc, FID["data"], i)]
        mi = [i for i in range(opt.n_meanfield)
              if have(None, FID["mf"], i) and have(sc, FID["mf"], i)]
        print(f"paired sims: {len(ri)} response, {len(di)} data, {len(mi)} mean field")
        if len(ri) < 2 or len(di) < 2 or len(mi) < 3:
            print("too few yet; skipped")
            continue
        legs_order = B.RESP_LABELS + [B.BASE]

        def diff_of(s):
            legs = load(s, legs_order, ri)
            return (legs[:-1] - legs[-1]).reshape((2, 3) + legs.shape[1:])
        dif = {"f": diff_of(None), "p": diff_of(sc)}
        mfm = load(None, ["mf mat-"], mi)[0]
        mf = {"f": load(None, ["mf mat+"], mi)[0] - mfm, "p": load(sc, ["mf mat+"], mi)[0] - mfm}
        dm = load(None, ["data mat-"], di)[0]
        dat = {"f": load(None, ["data mat+"], di)[0] - dm, "p": load(sc, ["data mat+"], di)[0] - dm}

        Kj = {k: per_sim_K(v) for k, v in dif.items()}           # (n_resp, 6, 6)
        K = {k: v.mean(0) for k, v in Kj.items()}
        Kfi = np.linalg.inv(K["f"])
        ym = {k: yvec(v) for k, v in mf.items()}                  # (n_mf, 6)
        yd = {k: yvec(v) for k, v in dat.items()}                 # (n_data, 6)
        combos = {"ref": ("f", "f", "f"), "kwrong": ("p", "f", "p"),
                  "kwrong_mfwrong": ("f", "f", "p"), "kmatched": ("p", "p", "p")}
        if not sc.power:                 # the mean field does not depend on b_nu
            del combos["kwrong_mfwrong"]
        u = {}
        for c, (m, k, d) in combos.items():
            u[c] = (yd[d] - ym[m].mean(0)) @ np.linalg.inv(K[k]).T * C   # (n_data, 6) km/s
        # what the hardened estimates pick up, per unit true boost: K_f^-1 K_p
        Mj = Kfi @ Kj["p"]
        M, Merr = mean_err(Mj)

        def along(v):
            """(..., 6) km/s -> (..., 2): aberration and modulation parts along the input."""
            return np.stack([v[..., 0:3] @ DHAT, v[..., 3:6] @ DHAT], -1)

        def stat(*parts):
            """Mean and error of a sum of independent per-sim sets (n_k, 6): the
            vector mean, and the along-input mean and error (2,)."""
            vec = sum(p.mean(0) for p in parts)
            al = sum(along(p).mean(0) for p in parts)
            er = np.sqrt(sum(along(p).var(0, ddof=1) / len(p) for p in parts))
            return vec, al, er
        pred_K = (Kfi @ (Kj["p"] - Kj["f"]) @ X6) * C            # per response sim (n_resp, 6)
        pred_mf = (ym["p"] - ym["f"]) @ Kfi.T * C                 # per mean-field sim (n_mf, 6)
        pred = {"kwrong": stat(pred_K), "kwrong_mfwrong": stat(pred_K, pred_mf),
                "kmatched": (np.zeros(6), np.zeros(2), np.zeros(2))}
        res = {}
        for c in combos:
            if c == "ref":
                continue
            sh = u[c] - u["ref"]                                  # paired, (n_data, 6)
            # the mean field's own MC error (paired too), for the combinations that change it
            mfpart = {"kwrong": -(ym["p"] - ym["f"]) @ Kfi.T * C,
                      "kmatched": -(ym["p"] @ np.linalg.inv(K["p"]).T - ym["f"] @ Kfi.T) * C,
                      "kwrong_mfwrong": None}[c]
            # the data part carries the mean; the mean-field part only adds its error
            vec, al, er = stat(sh)
            if mfpart is not None:
                er = np.hypot(er, np.sqrt(along(mfpart).var(0, ddof=1) / len(mfpart)))
            pv, pa, pe = pred[c]
            res[c] = dict(shift=sh, mean=vec, along_mean=al, along_err=er, pred=pv,
                          pred_along=pa, pred_err=pe, along=along(sh))
            for j, nm in enumerate(("aberration", "modulation")):
                s = slice(3 * j, 3 * j + 3)
                rows.append(dict(scen=sc.spec, note=sc.note, combo=c, est=nm, along=al[j],
                                 along_err=er[j], mag=np.linalg.norm(vec[s]), pred=pa[j],
                                 pred_err=pe[j], pred_mag=np.linalg.norm(pv[s]),
                                 n=(len(ri), len(di), len(mi))))
                print(f"   {c:15s} {nm:10s} shift along input {al[j]:+9.2f} +- {er[j]:6.2f} km/s "
                      f"(|vector| {np.linalg.norm(vec[s]):8.2f});  predicted from K and the "
                      f"mean field alone {pa[j]:+9.2f} +- {pe[j]:6.2f}")
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        print("   K_fid^-1 K_pert - 1 (rows: estimates ab xyz, mod xyz; columns: true ab xyz, mod xyz)")
        print(textwrap.indent(str(M - np.eye(6)), "      "))
        store[sc.tag] = dict(M=M, Merr=Merr, Kf=K["f"], Kp=K["p"],
                             **{f"u_{c}": u[c] for c in combos},
                             **{f"pred_{c}": res[c]["pred"] for c in res},
                             **{f"mean_{c}": res[c]["mean"] for c in res},
                             **{f"along_{c}": res[c]["along_mean"] for c in res},
                             **{f"along_err_{c}": res[c]["along_err"] for c in res},
                             **{f"pred_along_{c}": res[c]["pred_along"] for c in res},
                             n=[len(ri), len(di), len(mi)])

        # boost_act's own summaries (and figures) for each combination
        for c, (m, k, d) in combos.items():
            print(f"\n----- {sc.spec}  {c}: {WHAT[c]}", flush=True)
            standard_summary(sc.dir(c), mf[m], mfm, dif[k], dat[d], c in plots)

        # ---------------- figures for this scenario
        os.makedirs(sc.dir("figs"), exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained",
                               gridspec_kw=dict(width_ratios=[1.1, 1]))
        a = ax[0]
        l = ELL[2:]
        if sc.bnu:
            for r, lab, ls in ((sc.rT, "T", "-"), (sc.rP, "E, B", "--")):
                a.plot(l, (np.broadcast_to(np.asarray(r, float), ELL.shape) * B_K)[2:],
                       color=INK if lab == "T" else GREY, ls=ls, lw=1.6, label=f"sky, {lab}")
            a.axhline(B_K, color="#2a6fdb", lw=1.2, label=f"what K assumes ({B_K:.3f})")
            a.set_ylabel(r"modulation factor $b_\nu$")
        else:
            for g_, lab, ls in ((sc.gT, "T", "-"), (sc.gP, "E, B", "--")):
                a.plot(l, (np.broadcast_to(np.asarray(g_, float), ELL.shape) ** 2)[2:],
                       color=INK if lab == "T" else GREY, ls=ls, lw=1.6, label=f"sky / sims, {lab}")
            a.axhline(1, color="#2a6fdb", lw=1.2, label="what K assumes (fiducial)")
            a.set_ylabel(r"CMB power: sky / sims")
        a.axvspan(B.LMIN, B.LMAX, color="0.93", zorder=0, label="l range the estimators use")
        a.set_xlabel(r"$\ell$")
        a.set_xlim(0, B.MLMAX)
        a.legend(frameon=False, fontsize=8)
        a.set_title(textwrap.fill(sc.note, 70), fontsize=9)
        a = ax[1]
        D = M - np.eye(6)
        v = max(np.abs(D).max(), 1e-12)
        im = a.imshow(D, cmap="RdBu_r", vmin=-v, vmax=v)
        for p in range(6):
            for q in range(6):
                a.text(q, p, f"{D[p, q]:+.3f}" if v > 1e-3 else f"{D[p, q]:+.1e}",
                       ha="center", va="center", fontsize=7,
                       color="white" if abs(D[p, q]) > 0.6 * v else INK)
        lab6 = [f"ab {x}" for x in "xyz"] + [f"mod {x}" for x in "xyz"]
        a.set_xticks(range(6), lab6)
        a.set_yticks(range(6), lab6)
        a.set_xlabel("true boost component")
        a.set_ylabel("hardened estimate")
        a.axhline(2.5, color=INK, lw=0.6)
        a.axvline(2.5, color=INK, lw=0.6)
        a.set_title(r"$K_{\rm fid}^{-1}K_{\rm sky} - 1$: what each estimate picks up" + "\n"
                    f"per unit true boost ({len(ri)} paired response sims); for the input "
                    f"boost: aberration {pred['kwrong'][1][0]:+.2f}, modulation "
                    f"{pred['kwrong'][1][1]:+.2f} km/s", fontsize=9)
        fig.colorbar(im, ax=a, shrink=0.8)
        fig.savefig(sc.dir("figs", "perturbation.png"))
        plt.close(fig)

        cs = [c for c in res]
        fig, ax = plt.subplots(len(cs), 2, figsize=(12, 3.2 * len(cs)), layout="constrained",
                               squeeze=False)
        for r_, c in enumerate(cs):
            for j, nm in enumerate(("aberration", "modulation")):
                a = ax[r_, j]
                x = res[c]["along"][:, j]
                a_m, a_e = res[c]["along_mean"][j], res[c]["along_err"][j]
                p_m = res[c]["pred_along"][j]
                a.hist(x, bins=30, color=COL[c], alpha=0.55)
                a.axvline(a_m, color=COL[c], lw=2, label=f"mean {a_m:+.2f} +- {a_e:.2f}")
                a.axvline(p_m, color=INK, lw=1.2, ls="--",
                          label=f"predicted from K, mean field: {p_m:+.2f}")
                a.axvline(0, color=GREY, lw=0.8)
                a.set_xlabel(f"{nm}: shift along the input, vs ref [km/s]")
                a.set_ylabel("data sims")
                a.set_title(f"{nm}, hardened — {WHAT[c]}", fontsize=9)
                a.legend(frameon=False, fontsize=8)
        fig.suptitle(f"{sc.spec}: paired shift per data sim (same skies and noise as ref)")
        fig.savefig(sc.dir("figs", "paired_shift.png"))
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for j, nm in enumerate(("aberration", "modulation")):
            a = ax[j]
            s = slice(3 * j, 3 * j + 3)
            x0, x1 = u["ref"][:, s] @ DHAT, u["kwrong"][:, s] @ DHAT
            bins = np.histogram_bin_edges(np.r_[x0, x1], 30)
            a.hist(x0, bins=bins, color=GREY, alpha=0.5, label=f"ref: {x0.mean():+.1f} +- {x0.std(ddof=1):.0f} (sd)")
            a.hist(x1, bins=bins, color=COL["kwrong"], alpha=0.5,
                   label=f"kwrong: {x1.mean():+.1f} +- {x1.std(ddof=1):.0f}")
            a.axvline(C * np.linalg.norm(B.U_TRUE), color=INK, lw=1.5, label="input")
            a.set_xlabel(r"$v_\parallel$ along the input [km/s]")
            a.set_ylabel("data sims")
            a.set_title(f"{nm}, hardened", fontsize=9)
            a.legend(frameon=False, fontsize=8)
        fig.suptitle(f"{sc.spec}: the sky perturbed, K from fiducial sims (mean field matches the sky)")
        fig.savefig(sc.dir("figs", "amplitude_ref_vs_kwrong.png"))
        plt.close(fig)

    if not rows:
        print("\nnothing to summarise yet")
        return
    # ---------------- across scenarios
    np.savez(os.path.join(ROOT, "summary", "numbers.npz"),
             **{f"{t}.{k}": v for t, d in store.items() for k, v in d.items()},
             scenarios=np.array([s.spec for s in SCEN]))
    scen = [s.spec for s in SCEN if s.tag in store]
    cs = ["kwrong", "kwrong_mfwrong", "kmatched"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    w = 0.26
    for j, nm in enumerate(("aberration", "modulation")):
        a = ax[j]
        for k, c in enumerate(cs):
            xs, ys, es, ps = [], [], [], []
            for n, s in enumerate(scen):
                r = [q for q in rows if q["scen"] == s and q["combo"] == c and q["est"] == nm]
                if r:
                    xs.append(n + (k - 1) * w)
                    ys.append(r[0]["along"])
                    es.append(r[0]["along_err"])
                    ps.append(r[0]["pred"])
            if not xs:
                continue
            a.bar(xs, ys, w * 0.9, yerr=es, color=COL[c], label=WHAT[c], capsize=2,
                  error_kw=dict(lw=0.8))
            a.scatter(xs, ps, marker="D", s=14, color=INK, zorder=3,
                      label="predicted from K and the mean field" if k == 0 else None)
            for x_, y_, e_ in zip(xs, ys, es):
                end = y_ + e_ if y_ >= 0 else y_ - e_         # label past the error bar
                a.annotate(f"{y_:+.1f}", (x_, end), xytext=(0, 3 if y_ >= 0 else -10),
                           textcoords="offset points", ha="center", fontsize=7, color=INK)
        a.axhline(0, color=INK, lw=0.6)
        a.set_xticks(range(len(scen)), scen, fontsize=8)
        a.set_ylabel("mean shift along the input vs ref [km/s]")
        a.set_title(f"{nm}, hardened", fontsize=10)
        if nm == "modulation":
            a.set_yscale("symlog", linthresh=10)
            lo, hi = a.get_ylim()
            a.set_ylim(lo * 4 if lo < -10 else lo, hi * 4 if hi > 10 else hi)   # room for labels
        else:
            lo, hi = a.get_ylim()
            a.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    ax[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Sky vs our sims: shift of the hardened estimates (input {C * np.linalg.norm(B.U_TRUE):.0f} km/s)")
    fig.savefig(os.path.join(ROOT, "summary", "summary.png"))
    plt.close(fig)

    with open(os.path.join(ROOT, "summary", "summary.md"), "w") as f:
        f.write(f"# Sky vs K-sims mismatch ({B.CACHE_TAG}, mat noise)\n\n"
                f"Shift of the mean hardened estimate along the input direction, relative to the "
                f"fiducial run on the same data skies, in km/s (input "
                f"{C * np.linalg.norm(B.U_TRUE):.0f} km/s). 'Predicted' uses only the response "
                f"sims and the mean-field sims: K_fid^-1 (K_sky - K_fid) u (+ K_fid^-1 (ybar_sky - "
                f"ybar_fid) when the mean field is fiducial).\n\n")
        for s in SCEN:
            f.write(f"- **{s.spec}**: {s.note}\n")
        f.write("\n| scenario | combination | estimate | shift along input | |shift| | predicted | n (resp, data, mf) |\n"
                "|---|---|---|---|---|---|---|\n")
        for q in rows:
            f.write(f"| {q['scen']} | {q['combo']} | {q['est']} | {q['along']:+.2f} ± {q['along_err']:.2f} "
                    f"| {q['mag']:.2f} | {q['pred']:+.2f} ± {q['pred_err']:.2f} | {q['n']} |\n")
        f.write("\nCombinations: " + "; ".join(f"{c}: {WHAT[c]}" for c in WHAT) + ".\n")
        f.write("\nThe prediction is linear in K's separate columns (aberration alone, modulation "
                "alone).  The data sims have both at once, so the modulation acts on the already "
                "aberrated sky; that cross term, which no column of K contains, puts about 0.5% "
                "of any change in the modulation part into the aberration estimate.  In the b_nu "
                "scenarios it is the difference between measured and predicted (checked on one "
                "sky: with the aberration switched off in the data the difference goes to zero).\n")
    print(f"\nwrote {ROOT}/summary/ (summary.png, summary.md, numbers.npz) and "
          f"{ROOT}/<scenario>/ (figs/, and boost_act's summary for each combination)")


if __name__ == "__main__":
    if opt.selftest:
        selftest()
    elif opt.report:
        report()
    else:
        run_sims()