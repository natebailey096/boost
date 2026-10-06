"""
Doppler boost of the CMB, aberration and modulation, measured with two
quadratic estimators on simulations with ACT-like noise, on the full sky or
on the ACT DR6 lensing mask.  Successor to aberration_dr6.py, with a
different response and options for the mask and the noise level.

    python boost_act.py --mask dr6 --noise f150 --n-meanfield 800 \
        --n-response 60 --n-data 400

    --mask   none | dr6    the full sky, or the DR6 lensing mask (healpix
                           nside 4096, equatorial, ~0.8 GB; --mask-file says
                           where it is, default the working directory):
    https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/maps/baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits
    --noise  f150 | act    white noise in T of 24 (f150) or 14 (act) uK-arcmin,
                           sqrt2 x that in P.  Beam, 1/f knees and slopes are
                           the same for both.

Needs the falafel branch darby_birefringence_patchytau and the tempura branch
darbys_tau_edits2 (the ACT DR6 screening versions); it stops at start-up if
it finds the master versions instead.

To split the sims across independent jobs (e.g. a SLURM array, --array=0-9),
give every job the same options and sim counts plus
    --nshards 10 --shard $SLURM_ARRAY_TASK_ID
Each job makes and caches the sims i with i % nshards == shard.  When all
have finished, run once more without --shard/--nshards: that run makes any
sims still missing and writes the complete report and figures.

Within one job, --nproc N makes N sims at a time in N worker processes (set
OMP_NUM_THREADS = cpus / N, and allow memory for N sims).  The workers are
started fresh ("spawn") and each repeats the set-up below once, because
forked ones deadlock in OpenMP code after the set-up has used OpenMP.  It
combines with --shard/--nshards.

Estimators, each reduced to its L = 1 vector and scaled so that on the full
sky, for its own effect, it estimates the boost vector u = beta * d:
    aberration  lensing QE, TT+TE+EE          phi = -u . n
    modulation  TT amplitude-modulation QE    f = b u . n,  b = x coth(x/2) - 1
                (b = 2.05 at 150 GHz): falafel's qe_mask, the product of the
                Wiener- and inverse-variance-filtered T maps, normalised
                with tempura's norm_general.qtt("amp"), which on this branch
                equals norm_tau.qtt (checked at start-up).  Its output is
                +f, i.e. -tau in the screening convention.

The mask multiplies the map before either estimator sees it; both keep their
full-sky normalisations, and the mask is dealt with by simulation:
    A  mean field  unboosted, masked, noisy sims.  On the mask the modulation
                   QE reconstructs the mask itself, so its mean field is
                   large (printed as |mean field| / |K u_in|); it is
                   subtracted.  On the full sky it is zero in expectation
                   but still measured: its scatter sets the joint weights.
    B  response    noise-free masked sims, each reconstructed once unboosted
                   and once boosted by +beta along x, y and z, with
                   aberration only and with modulation only (7
                   reconstructions per sim).  (boosted - unboosted) / beta
                   is a column of the 6x6 response K of [aberration QE,
                   modulation QE] to [aberration, modulation]; the mean field
                   cancels sim by sim, and K absorbs everything the mask and
                   the normalisations do at L = 1.  The difference is
                   one-sided, so K has a truncation bias, expected at order
                   beta (~0.1%) and not in the quoted errors.  Per sim the
                   aberration legs differ from a central difference by a few
                   per cent (the 4' shift is not small at high l), but that
                   averages out: at reduced resolution on a stand-in mask the
                   two agreed within the Monte Carlo error.
    C  data        masked sims with the full boost (pixell's dipole) plus
                   noise.

    separate   u_aber and u_mod solved from K together, so each is free of
               the other effect and they are not assumed equal
    joint      one u for both effects (equal in these sims): per axis, a
               weighted average of the separate aberration and modulation
               estimates, w = s_mod^2 / (s_ab^2 + s_mod^2) on aberration,
               with s the sd of that estimate over the mean-field sims.  Each
               separate estimate is unbiased for the same u, so any weights
               that sum to 1 are; these are the inverse-variance ones when
               the two are uncorrelated.  The 6x6 covariance is not used.

The error on a mean is sd / sqrt(n_data), the scatter of the data sims
alone.  The Monte Carlo errors of the mean field and of K are shared by every
data sim and are left out: the mean field's is about sqrt(n_data / n_mf)
times this one (it understates the total by sqrt(1 + n_data / n_mf), 22% for
800 mean-field and 400 data sims), so use n_mf >> n_data; K's is small, as
each response sim cancels most of its CMB.

Each reconstruction is cached in cache_boost/<mask>_<noise>/sims/ (one .npy
per sim, named by stage and index; the seeds follow the index), so an
interrupted run resumes and a larger N only adds sims.  If you edit a
constant below, or point --mask-file at a different mask, change CACHE_TAG
too: each tag keeps its own sims.  The DR6 mask on the CAR grid is cached in
cache_boost/masks/.  Every run, shard or not, then reports on all sims cached
so far (by any shard) and writes summary.npz and the figures (plots/, drawn
by boost_act_plots.py) into cache_boost/<mask>_<noise>/, beside sims/, so a
shard that finishes early leaves a partial snapshot there.  Unlensed Gaussian
CMB, homogeneous noise high-passed below LMIN.
"""

import argparse
import multiprocessing
import os
import time
import warnings

import numpy as np

MASK_URL = ("https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/maps/"
            "baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits")

# The options are read before the heavy imports below, so that --help and a
# mistyped option answer at once.
parser = argparse.ArgumentParser(
    description="Doppler boost of the CMB: aberration and modulation, "
                "measured with two quadratic estimators")
parser.add_argument("--mask", default="dr6", choices=["none", "dr6"],
                    help="none: full sky.  dr6: the ACT DR6 lensing mask")
parser.add_argument("--mask-file", default=os.path.basename(MASK_URL),
                    help="the DR6 healpix mask (--mask dr6 only)")
parser.add_argument("--noise", default="f150", choices=["f150", "act", "none"],
                    help="white noise level in T: f150 is 24 uK-arcmin, act "
                         "is 14 uK-arcmin (P is sqrt2 x); the beam and the "
                         "1/f knees are the same")
parser.add_argument("--n-meanfield", type=int, default=800)
parser.add_argument("--n-response", type=int, default=60)
parser.add_argument("--n-data", type=int, default=400)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--nshards", type=int, default=1)
parser.add_argument("--nproc", type=int, default=1,
                    help="sims made at the same time in this job, each in its "
                         "own worker process; give each worker "
                         "OMP_NUM_THREADS = (cpus for the job) / nproc, and "
                         "memory for nproc sims at once")
args = parser.parse_args()

import healpy as hp                                            # noqa: E402
import camb                                                    # noqa: E402
import pytempura                                               # noqa: E402
from falafel import qe                                         # noqa: E402
from pixell import (enmap, curvedsky, aberration, reproject,   # noqa: E402
                    utils)

# qe_mask and the lensing QE are taken from the screening branches, whose
# APIs differ from master (get_norms and norm_general.qtt lose their
# weight-spectrum argument).  norm_tau and qe_tau_pol exist only there.
if not (hasattr(pytempura, "norm_tau") and hasattr(qe, "qe_tau_pol")):
    raise SystemExit(
        "boost_act.py needs the ACT DR6 screening branches:\n"
        "  falafel  darby_birefringence_patchytau  (found "
        f"{'it' if hasattr(qe, 'qe_tau_pol') else 'another version'})\n"
        "  tempura  darbys_tau_edits2              (found "
        f"{'it' if hasattr(pytempura, 'norm_tau') else 'another version'})")
# The falafel branch passes tweak=True to map2alm, which pixell 0.32 ignores
# and warns about on every call.
warnings.filterwarnings("ignore", message="The tweak argument is deprecated")

# ---------------------------------------------------------------- settings
LMIN, LMAX, MLMAX = 600, 3000, 3500
LOUT = 5                                  # highest reconstruction L kept
RES = 3.0 * utils.arcmin                  # fejer1 grid: exact SHTs to l = 3599
NOISE_LEVELS = {"f150": 24.0, "act": 14.0, "none": 0.0}    # uK-arcmin in T (P is sqrt2 x)
NOISE_T = NOISE_LEVELS[args.noise]
BEAM = 1.42                               # FWHM, arcmin
TCMB, C_KMS = 2.7255e6, 299792.458        # uK, km/s
KNEE_T, KNEE_P = 3000.0, 475.0            # 1/f knees
ALPHA_T, ALPHA_P = -3.0, -4.5             # 1/f slopes
FREQ = 150e9                              # Hz; sets the modulation factor b
BETA, BDIR = aberration.beta, aberration.dir_equ     # BDIR is (ra, dec), rad
AXES = [(0.0, 0.0), (np.pi / 2, 0.0), (0.0, np.pi / 2)]   # x, y, z as (ra, dec)
EST = ["TT", "TE", "EE"]
CASES = {"TT": ["TT"], "TE": ["TE"], "EE": ["EE"], "T+P": EST}
ABER_CASE = "T+P"                         # lensing combination used for u_aber
CACHE_TAG = f"{args.mask}_{args.noise}"   # new value whenever a setting changes
CACHE = os.path.join("cache_boost", CACHE_TAG)     # summary.npz and plots/
SIMS = os.path.join(CACHE, "sims")                 # one .npy per sim

X_NU = utils.h * FREQ / (utils.k * utils.T_cmb)
B_NU = X_NU / np.tanh(X_NU / 2) - 1       # linearised-T modulation factor
D_TRUE = np.array([np.cos(BDIR[1]) * np.cos(BDIR[0]),
                   np.cos(BDIR[1]) * np.sin(BDIR[0]),
                   np.sin(BDIR[1])])
U_TRUE = BETA * D_TRUE
GAL = hp.Rotator(coord=["C", "G"]).mat    # equatorial -> galactic vectors

# pixell's interpol_map (0.32.7 and master) doubles a full-sky map for the
# NUFFT by flipping ra where it should flip dec.  The seam this leaves at the
# poles rings: 26% of the map rms in the polar rows, ~5e-4 at |dec| < 62.
# Doubling the map here makes the boost exact to 1e-11.
_interpol = aberration.interpol_map


def _interpol_fixed(imap, pixs, epsilon=None, nthread=None, ydouble=False):
    if ydouble:
        flip = np.roll(imap[..., ::-1, :], imap.shape[-1] // 2, -1)
        imap = enmap.enmap(np.concatenate([imap, flip], -2), imap.wcs)
    return _interpol(imap, pixs, epsilon=epsilon, nthread=nthread)


aberration.interpol_map = _interpol_fixed

# ---------------------------------------------------------------- spectra
IN_WORKER = __name__ == "__mp_main__"       # a spawned --nproc worker
if not IN_WORKER:
    print("CAMB ...", flush=True)
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965,
                       As=2.1e-9, tau=0.06)
pars.set_for_lmax(MLMAX + 500)
cls = camb.get_results(pars).get_cmb_power_spectra(
    pars, raw_cl=True, spectra=["unlensed_scalar"])["unlensed_scalar"]
cltt, clee, clbb, clte = cls[:MLMAX + 1].T          # dimensionless (dT/T)^2

ell = np.arange(MLMAX + 1.0)
bl2 = hp.gauss_beam(BEAM * utils.arcmin, lmax=MLMAX) ** 2


def act_noise(white, knee, alpha):
    nl = ((white * utils.arcmin / TCMB) ** 2
          * (1.0 + (np.maximum(ell, 1.0) / knee) ** alpha) / bl2)
    nl[LMAX + 1:] = nl[LMAX]
    nl[:LMIN] = 0.0
    return nl


nltt = act_noise(NOISE_T, KNEE_T, ALPHA_T)
nlee = act_noise(NOISE_T * np.sqrt(2.0), KNEE_P, ALPHA_P)

ucls = {"TT": cltt, "EE": clee, "BB": clbb, "TE": clte}
tcls = {"TT": cltt + nltt, "EE": clee + nlee, "BB": clbb + nlee, "TE": clte}


def inverse_variance(total):
    f = np.zeros(MLMAX + 1)
    f[LMIN:LMAX + 1] = 1.0 / total[LMIN:LMAX + 1]
    return f


FILTERS = [inverse_variance(tcls[k]) for k in ("TT", "EE", "BB")]

# Full-sky normalisations.  A lensing case sums the unnormalised estimators
# and divides by sum(1/A_L), the inverse-variance combination.
norms = pytempura.get_norms(EST, ucls, tcls, LMIN, LMAX, k_ellmax=MLMAX)
NORM = {}
for case, members in CASES.items():
    inv = sum(1.0 / np.asarray(norms[e][0][1:LOUT + 1]) for e in members)
    NORM[case] = np.concatenate([[0.0], 1.0 / inv])       # L = 0 dropped
amp_norm = np.asarray(pytempura.norm_general.qtt(
    "amp", LOUT, LMIN, LMAX, cltt, tcls["TT"]))[0, 1:]
tau_norm = np.asarray(pytempura.norm_tau.qtt(LOUT, LMIN, LMAX, cltt,
                                             tcls["TT"]))[1:]
if not np.allclose(amp_norm, tau_norm, rtol=1e-8, atol=0):
    raise SystemExit(f"tempura: norm_general 'amp' {amp_norm} and norm_tau "
                     f"{tau_norm} disagree; check the tempura build")
NORM["MOD"] = np.concatenate([[0.0], amp_norm])

PS_CMB = np.zeros((3, 3, MLMAX + 1))
PS_CMB[0, 0], PS_CMB[1, 1], PS_CMB[2, 2] = cltt, clee, clbb
PS_CMB[0, 1] = PS_CMB[1, 0] = clte
PS_NOISE = np.zeros((3, 3, MLMAX + 1))
PS_NOISE[0, 0], PS_NOISE[1, 1], PS_NOISE[2, 2] = nltt, nlee, nlee

# ---------------------------------------------------------------- mask
shape, wcs = enmap.fullsky_geometry(res=RES)
px = qe.pixelization(shape=shape, wcs=wcs)
# qe_mask in double precision (falafel defaults to float32), as in
# aberration_dr6.py.
px_mod = qe.pixelization(shape=shape, wcs=wcs, dtype=np.float64)
pixarea = enmap.pixsizemap(shape, wcs, broadcastable=True)


def load_dr6_mask(path):
    """The DR6 mask on this CAR grid.  Read at each CAR pixel centre by
    bilinear interpolation, which stays inside [0, 1] where a harmonic
    reprojection rings.  The result is cached in cache_boost/masks/, written
    under a temporary name and renamed, so parallel shards read it once."""
    if not os.path.exists(path):
        raise SystemExit(f"DR6 mask not found at {path}.  Download it from\n"
                         f"{MASK_URL}\nor give its location with --mask-file.")
    name = (f"{os.path.splitext(os.path.basename(path))[0]}_"
            f"{os.path.getsize(path)}_car{RES / utils.arcmin:g}.fits")
    car = os.path.join("cache_boost", "masks", name)
    if os.path.exists(car):
        if not IN_WORKER:
            print(f"mask: {car}", flush=True)
        return enmap.enmap(np.asarray(enmap.read_map(car), np.float64), wcs)
    print(f"mask: {path} -> CAR ...", flush=True)
    hmask = np.clip(np.nan_to_num(hp.read_map(path, dtype=np.float32)), 0, 1)
    m = reproject.healpix2map(hmask, shape, wcs, spin=[0], method="spline",
                              order=1)
    del hmask
    m = enmap.enmap(np.asarray(m, np.float64).reshape(shape), wcs)
    os.makedirs(os.path.dirname(car), exist_ok=True)
    tmp = f"{car}.{os.getpid()}.fits"
    enmap.write_map(tmp, m)
    os.replace(tmp, car)
    return m


if args.mask == "dr6":
    MASK = load_dr6_mask(args.mask_file)
    W1, W2 = [float((MASK ** n * pixarea).sum() / (4 * np.pi)) for n in (1, 2)]
else:
    MASK = None                           # full sky: nothing to multiply
    W1 = W2 = 1.0

# ---------------------------------------------------------------- sims
PACK_L = np.array([l for l in range(LOUT + 1) for m in range(l + 1)])
PACK_M = np.array([m for l in range(LOUT + 1) for m in range(l + 1)])
PACK_IDX = hp.Alm.getidx(MLMAX, PACK_L, PACK_M)


def cmb(seed):
    return curvedsky.rand_map((3,) + shape, wcs, PS_CMB, lmax=MLMAX, seed=seed)


def noise(seed):
    return curvedsky.rand_map((3,) + shape, wcs, PS_NOISE, lmax=MLMAX,
                              seed=seed)


def boost(tqu, direction, beta, aberrate=True, modulate=True):
    """pixell's Doppler boost.  The maps are dT/T_cmb, hence map_unit=T_cmb.
    With aberrate=False, boost_map modulates its input in place."""
    return aberration.boost_map(tqu, dir=np.asarray(direction, float),
                                beta=beta, freq=FREQ, map_unit=utils.T_cmb,
                                aberrate=aberrate, modulate=modulate)


def reconstruct(tqu):
    """TQU map -> unnormalised TT, TE, EE lensing and TT modulation alm for
    L <= LOUT, shape (4, npack).  The mask, if any, multiplies the map here,
    so both estimators see the same masked, filtered T."""
    if MASK is not None:
        tqu = MASK * tqu
    alm = curvedsky.map2alm(tqu, lmax=MLMAX)
    f = [qe.filter_alms(a, fl, lmin=LMIN, lmax=LMAX)
         for a, fl in zip(alm, FILTERS)]
    rec = qe.qe_all(px, ucls, MLMAX, fTalm=f[0], fEalm=f[1], fBalm=f[2],
                    estimators=EST)
    mod = qe.qe_mask(px_mod, ucls, MLMAX, f[0])
    return np.array([rec[e][0][PACK_IDX] for e in EST] + [mod[PACK_IDX]])


def cache_path(label, i):
    return os.path.join(SIMS, f"{label.replace(' ', '_')}_{i:04d}.npy")


def move_old_sims():
    """Sims cached before they had their own folder (the previous version of
    this script kept them beside plots/) are moved into sims/, so they are
    reused rather than made again.  Shards starting together may race for a
    file; whichever loses finds it already moved."""
    for name in os.listdir(CACHE):
        if name.endswith(".npy"):
            try:
                os.replace(os.path.join(CACHE, name), os.path.join(SIMS, name))
            except FileNotFoundError:
                pass


# Response legs: the unboosted reconstruction of each sim, and the same sim
# boosted by +BETA along x, y and z with aberration only and modulation only.
BASE = "unboosted"
EFFECTS = ("aberration", "modulation")
RESP_LABELS = [f"{effect} {axis}" for effect in EFFECTS for axis in "xyz"]
STAGE = {"mf": "mf", "data": "data", BASE: "response",
         **{l: "response" for l in RESP_LABELS}}
TODO = {}           # sims this run still has to make, per label (set in main)


def one_sim(label, i):
    """Reconstruction of sim i of one stage; the seeds follow the index.
      mf          unboosted CMB + noise
      unboosted   unboosted CMB, no noise (response)
      <effect> <axis>  the same CMB boosted by +BETA along the axis, with
                  aberration only or modulation only (response)
      data        CMB with the full boost + noise"""
    if label == "mf":
        return reconstruct(cmb(1_000_000 + 2 * i) + noise(1_000_001 + 2 * i))
    if label == BASE:
        return reconstruct(cmb(2_000_000 + i))
    if label == "data":
        return reconstruct(boost(cmb(3_000_000 + 2 * i), BDIR, BETA)
                           + noise(3_000_001 + 2 * i))
    effect, axis = label.split()
    return reconstruct(boost(cmb(2_000_000 + i), AXES["xyz".index(axis)], BETA,
                             aberrate=effect == "aberration",
                             modulate=effect == "modulation"))


def make_and_cache(job):
    """Make one sim and cache it, in this process or in a worker.  The file
    is written under a temporary name and renamed, so no run ever reads one
    half-written."""
    label, i = job
    path = cache_path(label, i)
    tmp = f"{path}.{os.getpid()}.npy"
    np.save(tmp, one_sim(label, i))
    os.replace(tmp, path)
    return i


POOL = None         # the --nproc worker pool, made once per job in main()


def run(label, n):
    """Make and cache this shard's sims out of 0..n-1 (all of them with
    --nshards 1) that are not cached yet, --nproc at a time.  The time left
    assumes every remaining sim, of any stage, takes as long (wall clock) as
    the average so far in this call."""
    jobs = [(label, i) for i in range(args.shard, n, args.nshards)
            if not os.path.exists(cache_path(label, i))]
    made = (POOL.imap_unordered(make_and_cache, jobs) if POOL
            else map(make_and_cache, jobs))
    t0, done = time.time(), 0
    for i in made:
        _report_progress(label, n, i, t0, done := done + 1)


def _report_progress(label, n, i, t0, done):
    """One progress line per finished sim; s/sim is wall clock, so with
    several workers it is the time per sim of the whole job."""
    TODO[label] -= 1
    rate = (time.time() - t0) / done
    stage_left = sum(v for l, v in TODO.items() if STAGE[l] == STAGE[label])
    print(f"   {label} {i + 1}/{n}   {rate:.1f} s/sim "
          f"{rate * stage_left / 60:.1f} min left ({STAGE[label]}) "
          f"{rate * sum(TODO.values()) / 60:.1f} min left (total)",
          flush=True)


def collect(labels, n):
    """The sims 0..n-1 cached for every one of labels, by any shard, as an
    array (label, sim, ...).  A response sim counts once all its legs are in."""
    idx = [i for i in range(n)
           if all(os.path.exists(cache_path(l, i)) for l in labels)]
    return np.array([[np.load(cache_path(l, i)) for i in idx] for l in labels])


# ---------------------------------------------------------------- analysis
def l1_vector(a):
    """Cartesian vector of the L = 1 part of packed alm (along the last axis)."""
    return np.stack([-np.sqrt(3 / (2 * np.pi)) * a[..., 2].real,
                     np.sqrt(3 / (2 * np.pi)) * a[..., 2].imag,
                     np.sqrt(3 / (4 * np.pi)) * a[..., 1].real], axis=-1)


def cl_of(power):
    """Per-mode power (packed) -> C_L, counting m < 0."""
    w = np.where(PACK_M == 0, 1.0, 2.0) * power
    return np.array([w[PACK_L == l].sum() / (2 * l + 1)
                     for l in range(LOUT + 1)])


def debiased_cl(x):
    """C_L of the mean of the rows of x, minus its Monte Carlo noise."""
    var = (x.real.var(axis=0, ddof=1) + x.imag.var(axis=0, ddof=1)) / len(x)
    return cl_of(np.abs(x.mean(axis=0)) ** 2 - var)


def est_alm(rec, name):
    """Raw reconstructions (..., 4, npack) -> one estimator's packed alm, in
    units where its L = 1 vector estimates u = beta d on the full sky."""
    if name == "MOD":
        return NORM["MOD"][PACK_L] * rec[..., 3, :] / B_NU
    k = [EST.index(e) for e in CASES[name]]
    return -NORM[name][PACK_L] * rec[..., k, :].sum(axis=-2)      # phi = -u.n


def system(names, mf, diff, dat):
    """L = 1 vectors of two estimators stacked to 6, and their 6x6 response
    K (rows: components, columns: aberration xyz then modulation xyz).
    diff is (boosted - unboosted), shape (effect, axis, sim, 4, npack)."""
    Y = [[est_alm(x, n) for x in (mf, diff, dat)] for n in names]
    y_mf, y_dat = (np.concatenate([l1_vector(y[k]) for y in Y], -1)
                   for k in (0, 2))
    V = np.concatenate([l1_vector(y[1]) for y in Y], -1) / BETA
    K = V.mean(axis=2).reshape(6, 6).T
    Kerr = (V.std(axis=2, ddof=1) / np.sqrt(V.shape[2])).reshape(6, 6).T
    return Y, y_mf, y_dat, K, Kerr


def galactic(v):
    """Equatorial unit vectors (..., 3) -> galactic (l, b) in degrees."""
    g = v @ GAL.T
    return (np.degrees(np.arctan2(g[..., 1], g[..., 0])) % 360,
            np.degrees(np.arcsin(np.clip(g[..., 2], -1, 1))))


STATS = ["A", "v_x", "v_y", "v_z", "l", "b"]        # per-sim statistics


def stats(u):
    """Per-sim u (n, 3) -> amplitude A, velocity in km/s, and the galactic
    l, b of its direction in degrees (l unwrapped around the input)."""
    l_in = galactic(D_TRUE)[0]
    l, b = galactic(u / np.linalg.norm(u, axis=1)[:, None])
    return np.column_stack([u @ U_TRUE / (U_TRUE @ U_TRUE), u * C_KMS,
                            l_in + (l - l_in + 180) % 360 - 180, b])


def report(title, e):
    """Print one estimate and return what the summary keeps of it."""
    mean, sd, err = e["mean"], e["sd"], e["err"]
    dirs = e["u"] / np.linalg.norm(e["u"], axis=1)[:, None]
    ubar = e["u"].mean(axis=0) / np.linalg.norm(e["u"].mean(axis=0))
    l_in, b_in = galactic(D_TRUE)
    v_in = C_KMS * U_TRUE
    print(f"\n{title}")
    print(f"   A = {mean[0]:+.4f} +- {sd[0]:.4f} (sd), +- {err[0]:.4f} "
          f"(error on the mean, sd/sqrt(n_data))   expect 1")
    print(f"   null A = {e['null'].mean():+.4f} +- {e['null_err']:.4f}"
          f"   expect 0")
    print("   v [km/s]  " + "   ".join(
        f"{'xyz'[k]} {mean[k + 1]:+7.1f} +- {err[k + 1]:5.1f} "
        f"(in {v_in[k]:+6.1f})" for k in range(3)))
    print(f"   galactic  l = {mean[4]:.2f} +- {err[4]:.2f} (sd {sd[4]:.2f}), "
          f"b = {mean[5]:+.2f} +- {err[5]:.2f} (sd {sd[5]:.2f});  "
          f"input l = {l_in:.2f}, b = {b_in:+.2f}")
    print(f"   mean velocity "
          f"{np.degrees(np.arccos(np.clip(ubar @ D_TRUE, -1, 1))):.1f} deg "
          f"from the input, single sims median "
          f"{np.median(np.degrees(np.arccos(np.clip(dirs @ D_TRUE, -1, 1)))):.1f} deg")
    return dict(amp=e["stats"][:, 0], vel=e["stats"][:, 1:4], dir=dirs,
                null=e["null"], null_err=e["null_err"], stat_mean=mean,
                stat_sd=sd, stat_err=err)


def higher_l(Y, own):
    """(L, M) response of one estimator to its own effect, and the coherent
    power the true (full) boost leaves at each L, with a jackknife error."""
    S = Y[1][own] / BETA                               # (axis, sim, npack)
    P = np.einsum("jia,j->ia", (Y[1][0] + Y[1][1]) / BETA, U_TRUE)
    n_r = len(P)
    jk = np.array([debiased_cl(np.delete(P, k, axis=0)) for k in range(n_r)])
    M, D = Y[0], Y[2]
    var_m = M.real.var(axis=0, ddof=1) + M.imag.var(axis=0, ddof=1)
    var_d = D.real.var(axis=0, ddof=1) + D.imag.var(axis=0, ddof=1)
    return dict(Ralm=S.mean(axis=1).T, Ralm_sd=np.abs(S.std(axis=1, ddof=1)).T,
                leak=debiased_cl(P), leakerr=np.sqrt((n_r - 1) * jk.var(axis=0)),
                datleak=cl_of(np.abs(D.mean(axis=0) - M.mean(axis=0)) ** 2
                              - var_d / len(D) - var_m / len(M)),
                noise=cl_of(var_m))


def joint_weights(K, y_mf):
    """Per-axis weight on the aberration estimate in the joint one,
    w = s_mod^2 / (s_ab^2 + s_mod^2), where s is the sd over the mean-field
    sims of that component of the separate estimates K^-1 y.  The modulation
    estimate gets 1 - w."""
    s = (y_mf @ np.linalg.inv(K).T).std(axis=0, ddof=1)    # u_ab xyz, u_mod xyz
    return s[3:] ** 2 / (s[:3] ** 2 + s[3:] ** 2)


def linear_maps(K, y_mf):
    """3x6 maps from the estimator vectors (mean field removed) to u: rows of
    K^-1 for the separate estimates, and for the joint one the weighted
    average of the two separate estimates, [diag(w) diag(1-w)] K^-1."""
    Kinv = np.linalg.inv(K)
    w = joint_weights(K, y_mf)
    G = np.hstack([np.diag(w), np.diag(1 - w)]) @ Kinv
    return {"aberration": Kinv[:3], "modulation": Kinv[3:], "joint": G}


def solve(names, mf, diff, dat):
    """The three estimates of u from one pair of estimators, with the mean,
    sd and error on the mean of their per-sim statistics.  The error on the
    mean is the scatter of the data sims only, sd / sqrt(n_data).  The mean
    field and K, which every data sim shares, add errors of their own that
    this leaves out: for the mean field about sqrt(n_data / n_mf) times this
    one, so it needs n_mf >> n_data to be negligible."""
    Y, y_mf, y_dat, K, Kerr = system(names, mf, diff, dat)
    n_m = len(y_mf)
    m, C = y_mf.mean(axis=0), np.cov(y_mf, rowvar=False)
    L = linear_maps(K, y_mf)
    h = n_m // 2
    null_y = y_mf[h:] - y_mf[:h].mean(axis=0)     # split-half null test
    est = {}
    for e in L:
        u = (y_dat - m) @ L[e].T
        st = stats(u)
        sd = st.std(axis=0, ddof=1)
        null = null_y @ L[e].T @ U_TRUE / (U_TRUE @ U_TRUE)
        est[e] = dict(u=u, stats=st, mean=st.mean(axis=0), sd=sd,
                      err=sd / np.sqrt(len(st)), null=null,
                      null_err=null.std(ddof=1) * np.sqrt(1 / len(null) + 1 / h))
    return Y, K, Kerr, m, C, est


def analyse(mf, diff, dat):
    Y, K, Kerr, m, C, est = solve((ABER_CASE, "MOD"), mf, diff, dat)
    # Leave-one-response-sim-out copies of K, for the figures' errors on
    # R^-1, K^-1, G and the eigenvalues of R; no estimate or error uses them.
    K_jk = [system((ABER_CASE, "MOD"), mf, np.delete(diff, k, axis=2), dat)[3]
            for k in range(diff.shape[2])]
    y_mf = system((ABER_CASE, "MOD"), mf, diff, dat)[1]
    Kj = K[:, :3] + K[:, 3:]
    mf_ratio = np.array([np.linalg.norm(m[i:i + 3])
                         / np.linalg.norm(Kj[i:i + 3] @ U_TRUE) for i in (0, 3)])

    print(f"\nresponse K per unit u: rows {ABER_CASE} QE xyz, MOD QE xyz; "
          f"columns aberration xyz, modulation xyz")
    for i in range(6):
        print("   " + " ".join(f"{K[i, j]:+.3f}({Kerr[i, j] * 1e3:3.0f})"
                               for j in range(6)))
    print(f"   (MC error on the mean in units of 1e-3; b = {B_NU:.4f})")
    print(f"   |mean field| / |K u_in|:  {ABER_CASE} {mf_ratio[0]:.1f},  "
          f"MOD {mf_ratio[1]:.1f}")
    w_joint = joint_weights(K, y_mf)
    print("   joint weight on the aberration estimate x, y, z: "
          + ", ".join(f"{w:.3f}" for w in w_joint))

    titles = {"aberration": "aberration (separate)",
              "modulation": "modulation (separate)",
              "joint": "joint (one u for both effects)"}
    out = {name: report(title, est[name]) for name, title in titles.items()}
    for i, name in enumerate(("aberration", "modulation")):
        blk = np.s_[3 * i:3 * i + 3, 3 * i:3 * i + 3]
        eig = [np.linalg.eigvalsh(0.5 * (Kk[blk] + Kk[blk].T)) for Kk in K_jk]
        out[name].update(higher_l(Y[i], i), R=K[blk], Rerr=Kerr[blk],
                         eig_err=np.sqrt((len(eig) - 1) * np.var(eig, axis=0)),
                         R_jk=np.array([Kk[blk] for Kk in K_jk]))
    sd = np.sqrt(np.diag(C))
    # For the figures only: the copies of K, and the joint linear map G with
    # its copies (weights recomputed for each), so the plots can show errors
    # on R^-1, K^-1 and G.  C is kept for the correlation figure.
    out["system"] = dict(K=K, Kerr=Kerr, corr=C / np.outer(sd, sd),
                         mf_ratio=mf_ratio, K_jk=np.array(K_jk),
                         w_joint=w_joint, G=linear_maps(K, y_mf)["joint"],
                         G_jk=np.array([linear_maps(Kk, y_mf)["joint"]
                                        for Kk in K_jk]))

    print("\nseparate aberration amplitude for each lensing combination")
    rows = []
    for case in CASES:
        e = solve((case, "MOD"), mf, diff, dat)[5]["aberration"]
        rows.append([e["mean"][0], e["err"][0], e["sd"][0]])
        print(f"   {case:4s}  A = {e['mean'][0]:+.4f} +- {e['sd'][0]:.4f} (sd), "
              f"+- {e['err'][0]:.4f} (error on the mean)")
    out["lensing_cases"] = dict(names=np.array(list(CASES)), A=np.array(rows))
    return out


def main():
    print(f"mask {args.mask}"
          + (f" ({args.mask_file})" if MASK is not None else " (full sky)")
          + f", noise {args.noise} ({NOISE_T:g} uK-arcmin in T), "
          f"cache tag {CACHE_TAG}")
    print(f"lmin {LMIN}, lmax {LMAX}, mlmax {MLMAX}, {RES / utils.arcmin:g}' "
          f"CAR {shape[0]}x{shape[1]}, fsky {W1:.4f}, w2 {W2:.4f}")
    print(f"beta {BETA:.4e} towards ra {np.degrees(BDIR[0]):.2f}, "
          f"dec {np.degrees(BDIR[1]):.2f}; modulation b = {B_NU:.4f} at "
          f"{FREQ / 1e9:g} GHz", flush=True)

    print(f"noise 1/f slopes T {ALPHA_T:g}, P {ALPHA_P:g}; cache {CACHE}/"
          + (f"; shard {args.shard} of {args.nshards}" if args.nshards > 1
             else ""))
    os.makedirs(SIMS, exist_ok=True)
    move_old_sims()
    counts = {"mf": args.n_meanfield, "data": args.n_data,
              BASE: args.n_response, **{l: args.n_response for l in RESP_LABELS}}
    TODO.update({l: sum(not os.path.exists(cache_path(l, i))
                        for i in range(args.shard, n, args.nshards))
                 for l, n in counts.items()})
    print(f"this run makes {sum(TODO.values())} sims; the rest are cached "
          f"or belong to other shards")
    cores = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
             else os.cpu_count())
    print(f"{args.nproc} worker process(es), OMP_NUM_THREADS="
          f"{os.environ.get('OMP_NUM_THREADS', 'unset')} each; "
          f"{cores} cores available to this job", flush=True)
    global POOL
    if args.nproc > 1 and sum(TODO.values()):
        POOL = multiprocessing.get_context("spawn").Pool(args.nproc)
    try:
        print("\n[A] mean field")
        run("mf", args.n_meanfield)
        print("\n[B] response")
        for label in [BASE] + RESP_LABELS:
            run(label, args.n_response)
        print("\n[C] data")
        run("data", args.n_data)
    finally:
        if POOL:
            POOL.terminate()
            POOL = None

    # Report on everything cached so far, whichever shard made it.
    mf = collect(["mf"], args.n_meanfield)[0]
    legs = collect(RESP_LABELS + [BASE], args.n_response)   # (7 legs, sim, ...)
    diff = (legs[:-1] - legs[-1]).reshape((2, 3) + legs.shape[1:])
    dat = collect(["data"], args.n_data)[0]
    counts = (len(mf), diff.shape[2], len(dat))
    print(f"\ncollected {counts[0]}/{args.n_meanfield} mean-field, "
          f"{counts[1]}/{args.n_response} response and "
          f"{counts[2]}/{args.n_data} data sims")
    if counts[0] < 8 or counts[1] < 3 or counts[2] < 2:
        print("too few for a report yet: run the remaining shards")
        return
    if counts != (args.n_meanfield, args.n_response, args.n_data):
        print("partial: the report and figures cover the sims cached so far")
    results = analyse(mf, diff, dat)

    out = dict(v_true=C_KMS * U_TRUE, d_true=D_TRUE, beta=BETA, b_nu=B_NU,
               freq=FREQ, res_arcmin=RES / utils.arcmin,
               noise_name=args.noise, noise_t=NOISE_T, beam=BEAM,
               mask_kind=args.mask,
               mask_file=args.mask_file if MASK is not None else "",
               cache_tag=CACHE_TAG, stat_names=STATS,
               requested=[args.n_meanfield, args.n_response, args.n_data],
               n_mf=len(mf), n_resp=diff.shape[2], n_data=len(dat),
               lmin=LMIN, lmax=LMAX, knee=[KNEE_T, KNEE_P],
               alpha=[ALPHA_T, ALPHA_P], tcmb=TCMB,
               cl_tt=cltt, cl_ee=clee, nl_tt=nltt, nl_ee=nlee,
               pack_l=PACK_L, pack_m=PACK_M, fsky=W1, w2=W2,
               aber_case=ABER_CASE)
    if MASK is not None:
        thumb = enmap.downgrade(MASK, max(1, round(0.5 * utils.degree / RES)))
        dec, ra = enmap.posaxes(thumb.shape, thumb.wcs)
        out.update(mask_thumb=np.asarray(thumb, np.float32),
                   mask_dec=np.degrees(dec), mask_ra=np.degrees(ra))
    for name, res in results.items():
        for key, val in res.items():
            out[f"{name}.{key}"] = val
    path = os.path.join(CACHE, "summary.npz")
    tmp = f"{path}.{os.getpid()}.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)
    print(f"\nwrote {path}")

    import boost_act_plots
    boost_act_plots.make_all(path)             # -> CACHE/plots


if __name__ == "__main__":
    main()