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

--cmb lensed lenses every sim first: an unlensed CMB and a lensing potential
phi, drawn from CAMB's C_L^phiphi (C_1 = 0: a lensing dipole would be a
uniform deflection, the same as aberration), are combined with
pixell.lensing.lens_map_curved (pixell 0.32.9 or later: earlier versions
get pixels deflected over a pole and the polarization rotation wrong), and
the result is then boosted and given noise as before.  The filters and
weights use the lensed spectra.  It has its own cache tag,
<mask>_<noise>_lensed.  It also adds a lensing response,
measured like the boost response: each of --n-lens-response noise-free CMB
sims is lensed by one real harmonic phi_{L'M'} at a time (L' = 1 to
--lens-lmax, default 5), with amplitude +eps and -eps, and reconstructed;
[rec(+eps) - rec(-eps)] / 2 eps is how that lensing mode moves each
reconstructed (L, M).  The difference is
central because the estimator's own quadratic term, Q(dT, dT), is second
order in eps and only a central difference cancels it; one-sided it would
bias the response by ~l eps, ~15% per sim at l ~ 3000.  Without a mask it is
diagonal; with one, lensing at L' >= 2 leaks into the dipole.  Weighted by
C_L^phiphi it gives the scatter lensing adds to each velocity estimate
(lensing_dipole.png, and lensing_response.png per estimator).

--boost aberration or --boost modulation applies only that effect to the
data sims (the default, full, applies both), to check that each estimator
recovers its own effect and that hardening removes the other: the expected
value of each estimate is then printed and drawn in place of the input.
Those data sims are cached as data_aberration_XXXX.npy or
data_modulation_XXXX.npy, with the same seeds as the full-boost data sims,
and the summary and plots go to cache_boost/<mask>_<noise>/boost_<effect>/.
The mean-field and response sims do not depend on the boost, so all three
options share them.

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
    *_all      the same three from every reconstructed mode L <= LOUT of both
               estimators, not only the dipoles: generalised least squares
               with their covariance over the mean-field sims, which holds
               the lensing at every L' (in a lensed run), the mean field's
               scatter and the mask's coupling of the noise.  With the
               dipoles alone it is the estimate above.  Needs n_mf well
               above the number of modes (70).  Report only: on a partial
               sky the boost and low-L lensing leave nearly the same
               imprint on these modes, so it gains little (tested).
    *_lens     (--cmb lensed) the dipole estimates less the leakage of each
               data sim's own lensing modes phi_L'M', L' <= --lens-lmax,
               predicted with the lensing response: u - sum V_L'M' phi_L'M'.
               phi is known in a simulation; for the sky it would have to
               come from outside these data (a full-sky lensing map), so
               this is the most such a map could give
               (amplitude_direction_lens.png).  Each data sim's modes are
               cached as lensphi_XXXX.npy (~0.5 s each to draw, with
               --nproc in parallel).

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
parser.add_argument("--noise", default="f150", choices=["f150", "act"],
                    help="white noise level in T: f150 is 24 uK-arcmin, act "
                         "is 14 uK-arcmin (P is sqrt2 x); the beam and the "
                         "1/f knees are the same")
parser.add_argument("--boost", default="full",
                    choices=["full", "aberration", "modulation"],
                    help="which effects the data sims get: both (full), or "
                         "only one, to check that each estimate recovers its "
                         "own effect.  Mean-field and response sims are "
                         "shared; the data sims and the summary and plots "
                         "are kept separately")
parser.add_argument("--cmb", default="unlensed", choices=["unlensed", "lensed"],
                    help="lensed: lens every sim with a phi drawn from "
                         "C_L^phiphi before the boost; cached under its own "
                         "tag, <mask>_<noise>_lensed")
parser.add_argument("--n-lens-response", type=int, default=10,
                    help="--cmb lensed only: CMB sims for the lensing "
                         "response, each reconstructed 2((Lmax + 1)^2 - 1) "
                         "times (70 for --lens-lmax 5): lensed by +eps and "
                         "-eps of each mode phi_LM, 1 <= L <= Lmax")
parser.add_argument("--lens-lmax", type=int, default=5,
                    help="--cmb lensed only: highest multipole L' of the "
                         "lensing modes in the lensing response.  Raising it "
                         "shows whether higher L' still add; modes already "
                         "cached are reused")
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
import pixell                                                  # noqa: E402
import pytempura                                               # noqa: E402
from falafel import qe                                         # noqa: E402
from packaging.version import Version                          # noqa: E402
from pixell import (enmap, curvedsky, aberration, reproject, lensing,  # noqa: E402
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
warnings.filterwarnings("ignore", message="In the future, the default will be")

# ---------------------------------------------------------------- settings
LMIN, LMAX, MLMAX = 600, 3000, 3500
LOUT = 5                                  # highest reconstruction L kept
RES = 3.0 * utils.arcmin                  # fejer1 grid: exact SHTs to l = 3599
NOISE_LEVELS = {"f150": 24.0, "act": 14.0}    # uK-arcmin in T (P is sqrt2 x)
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
LENSED = args.cmb == "lensed"
CACHE_TAG = (f"{args.mask}_{args.noise}"  # new value whenever a setting changes
             + ("_lensed" if LENSED else ""))
CACHE = os.path.join("cache_boost", CACHE_TAG)     # summary.npz and plots/
SIMS = os.path.join(CACHE, "sims")                 # one .npy per sim
# Effects applied to the data sims, the data sims' label, and where this
# run's summary and plots go.  --boost full keeps the original names.
ABERRATE = args.boost in ("full", "aberration")
MODULATE = args.boost in ("full", "modulation")
DATA = "data" if args.boost == "full" else f"data {args.boost}"
OUT = CACHE if args.boost == "full" else os.path.join(CACHE, f"boost_{args.boost}")

X_NU = utils.h * FREQ / (utils.k * utils.T_cmb)
B_NU = X_NU / np.tanh(X_NU / 2) - 1       # linearised-T modulation factor
D_TRUE = np.array([np.cos(BDIR[1]) * np.cos(BDIR[0]),
                   np.cos(BDIR[1]) * np.sin(BDIR[0]),
                   np.sin(BDIR[1])])
U_TRUE = BETA * D_TRUE
GAL = hp.Rotator(coord=["C", "G"]).mat    # equatorial -> galactic vectors

# pixell's interpol_map (0.32.7 to 0.32.9) doubles a full-sky map for the
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

# pixell up to 0.32.7 lenses each pixel along a geodesic with an arcsin for
# the change in longitude.  For a pixel deflected over a pole that gives the
# wrong position, or NaN, which the SHT rejects ("phi out of range: -nan",
# mean-field sim 32 at full resolution); and its polarization rotation is
# wrong, by ~1% of Q, U at |dec| < 63 deg for a 34' deflection.  0.32.9
# (master) does both with ducc0 and lensing by a dipole then matches
# aberration in T, Q and U alike.
if LENSED and Version(pixell.__version__) < Version("0.32.9"):
    raise SystemExit(f"--cmb lensed needs pixell 0.32.9 or later (found "
                     f"{pixell.__version__}):  pip install --upgrade pixell")

# ---------------------------------------------------------------- spectra
IN_WORKER = __name__ == "__mp_main__"       # a spawned --nproc worker
if not IN_WORKER:
    print("CAMB ...", flush=True)
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965,
                       As=2.1e-9, tau=0.06)
if LENSED:
    pars.set_for_lmax(MLMAX + 500, lens_potential_accuracy=1)
else:                                     # as before, so old caches stay valid
    pars.set_for_lmax(MLMAX + 500)
camb_res = camb.get_results(pars)
cls = camb_res.get_cmb_power_spectra(
    pars, raw_cl=True, spectra=["unlensed_scalar", "lensed_scalar"])
# The sims are drawn from the unlensed spectra (and lensed, with --cmb
# lensed); the filters and weights use what the sky looks like, the lensed
# spectra in that case.
un_tt, un_ee, un_bb, un_te = cls["unlensed_scalar"][:MLMAX + 1].T
cltt, clee, clbb, clte = cls["lensed_scalar" if LENSED else
                             "unlensed_scalar"][:MLMAX + 1].T   # (dT/T)^2
CLPP = np.zeros(MLMAX + 1)                # lensing potential C_L^phiphi
if LENSED:
    _pp = camb_res.get_lens_potential_cls(lmax=MLMAX)[:, 0]
    _L = np.arange(2, MLMAX + 1.0)
    CLPP[2:] = 2 * np.pi * _pp[2:] / (_L * (_L + 1)) ** 2   # C_0 = C_1 = 0

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

PS_CMB = np.zeros((3, 3, MLMAX + 1))        # unlensed: what the sims draw
PS_CMB[0, 0], PS_CMB[1, 1], PS_CMB[2, 2] = un_tt, un_ee, un_bb
PS_CMB[0, 1] = PS_CMB[1, 0] = un_te
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
# Coarse grid for maps of the reconstructions at L <= LOUT (figures only).
MAP_SHAPE, MAP_WCS = enmap.fullsky_geometry(res=2 * utils.degree)


def lowl_map(packed):
    """Packed alm (L <= LOUT), in units of u, -> map on a 2-degree grid in
    km/s: at L = 1 its value at n is c u.n."""
    a = np.zeros(hp.Alm.getsize(LOUT), complex)
    a[hp.Alm.getidx(LOUT, PACK_L, PACK_M)] = packed
    return np.asarray(curvedsky.alm2map(a, enmap.zeros(MAP_SHAPE, MAP_WCS))
                      ) * C_KMS


PHI_SEED = 1_000_000_000      # phi of sim seed s is drawn with PHI_SEED + s


def lens(alm, phi_alm):
    """TEB alm lensed by phi: T(n + grad phi), on geodesics, as a TQU map."""
    return lensing.lens_map_curved((3,) + shape, wcs, phi_alm, alm,
                                   output="l")[0]


def unlensed_alm(seed):
    """The TEB alm the unlensed run's rand_map draws for this seed, so that
    a lensed sim is the unlensed run's sim of the same seed, lensed."""
    return curvedsky.rand_alm_healpy(PS_CMB, lmax=MLMAX, seed=seed)


def cmb(seed):
    """TQU CMB of one seed; with --cmb lensed, lensed by its own phi, drawn
    from the same seed so that every leg of a response sim shares it."""
    if not LENSED:
        return curvedsky.rand_map((3,) + shape, wcs, PS_CMB, lmax=MLMAX,
                                  seed=seed)
    alm = unlensed_alm(seed)
    phi = curvedsky.rand_alm(CLPP, lmax=MLMAX, seed=PHI_SEED + seed)
    return lens(alm, phi)


# Lensing response: one real harmonic of the lensing potential at a time,
# phi = eps_L Y_LM, L = 1..LPHI, M = -L..L, in the real basis of the plots
# (r_L0 = a_L0, r_LM = sqrt2 (-1)^M Re a_LM, r_L,-M = -sqrt2 (-1)^M Im a_LM).
# eps_L makes the rms deflection a tenth of the aberration's (beta sqrt(2/3)),
# small enough to stay linear.
LPHI = args.lens_lmax
PHI_MODES = [(L, M) for L in range(1, LPHI + 1) for M in range(-L, L + 1)]
PHI_EPS = {L: 0.1 * BETA * np.sqrt(2 / 3) / np.sqrt(L * (L + 1) / (4 * np.pi))
           for L in range(1, LPHI + 1)}


def real_mode_alm(L, M, lmax):
    """alm (healpy order, lmax) of the unit real harmonic Y_LM."""
    a = np.zeros(hp.Alm.getsize(lmax), complex)
    k = hp.Alm.getidx(lmax, L, abs(M))
    a[k] = (1.0 if M == 0 else (-1) ** M / np.sqrt(2) if M > 0
            else -1j * (-1) ** abs(M) / np.sqrt(2))
    return a


def lensed_by_mode(seed, mode, sign):
    """Noise-free unlensed CMB of seed lensed by sign * eps_L Y_LM, for
    mode (L, M) and sign +1 or -1."""
    alm = unlensed_alm(seed)
    L, M = mode
    return lens(alm, sign * PHI_EPS[L] * real_mode_alm(L, M, L))


# The lensing modes L' <= LENSPHI_L of each data sim's own phi, in the real
# basis of PHI_MODES (ordered as it is, so its first len(PHI_MODES) entries
# are those modes), cached as lensphi_XXXX.npy: what amplitude_direction_lens
# subtracts the predicted leakage of.  The same draw as cmb() makes.
LENSPHI_L = max(10, LPHI)


def lens_modes(seed):
    """Real-basis coefficients of the phi that cmb(seed) lenses by, for
    1 <= L' <= LENSPHI_L."""
    a = curvedsky.rand_alm(CLPP, lmax=MLMAX, seed=PHI_SEED + seed)
    r = []
    for L in range(1, LENSPHI_L + 1):
        for M in range(-L, L + 1):
            x = a[hp.Alm.getidx(MLMAX, L, abs(M))]
            r.append(x.real if M == 0 else np.sqrt(2) * (-1) ** abs(M)
                     * (x.real if M > 0 else -x.imag))
    return np.array(r)


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
# Lensing-response legs (--cmb lensed): each mode at +eps and -eps.
# Files phi_2_-1_p_XXXX.npy, phi_2_-1_m_XXXX.npy, ...
PHI_LABELS = [f"phi {L} {M:+d} {s}" for L, M in PHI_MODES for s in "pm"]
STAGE = {"mf": "mf", DATA: "data", "lensphi": "data", BASE: "response",
         **{l: "response" for l in RESP_LABELS},
         **{l: "lensing response" for l in PHI_LABELS}}
TODO = {}           # sims this run still has to make, per label (set in main)


def one_sim(label, i):
    """Reconstruction of sim i of one stage; the seeds follow the index.
      mf          unboosted CMB + noise
      unboosted   unboosted CMB, no noise (response)
      <effect> <axis>  the same CMB boosted by +BETA along the axis, with
                  aberration only or modulation only (response)
      data        CMB with the full boost + noise
      data <effect>  the same CMB with that effect only + the same noise
      phi L M p/m  noise-free CMB lensed by +eps or -eps of one lensing
                  mode (lensing response, --cmb lensed)
      lensphi     not a reconstruction: the low-L lensing modes of data
                  sim i (--cmb lensed)"""
    if label == "lensphi":
        return lens_modes(3_000_000 + 2 * i)
    if label == "mf":
        return reconstruct(cmb(1_000_000 + 2 * i) + noise(1_000_001 + 2 * i))
    if label == BASE:
        return reconstruct(cmb(2_000_000 + i))
    if label.startswith("phi"):
        _, L, M, sign = label.split()
        return reconstruct(lensed_by_mode(2_000_000 + i, (int(L), int(M)),
                                          1 if sign == "p" else -1))
    if label.startswith("data"):
        return reconstruct(boost(cmb(3_000_000 + 2 * i), BDIR, BETA,
                                 aberrate=ABERRATE, modulate=MODULATE)
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
    v_exp = C_KMS * e["u_expect"]
    A_exp = e["u_expect"] @ U_TRUE / (U_TRUE @ U_TRUE)
    dirs = e["u"] / np.linalg.norm(e["u"], axis=1)[:, None]
    ubar = e["u"].mean(axis=0) / np.linalg.norm(e["u"].mean(axis=0))
    l_in, b_in = galactic(D_TRUE)
    v_in = C_KMS * U_TRUE
    print(f"\n{title}")
    print(f"   A = {mean[0]:+.4f} +- {sd[0]:.4f} (sd), +- {err[0]:.4f} "
          f"(error on the mean, sd/sqrt(n_data))   expect {A_exp:.3f}")
    print(f"   null A = {e['null'].mean():+.4f} +- {e['null_err']:.4f}"
          f"   expect 0")
    print("   v [km/s]  " + "   ".join(
        f"{'xyz'[k]} {mean[k + 1]:+7.1f} +- {err[k + 1]:5.1f} "
        f"(expect {v_exp[k]:+6.1f})" for k in range(3)))
    if not np.allclose(v_exp, v_in):
        print("   (expected values are for the effect(s) applied; the "
              "input velocity is " + ", ".join(f"{x:+.1f}" for x in v_in)
              + " km/s)")
    print(f"   galactic  l = {mean[4]:.2f} +- {err[4]:.2f} (sd {sd[4]:.2f}), "
          f"b = {mean[5]:+.2f} +- {err[5]:.2f} (sd {sd[5]:.2f});  "
          f"input l = {l_in:.2f}, b = {b_in:+.2f}")
    print(f"   mean velocity "
          f"{np.degrees(np.arccos(np.clip(ubar @ D_TRUE, -1, 1))):.1f} deg "
          f"from the input, single sims median "
          f"{np.median(np.degrees(np.arccos(np.clip(dirs @ D_TRUE, -1, 1)))):.1f} deg")
    return dict(amp=e["stats"][:, 0], vel=e["stats"][:, 1:4], dir=dirs,
                null=e["null"], null_err=e["null_err"], stat_mean=mean,
                stat_sd=sd, stat_err=err, v_expect=v_exp, A_expect=A_exp)


def higher_l(Y, own):
    """(L, M) response of one estimator to its own effect, and the coherent
    power the applied boost (--boost) leaves at each L, with a jackknife
    error."""
    S = Y[1][own] / BETA                               # (axis, sim, npack)
    P = np.einsum("jia,j->ia", (ABERRATE * Y[1][0] + MODULATE * Y[1][1])
                  / BETA, U_TRUE)
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


def estimates(maps, K, y_mf, y_dat):
    """Per-sim u from each linear map in maps (estimate -> rows applied to
    the estimator vector y less its mean over the mean-field sims), with the
    mean, sd and error on the mean of its statistics, and the split-half
    null test.  K is the response of y to (u_ab, u_mod)."""
    m = y_mf.mean(axis=0)
    # What each estimate should average to for the effects applied: L K x,
    # with x the true (u_ab, u_mod).  For the full boost this is u_in for
    # all three; with one effect it is u_in or 0 for the separate estimates
    # and that effect's share of the weights for the joint one.
    x_true = np.r_[ABERRATE * U_TRUE, MODULATE * U_TRUE]
    h = len(y_mf) // 2
    null_y = y_mf[h:] - y_mf[:h].mean(axis=0)     # split-half null test
    est = {}
    for e in maps:
        u_exp = maps[e] @ K @ x_true
        u_exp[np.abs(u_exp) < 1e-9 * np.linalg.norm(U_TRUE)] = 0.0  # rounding
        u = (y_dat - m) @ maps[e].T
        st = stats(u)
        sd = st.std(axis=0, ddof=1)
        null = null_y @ maps[e].T @ U_TRUE / (U_TRUE @ U_TRUE)
        est[e] = dict(u=u, stats=st, mean=st.mean(axis=0), sd=sd,
                      u_expect=u_exp,
                      err=sd / np.sqrt(len(st)), null=null,
                      null_err=null.std(ddof=1) * np.sqrt(1 / len(null) + 1 / h))
    return est


def solve(names, mf, diff, dat):
    """The three estimates of u from one pair of estimators, with the mean,
    sd and error on the mean of their per-sim statistics.  The error on the
    mean is the scatter of the data sims only, sd / sqrt(n_data).  The mean
    field and K, which every data sim shares, add errors of their own that
    this leaves out: for the mean field about sqrt(n_data / n_mf) times this
    one, so it needs n_mf >> n_data to be negligible."""
    Y, y_mf, y_dat, K, Kerr = system(names, mf, diff, dat)
    m, C = y_mf.mean(axis=0), np.cov(y_mf, rowvar=False)
    est = estimates(linear_maps(K, y_mf), K, y_mf, y_dat)
    return Y, K, Kerr, m, C, est


# ---------------------------------------- estimates from every mode L <= LOUT
# The estimates above use only the reconstructed dipoles (L = 1, six numbers),
# so whatever the mask, lensing and the noise put into the dipoles stays in
# u.  Lensing at L' >= 2 leaks into the masked dipoles, but it also shows up,
# much more strongly, at L = L' itself, and the mean field and the mask's
# coupling of the noise are shared across L too.  The modes 2 <= L <= LOUT of
# both estimators therefore carry information about what contaminates the
# dipoles.  The estimates below use all of them, (LOUT + 1)^2 - 1 per
# estimator (70 for LOUT = 5), weighted by their covariance C over the
# mean-field sims, which are lensed in a lensed run, so C contains the
# lensing at every L' as well as the noise:
#     u = (A^T C^-1 A)^-1 A^T C^-1 (x - x_MF)       (generalised least squares)
# with A the response of all the modes to (u_ab, u_mod), measured with the
# response sims as K is.  Of all linear estimates from these modes that
# recover u whatever its value, this has the smallest variance.  With the
# dipoles alone A is K and this is K^-1, the estimate above.  The joint
# estimate weights the two separate ones per axis as before.  The data sims
# are not used to make the weights, so their scatter measures the result
# honestly.
# In tests at reduced resolution on a stand-in mask, even with the low-L
# lensing raised 100x so that the modes L = 2-5 measure it at S/N ~2 per
# mode, this gained ~9% in variance at best: aberration is itself a lensing
# dipole, and on part of the sky it leaves nearly the same imprint on these
# modes as lensing at L' = 2, 3, ... does, so they cannot tell the two apart.


def all_modes(a, lmax):
    """Packed alm (..., npack) -> the real vector of its modes 1 <= L <= lmax:
    the L = 1 Cartesian vector (as K uses), then for each L >= 2 the real
    harmonics a_L0, sqrt2 Re a_LM and sqrt2 Im a_LM (M > 0)."""
    parts = [l1_vector(a)]
    for L in range(2, lmax + 1):
        for M in range(L + 1):
            k = np.flatnonzero((PACK_L == L) & (PACK_M == M))[0]
            parts.append((1.0 if M == 0 else np.sqrt(2)) * a[..., k:k + 1].real)
            if M:
                parts.append(np.sqrt(2) * a[..., k:k + 1].imag)
    return np.concatenate(parts, -1)


def system_all(names, mf, diff, dat, lmax):
    """All modes 1 <= L <= lmax of two estimators, stacked, for the
    mean-field and data sims, and their response A (rows: modes; columns:
    aberration xyz then modulation xyz), whose L = 1 rows are K."""
    Y = [[est_alm(x, n) for x in (mf, diff, dat)] for n in names]
    x_mf, x_dat = (np.concatenate([all_modes(y[k], lmax) for y in Y], -1)
                   for k in (0, 2))
    V = np.concatenate([all_modes(y[1], lmax) for y in Y], -1) / BETA
    return x_mf, x_dat, V.mean(axis=2).reshape(6, -1).T


def gls_maps(A, x_mf):
    """Generalised least squares from all the modes: P = (A^T C^-1 A)^-1
    A^T C^-1, rows u_ab then u_mod, with C the modes' covariance over the
    mean-field sims; and the joint map, the per-axis weighted average of
    the two with w = s_mod^2 / (s_ab^2 + s_mod^2) on aberration (s their sd
    over the mean-field sims), as for K^-1.  Returns the maps and w."""
    C = np.cov(x_mf, rowvar=False)
    CiA = np.linalg.solve(C, A)
    P = np.linalg.solve(A.T @ CiA, CiA.T)
    s2 = np.einsum("ij,jk,ik->i", P, C, P)
    w = s2[3:] / (s2[:3] + s2[3:])
    G = np.hstack([np.diag(w), np.diag(1 - w)]) @ P
    return {"aberration": P[:3], "modulation": P[3:], "joint": G}, w


def solve_all_modes(names, mf, diff, dat):
    """The three estimates from all modes L <= LOUT (as solve() gives from
    the dipoles), and the sd of their statistics when the modes are limited
    to L <= 1, 2, ..., LOUT (L <= 1 is solve()'s estimate)."""
    x_mf, x_dat, A = system_all(names, mf, diff, dat, LOUT)
    n_mode = [2 * ((l + 1) ** 2 - 1) for l in range(1, LOUT + 1)]
    by_l = []
    for l, n in zip(range(1, LOUT + 1), n_mode):
        keep = np.r_[0:n // 2, A.shape[0] // 2:A.shape[0] // 2 + n // 2]
        maps, w = gls_maps(A[keep], x_mf[:, keep])
        est = estimates(maps, A[keep], x_mf[:, keep], x_dat[:, keep])
        by_l.append([[est[e][k] for k in ("mean", "sd", "err")]
                     for e in ("aberration", "modulation", "joint")])
    return est, w, A, np.array(by_l), np.array(n_mode)


def lens_response(dphi, mf, diff, dat):
    """What lensing does to the estimates, from the lensing-response sims.
    dphi is (mode, sim, 4, npack): the change in the raw reconstructions per
    unit coefficient of each real lensing mode phi_{L'M'} (PHI_MODES).
      lens_R_ab, lens_R_mod  (mode, npack) mean response of each estimator's
                 (L, M) to each mode: phi-hat per phi for the lensing
                 estimator, f-hat per phi for the modulation one.  Without a
                 mask the first is the identity and the second zero.
      lens_V     (estimate, mode, 3) the hardened velocity (u units) a unit
                 phi_{L'M'} puts into the aberration, modulation and joint
                 estimates: K^-1 or G applied to the dipoles it produces.
      lens_sigma (estimate, L') per-component rms velocity [km/s] that
                 lensing at L' adds, sqrt(C_L' sum_M' |V|^2 / 3), with |V|^2
                 less its Monte Carlo noise (which would bias it up); errors
                 by jackknife over the lensing-response sims.
      lens_coupling (estimate, L') rms |V| over M' at L' (debiased the same
                 way), relative to the aberration estimate's response to a
                 lensing dipole (L' = 1): how strongly each L' leaks into
                 the dipole (0 for L' >= 2 without a mask).
      lens_check (3,) the L' = 1 modes against the aberration columns of K,
                 |difference| / |K column|: a lensing dipole phi = -u.n is
                 the same deflection as aberration by u."""
    names = (ABER_CASE, "MOD")
    _, y_mf, _, K, _ = system(names, mf, diff, dat)
    maps = linear_maps(K, y_mf)
    est = [est_alm(dphi, n) for n in names]           # (mode, sim, npack)
    y6 = np.concatenate([l1_vector(e) for e in est], -1)    # (mode, sim, 6)
    n = y6.shape[1]
    modes = np.array(PHI_MODES)
    cl = CLPP[modes[:, 0]]
    order = ("aberration", "modulation", "joint")

    def scatter(y):
        """y (mode, sim, 6) -> V, sigma, coupling, with |mean V|^2 less
        its Monte Carlo noise var(V) / n_sim."""
        Vs = np.stack([y @ maps[e].T for e in order])   # (est, mode, sim, 3)
        V = Vs.mean(axis=2)
        p = (V ** 2).sum(-1) - Vs.var(axis=2, ddof=1).sum(-1) / Vs.shape[2]
        by_L = [modes[:, 0] == L for L in range(1, LPHI + 1)]
        sig = C_KMS * np.array([[np.sqrt(max((cl * p[k])[b].sum() / 3, 0))
                                 for b in by_L] for k in range(3)])
        rms = np.array([[np.sqrt(max(p[k][b].mean(), 0)) for b in by_L]
                        for k in range(3)])
        return V, sig, rms / rms[0, 0]

    V, sig, coup = scatter(y6)
    jk = [scatter(np.delete(y6, j, axis=1)) for j in range(n)]
    sig_err = np.sqrt((n - 1) * np.var([j[1] for j in jk], axis=0))
    coup_err = np.sqrt((n - 1) * np.var([j[2] for j in jk], axis=0))
    check = []
    for k, (L, M) in enumerate(PHI_MODES):
        if L == 1:
            a = real_mode_alm(1, M, LOUT)[hp.Alm.getidx(LOUT, PACK_L, PACK_M)]
            pred = K[:, :3] @ -l1_vector(a)           # phi = a.n = -u.n
            check.append(np.linalg.norm(y6[k].mean(axis=0) - pred)
                         / np.linalg.norm(pred))
    R = {"ab": -est[0], "mod": B_NU * est[1]}         # phi-hat, f-hat per phi
    print(f"\nlensing response from {n} sims, modes L' = 1..{LPHI}")
    print("   L' = 1 against the aberration columns of K: "
          + ", ".join(f"{c:.3f}" for c in check))
    for k, e in enumerate(order):
        print(f"   {e:10s} velocity scatter from lensing [km/s], L' = 1..{LPHI}: "
              + "  ".join(f"{x:.1f}" for x in sig[k])
              + f"   total {np.sqrt((sig[k] ** 2).sum()):.1f}")
    return dict(phi_modes=modes, phi_eps=np.array([PHI_EPS[L] for L, M in
                                                    PHI_MODES]),
                n_lens_resp=n, lens_V=V, lens_sigma=sig,
                lens_sigma_err=sig_err, lens_coupling=coup,
                lens_coupling_err=coup_err, lens_check=np.array(check),
                **{f"lens_R_{k}": r.mean(axis=1) for k, r in R.items()},
                **{f"lens_R_{k}_sd": np.abs(r.std(axis=1, ddof=1))
                   for k, r in R.items()})


def lens_removed(mf, diff, dat, lens_V):
    """The three dipole estimates of each data sim less the leakage of its
    own lensing modes L' <= LPHI predicted by the lensing response,
    u - sum_L'M' V_L'M' phi_L'M' (amplitude_direction_lens.png).  Every
    term has zero mean, so the expected values stay as they were; so does
    the null test, on the mean-field sims.  The sims' phi is read from
    lensphi_XXXX.npy; missing ones (other shards) leave this out."""
    idx = [i for i in range(args.n_data) if os.path.exists(cache_path(DATA, i))]
    if not all(os.path.exists(cache_path("lensphi", i)) for i in idx):
        print("lensing removed: the lensing modes of some data sims are not "
              "cached yet; run the remaining shards")
        return {}
    r = np.array([np.load(cache_path("lensphi", i)) for i in idx])
    r = r[:, :len(PHI_MODES)]
    est = solve((ABER_CASE, "MOD"), mf, diff, dat)[5]
    out = {}
    print(f"\nwith the leakage of each data sim's lensing modes L' <= {LPHI} "
          "removed (phi known, as from a perfect external lensing map)")
    for k, name in enumerate(("aberration", "modulation", "joint")):
        e = dict(est[name])
        u = e["u"] - r @ lens_V[k]
        st = stats(u)
        sd = st.std(axis=0, ddof=1)
        e.update(u=u, stats=st, mean=st.mean(axis=0), sd=sd,
                 err=sd / np.sqrt(len(st)))
        out[f"{name}_lens"] = report(f"{name}, lensing L' <= {LPHI} removed", e)
        out[f"{name}_lens"]["phi_lmax"] = LPHI
    return out


def analyse(mf, diff, dat):
    Y, K, Kerr, m, C, est = solve((ABER_CASE, "MOD"), mf, diff, dat)
    # Leave-one-response-sim-out copies of K, for the figures' errors on
    # R^-1, K^-1, G and the eigenvalues of R; no estimate or error uses them.
    K_jk = [system((ABER_CASE, "MOD"), mf, np.delete(diff, k, axis=2), dat)[3]
            for k in range(diff.shape[2])]
    y_mf = system((ABER_CASE, "MOD"), mf, diff, dat)[1]
    # Size of each estimator's mean-field dipole relative to the input boost
    # (both in units of u, so this is |mean field| in km/s over 370 km/s).
    mf_over_u = np.array([np.linalg.norm(m[i:i + 3]) / np.linalg.norm(U_TRUE)
                          for i in (0, 3)])

    print(f"\nresponse K per unit u: rows {ABER_CASE} QE xyz, MOD QE xyz; "
          f"columns aberration xyz, modulation xyz")
    for i in range(6):
        print("   " + " ".join(f"{K[i, j]:+.3f}({Kerr[i, j] * 1e3:3.0f})"
                               for j in range(6)))
    print(f"   (MC error on the mean in units of 1e-3; b = {B_NU:.4f})")
    print(f"   |mean field| / |u_in|:  {ABER_CASE} {mf_over_u[0]:.1f},  "
          f"MOD {mf_over_u[1]:.1f}")
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
        # Maps of what this estimator reconstructs at L <= LOUT: the mean
        # over the unboosted (mean-field) sims, one of those sims alone, and
        # the signal of the applied boost, from the response sims.
        D = Y[i][1] / BETA                      # (effect, axis, sim, npack)
        sig = np.einsum("jsa,j->a", (ABERRATE * D[0] + MODULATE * D[1]),
                        U_TRUE) / D.shape[2]
        out[name].update(mf_map=lowl_map(Y[i][0].mean(axis=0)),
                         sim_map=lowl_map(Y[i][0][0]),
                         signal_map=lowl_map(sig),
                         dat_map=lowl_map(Y[i][2].mean(axis=0)),
                         dat_sim_map=lowl_map(Y[i][2][0]))
    sd = np.sqrt(np.diag(C))
    # For the figures only: the copies of K, and the joint linear map G with
    # its copies (weights recomputed for each), so the plots can show errors
    # on R^-1, K^-1 and G.  C is kept for the correlation figure.
    out["system"] = dict(K=K, Kerr=Kerr, corr=C / np.outer(sd, sd),
                         mf_over_u=mf_over_u, K_jk=np.array(K_jk),
                         w_joint=w_joint, G=linear_maps(K, y_mf)["joint"],
                         G_jk=np.array([linear_maps(Kk, y_mf)["joint"]
                                        for Kk in K_jk]))

    # The same three estimates from every reconstructed mode L <= LOUT of both
    # estimators (report.md).
    est_all, w_all, A_all, by_l, n_mode = solve_all_modes(
        (ABER_CASE, "MOD"), mf, diff, dat)
    n_m, p = len(mf), n_mode[-1]
    infl = (n_m - 1) / max(n_m - p + 5, 1) - 1      # Wishart: (n-1)/(n-p+q-1)
    print(f"\nestimates from all reconstructed modes L <= {LOUT} of both "
          f"estimators ({p} modes), weighted by their covariance over the "
          f"{n_m} mean-field sims")
    if infl > 0.05:
        print(f"   (with {n_m} mean-field sims for {p} modes the noise in the "
              f"covariance adds about {infl:.0%} to the variance; more "
              f"mean-field sims would lower it)")
    for name, title in titles.items():
        out[f"{name}_all"] = report(f"{title}, all modes L <= {LOUT}",
                                    est_all[name])
    print("\nsd of one sim, A, from the modes L <= 1 (the dipoles, as above) "
          f"to L <= {LOUT}:")
    for k, name in enumerate(titles):
        print(f"   {name:10s} " + "  ".join(
            f"L<={l + 1}: {by_l[l, k, 1, 0]:.4f}" for l in range(LOUT)))
    out["system"].update(all_A=A_all, all_w=w_all, all_by_l=by_l,
                         all_n_mode=n_mode, all_var_infl=infl)

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
    print(f"boost {args.boost}: data sims "
          + {"full": "aberrated and modulated", "aberration": "aberrated only",
             "modulation": "modulated only"}[args.boost]
          + f"; summary and plots in {OUT}/")
    print(f"lmin {LMIN}, lmax {LMAX}, mlmax {MLMAX}, {RES / utils.arcmin:g}' "
          f"CAR {shape[0]}x{shape[1]}, fsky {W1:.4f}, w2 {W2:.4f}")
    print(f"beta {BETA:.4e} towards ra {np.degrees(BDIR[0]):.2f}, "
          f"dec {np.degrees(BDIR[1]):.2f}; modulation b = {B_NU:.4f} at "
          f"{FREQ / 1e9:g} GHz", flush=True)

    print(f"CMB {args.cmb}" + (f": phi from C_L^phiphi, lensed with pixell; "
          f"{args.n_lens_response} lensing-response sims" if LENSED else ""))
    print(f"noise 1/f slopes T {ALPHA_T:g}, P {ALPHA_P:g}; cache {CACHE}/"
          + (f"; shard {args.shard} of {args.nshards}" if args.nshards > 1
             else ""))
    os.makedirs(SIMS, exist_ok=True)
    move_old_sims()
    n_lens = args.n_lens_response if LENSED else 0
    counts = {"mf": args.n_meanfield, DATA: args.n_data,
              BASE: args.n_response, **{l: args.n_response for l in RESP_LABELS},
              **{l: n_lens for l in PHI_LABELS},
              **({"lensphi": args.n_data} if LENSED else {})}
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
        run(DATA, args.n_data)
        if LENSED:              # their low-L lensing modes, ~1 s each
            run("lensphi", args.n_data)
        if n_lens:
            print("\n[D] lensing response")
            for label in PHI_LABELS:
                run(label, n_lens)
    finally:
        if POOL:
            POOL.terminate()
            POOL = None

    # Report on everything cached so far, whichever shard made it.
    mf = collect(["mf"], args.n_meanfield)[0]
    legs = collect(RESP_LABELS + [BASE], args.n_response)   # (7 legs, sim, ...)
    diff = (legs[:-1] - legs[-1]).reshape((2, 3) + legs.shape[1:])
    dat = collect([DATA], args.n_data)[0]
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
    if n_lens:
        phi = collect(PHI_LABELS, n_lens)          # (70, sim, 4, npack)
        if phi.shape[1] >= 2:
            eps = np.array([PHI_EPS[L] for L, M in PHI_MODES])
            results["system"].update(lens_response(
                (phi[0::2] - phi[1::2]) / (2 * eps[:, None, None, None]),
                mf, diff, dat))
            results.update(lens_removed(mf, diff, dat,
                                        results["system"]["lens_V"]))
        else:
            print("lensing response: too few sims cached yet")

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
               aber_case=ABER_CASE, boost=args.boost, cmb=args.cmb,
               clpp_low=CLPP[:LPHI + 1],
               map_dec=np.degrees(enmap.posaxes(MAP_SHAPE, MAP_WCS)[0]),
               map_ra=np.degrees(enmap.posaxes(MAP_SHAPE, MAP_WCS)[1]))
    if MASK is not None:
        thumb = enmap.downgrade(MASK, max(1, round(0.5 * utils.degree / RES)))
        dec, ra = enmap.posaxes(thumb.shape, thumb.wcs)
        out.update(mask_thumb=np.asarray(thumb, np.float32),
                   mask_dec=np.degrees(dec), mask_ra=np.degrees(ra))
    for name, res in results.items():
        for key, val in res.items():
            out[f"{name}.{key}"] = val
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "summary.npz")
    tmp = f"{path}.{os.getpid()}.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)
    print(f"\nwrote {path}")

    import boost_act_plots
    boost_act_plots.make_all(path)             # -> OUT/plots


if __name__ == "__main__":
    main()