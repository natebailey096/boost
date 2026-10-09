"""
Figures for boost_act.py, drawn from its summary.npz (numpy and
matplotlib only, so they can be redrawn without rerunning anything).

    python boost_act_plots.py cache_boost/<mask>_<noise>/summary.npz

    plots/             coverage, noise, response_6x6 (K and K^-1),
                       comparison (A from the three estimates, and
                       aberration against modulation)
    plots/aberration/  lensing QE, solved without assuming the modulation;
                       unhardened shows the bias of R^-1 alone
    plots/modulation/  modulation QE, solved without assuming the aberration;
                       unhardened likewise
    plots/joint/       one velocity for both effects; inverse_response is
                       K^-1 (what the separate estimates apply) and the
                       joint map G (the weighted average of the two
                       separate estimates), with jackknife errors
    plots/report.md    the setup, where every error comes from, and the
                       mean +- error of each quantity in the figures

Boxes and ellipses are standard deviations (1 and 2 sd).  Error bars on a
mean are the pipeline's stat_err, sd / sqrt(n_data) over the data sims; the
Monte Carlo errors of the mean field and of K are not included (report.md).
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Ellipse, Rectangle

COR, TRUTH, RAW, INK = "#0072B2", "#C02A2A", "#E69F00", "#1A1A1A"
COLOURS = {"aberration": COR, "modulation": RAW, "joint": "#009E73"}
UNHARD = "#CC79A7"          # an estimate corrected with its own R alone
TITLES = {"aberration": "Aberration", "modulation": "Modulation",
          "joint": "Boost"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.2,
                     "legend.frameon": False, "figure.dpi": 150,
                     "savefig.bbox": "tight"})


def _rz(t):
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0],
                     [0, 0, 1]])


def _ry(t):
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0],
                     [-np.sin(t), 0, np.cos(t)]])


# Equatorial -> galactic (J2000): NGP at ra 192.85948, dec 27.12825, and the
# north celestial pole at l = 122.93192.
EQU2GAL = (_rz(np.radians(122.93192 - 180)) @ _ry(np.radians(27.12825 - 90))
           @ _rz(np.radians(-192.85948)))


def unit(lon, lat):
    lon, lat = np.broadcast_arrays(np.radians(lon), np.radians(lat))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def lonlat(v):
    v = np.asarray(v) / np.linalg.norm(v, axis=-1, keepdims=True)
    return (np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360,
            np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))))


def ellipse(ax, xy, cov, nsig, **kw):
    """nsig-sd ellipse of a 2x2 covariance.  With very few sims the
    covariance can be degenerate (two points span a line) and round a zero
    eigenvalue slightly negative; clip it, and draw nothing if it is empty."""
    val, vec = np.linalg.eigh(cov)
    if not (np.all(np.isfinite(val)) and np.all(np.isfinite(xy))
            and val.max() > 0):
        return
    val = np.clip(val, 0, None)
    ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    ax.add_patch(Ellipse(xy, 2 * nsig * np.sqrt(val[1]),
                         2 * nsig * np.sqrt(val[0]), angle=ang, fill=False,
                         **kw))


def jackknife_sd(reps):
    """Jackknife error from leave-one-out copies stacked along axis 0."""
    reps = np.asarray(reps)
    return np.sqrt((len(reps) - 1) * reps.var(axis=0))


def pm(value, err=None):
    """A matrix cell: the value, and under it +- err, with enough decimals
    (at least 3) to show two significant figures of err."""
    if err is None or not np.isfinite(err) or err <= 0:
        return f"{value:+.3f}"
    d = int(max(3, 1 - np.floor(np.log10(err))))
    return f"{value:+.{d}f}\n$\\pm${err:.{d}f}"


NO_JK = ("errors on the inverses need the jackknife copies of K, which this "
         "summary.npz (from an older boost_act.py) lacks: rerun boost_act.py "
         "with the same options; it reuses the cached sims and makes none")


def load(path, name=None):
    """Shared entries of summary.npz, plus those of one estimate
    (aberration, modulation, joint or system) with the prefix dropped."""
    d = dict(np.load(path))
    S = {k: v for k, v in d.items() if "." not in k}
    if name:
        S.update({k.split(".", 1)[1]: v for k, v in d.items()
                  if k.startswith(name + ".")})
        S["name"] = name
    return S


def plot_coverage(S, path):
    """Mask in equatorial Mollweide, RA increasing to the left, centred on
    RA 180, with the galactic plane and the dipole axis."""
    def x_of(ra):
        return -np.radians((np.asarray(ra) - 180 + 180) % 360 - 180)

    if "mask_thumb" in S:
        ra_ax, dec_ax, W = S["mask_ra"], S["mask_dec"], S["mask_thumb"]
    else:                                   # --mask none: the full sky
        ra_ax, dec_ax = np.linspace(0, 360, 361), np.linspace(-90, 90, 181)
        W = np.ones((dec_ax.size, ra_ax.size), np.float32)
    x = x_of(ra_ax)
    order = np.argsort(x)
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(projection="mollweide")
    W = W[:, order]
    pcm = ax.pcolormesh(x[order], np.radians(dec_ax),
                        np.where(W > 1e-3, W, np.nan), cmap="Blues",
                        vmin=0, vmax=1, shading="auto", rasterized=True)
    ra, dec = lonlat(unit(np.linspace(0, 360, 721), 0.0) @ EQU2GAL)
    xs = x_of(ra)
    cut = np.flatnonzero(np.abs(np.diff(xs)) > np.pi) + 1
    for xx, yy in zip(np.split(xs, cut), np.split(np.radians(dec), cut)):
        ax.plot(xx, yy, color=TRUTH, ls="--", lw=1.5)
    ra_d, dec_d = lonlat(S["d_true"])
    ax.plot(x_of(ra_d), np.radians(dec_d), "*", ms=16, mfc="gold", mec=INK,
            label="input dipole")
    ax.plot(x_of(ra_d + 180), np.radians(-dec_d), "*", ms=12, mfc="none",
            mec=INK, label="antipode")
    ax.plot([], [], color=TRUTH, ls="--", label="galactic plane")
    xt = np.radians(np.arange(-150, 151, 30))
    ax.set_xticks(xt, [f"{(180 - d) % 360:.0f}" for d in np.degrees(xt)],
                  fontsize=8)
    ax.set_title(f"mask {str(S['mask_kind'])}, "
                 f"f_sky = {float(S['fsky']):.3f}")
    fig.colorbar(pcm, orientation="horizontal", shrink=0.4, pad=0.06,
                 label="mask weight")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3,
              fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_noise(S, path):
    """Theory signal and the noise the filters use, as D_l in uK^2."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, name, knee in zip(axes, ("tt", "ee"), S["knee"]):
        cl, nl = S[f"cl_{name}"], S[f"nl_{name}"]
        ell = np.arange(cl.size)
        dl = ell * (ell + 1) / (2 * np.pi) * float(S["tcmb"]) ** 2
        ax.plot(ell[2:], (dl * cl)[2:], color=COR,
                label=f"$C_\\ell$ {name.upper()}")
        ax.plot(ell[nl > 0], (dl * nl)[nl > 0], color=RAW, ls="--",
                label=f"$N_\\ell$ {name.upper()}")
        ax.axvspan(S["lmin"], S["lmax"], color=COR, alpha=0.07,
                   label="used by the estimators")
        ax.axvline(knee, color=RAW, ls=":", label=r"$\ell_{\rm knee}$")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
        ax.legend(fontsize=9)
    fig.suptitle(f"noise {str(S['noise_name'])}: {float(S['noise_t']):g} "
                 r"$\mu$K-arcmin in T")
    fig.savefig(path)
    plt.close(fig)


def per_sim_K(S):
    """Each response sim's own 6x6 response, recovered exactly from the mean
    K and its leave-one-out copies: K_i = n K - (n - 1) K_jk[i]."""
    n = len(S["K_jk"])
    return n * S["K"][None] - (n - 1) * np.asarray(S["K_jk"])


def plot_response_6x6(S, path):
    """K per unit boost (rows: the six estimator components, columns: the
    six boosts) and its inverse, which the separate estimates apply.  K's
    errors are the sd of each element over the response sims divided by
    sqrt(n_r); K^-1's are the same for the inverse of each response sim's
    own K, which keeps the correlations between K's elements.  Off-diagonal
    blocks of K are each estimator's response to the other effect."""
    labels = ["aber x", "aber y", "aber z", "mod x", "mod y", "mod z"]
    Kinv, Kinv_err, n_r = np.linalg.inv(S["K"]), None, None
    if "K_jk" in S:
        try:
            inv_i = np.linalg.inv(per_sim_K(S))
            n_r = len(inv_i)
            Kinv_err = inv_i.std(axis=0, ddof=1) / np.sqrt(n_r)
        except np.linalg.LinAlgError:          # a singular single-sim K
            pass
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), layout="constrained")
    for ax, M, E, title in ((axes[0], S["K"], S["Kerr"],
                             "response $K$ per unit boost"),
                            (axes[1], Kinv, Kinv_err, "$K^{-1}$")):
        v = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
        for i in range(6):
            for j in range(6):
                txt = pm(M[i, j], None if E is None else E[i, j])
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        ax.axhline(2.5, color="k", lw=1.5)
        ax.axvline(2.5, color="k", lw=1.5)
        ax.set_yticks(range(6), labels)
        ax.set_xticks(range(6), labels, rotation=45)
        # K maps a boost to estimator components; K^-1 the other way.
        ax.set_ylabel("estimator component" if M is S["K"]
                      else "boost: effect and axis")
        ax.set_xlabel("boost: effect and axis" if M is S["K"]
                      else "estimator component")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.supxlabel(
        (f"± sd over the {n_r} response sims / $\\sqrt{{{n_r}}}$: for $K$ of "
         "each element, for $K^{-1}$ of the inverse of each sim's own $K$"
         if Kinv_err is not None else
         "$K$: ± sd / √n over the response sims.  " + NO_JK), fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(S3, path):
    """A from the three estimates, and aberration against modulation sim by
    sim: the more correlated they are, the less combining them gains."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   layout="constrained")
    allamp = np.concatenate([S["amp"] for S in S3])
    bins = np.linspace(*np.percentile(allamp, [0.5, 99.5]), 40)
    for S in S3:
        a = S["amp"]
        ax0.hist(a, bins=bins, histtype="step", lw=1.8,
                 color=COLOURS[S["name"]],
                 label=f"{TITLES[S['name']]}: {a.mean():+.3f} $\\pm$ "
                       f"{a.std(ddof=1):.3f} (sd)")
    ax0.axvline(1.0, color=TRUTH, lw=2, label="input")
    ax0.set_xlabel("amplitude $A$")
    ax0.set_ylabel("sims")
    ax0.legend(fontsize=9)
    a, m = S3[0]["amp"], S3[1]["amp"]
    ax1.scatter(a, m, s=8, color=INK, alpha=0.4, lw=0)
    ax1.plot(1, 1, "*", ms=16, color=TRUTH)
    ax1.set_xlabel("$A$ aberration")
    ax1.set_ylabel("$A$ modulation")
    ax1.set_title(f"correlation {np.corrcoef(a, m)[0, 1]:+.2f}")
    fig.savefig(path)
    plt.close(fig)


def plot_whisker(S, path):
    """v_x, v_y, v_z: mean +- 1 sd boxes, +- 2 sd whiskers.  Below, the
    mean minus the input with its error on the mean, sd / sqrt(n_data)
    over the data sims."""
    vel, v_true = S["vel"], S["v_true"]
    m, sd = vel.mean(axis=0), vel.std(axis=0, ddof=1)
    err = S["stat_err"][1:4]
    stats = [dict(med=m[k], q1=m[k] - sd[k], q3=m[k] + sd[k],
                  whislo=m[k] - 2 * sd[k], whishi=m[k] + 2 * sd[k],
                  fliers=[]) for k in range(3)]
    col = COLOURS[S["name"]]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.3, 1]))
    bp = ax0.bxp(stats, positions=range(3), widths=0.45, patch_artist=True,
                 medianprops=dict(color=INK))
    for b in bp["boxes"]:
        b.set(facecolor=col, alpha=0.5, edgecolor=col)
    ax0.hlines(v_true, np.arange(3) - 0.35, np.arange(3) + 0.35, color=TRUTH,
               lw=2, label="input", zorder=5)
    ax0.set_ylabel("velocity [km/s]")
    ax0.set_title(TITLES[S["name"]])
    ax0.legend(loc="upper left")
    ax1.errorbar(range(3), m - v_true, yerr=err, fmt="o", color=col,
                 capsize=4)
    ax1.axhline(0, color=TRUTH)
    ax1.set_ylabel("residual [km/s]")
    ax1.set_xticks(range(3), ["$v_x$", "$v_y$", "$v_z$"])
    fig.savefig(path)
    plt.close(fig)


def unhardened(S):
    """Per-sim velocities [km/s] of one estimate corrected with its own
    3x3 block R of K alone, as a single-effect analysis would, instead of
    K^-1 for both; and the bias this predicts.  Exact from the hardened
    estimates, which the summary holds: y - y_MF = K (u_ab, u_mod), so
    R^-1 (y - y_MF)_own = u_own + R^-1 K_x u_other, where K_x is the block
    of K coupling this estimator to the other effect.  The other estimate
    averages to v_in, so the bias is R^-1 K_x v_in.  S is load(summary,
    name) plus vel_other (the other estimate's vel) and K_full (system.K)."""
    own, oth = ((slice(0, 3), slice(3, 6)) if S["name"] == "aberration"
                else (slice(3, 6), slice(0, 3)))
    K = S["K_full"]
    leak = np.linalg.solve(K[own, own], K[own, oth])
    return S["vel"] + S["vel_other"] @ leak.T, leak @ S["v_true"]


def plot_unhardened(S, path):
    """The estimate with and without hardening.  Left: A from each, with
    the input and the A the leak predicts.  Middle: mean minus input per
    component, with its error on the mean.  Right: the leak itself, not
    hardened minus hardened sim by sim, which removes the noise the two
    share, against the leak K predicts."""
    v_in, v_h = S["v_true"], S["vel"]
    v_u, bias = unhardened(S)
    vv, n = v_in @ v_in, len(v_h)
    A_h, A_u = v_h @ v_in / vv, v_u @ v_in / vv
    dA = A_u - A_h
    col = COLOURS[S["name"]]
    lab_h, lab_u = r"hardened, $K^{-1}$", r"not hardened, $R^{-1}$"
    lab_p = r"predicted, $R^{-1}K_\times v_{\rm in}$"
    sem = lambda a: a.std(axis=0, ddof=1) / np.sqrt(n)
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(16, 4.8), layout="constrained",
        gridspec_kw=dict(width_ratios=[1.25, 1, 1]))

    bins = np.histogram_bin_edges(np.r_[A_h, A_u], 40)
    for A, c, lab, kw in ((A_h, col, lab_h,
                           dict(histtype="stepfilled", alpha=0.45)),
                          (A_u, UNHARD, lab_u, dict(histtype="step", lw=2))):
        ax0.hist(A, bins, color=c, label=f"{lab}:  {A.mean():+.3f} $\\pm$ "
                 f"{sem(A):.3f}", **kw)
    ax0.axvline(1, color=TRUTH, lw=2, label="input")
    ax0.axvline(1 + bias @ v_in / vv, color=UNHARD, ls="--", lw=1.5,
                label=f"predicted, not hardened:  {1 + bias @ v_in / vv:+.3f}")
    ax0.plot([], [], " ", label=f"difference, sim by sim:  {dA.mean():+.3f} "
             f"$\\pm$ {sem(dA):.3f}")
    ax0.set_xlabel("amplitude $A$")
    ax0.set_ylabel("sims")
    ax0.set_title("amplitude")
    ax0.legend(fontsize=8.5, loc="upper left")

    x = np.arange(3)
    ax1.axhline(0, color=TRUTH, lw=1.5, label="input")
    for v, dx, fmt, c, lab in ((v_h, -0.12, "o", col, lab_h),
                               (v_u, 0.12, "s", UNHARD, lab_u)):
        ax1.errorbar(x + dx, v.mean(axis=0) - v_in, yerr=sem(v), fmt=fmt,
                     ms=8, color=c, capsize=4, label=lab)
    ax1.plot(x + 0.12, bias, "_", ms=24, mew=2, color=INK, label=lab_p)
    ax1.set_ylabel("mean − input [km/s]")
    ax1.set_title("bias against the input")
    ax1.legend(fontsize=8.5)

    ax2.axhline(0, color=INK, lw=0.8)
    ax2.errorbar(x, (v_u - v_h).mean(axis=0), yerr=sem(v_u - v_h), fmt="s",
                 ms=8, color=UNHARD, capsize=4,
                 label="not hardened − hardened, sim by sim")
    ax2.plot(x, bias, "_", ms=24, mew=2, color=INK, label=lab_p)
    ax2.set_ylabel("leak [km/s]")
    ax2.set_title("leak of the other effect")
    ax2.legend(fontsize=8.5)
    for ax in (ax1, ax2):
        ax.set_xticks(x, ["$v_x$", "$v_y$", "$v_z$"])
    fig.suptitle(f"{TITLES[S['name']]}: with and without hardening")
    fig.supxlabel(f"error bars and ±: error on the mean, sd / √{n} over the "
                  "data sims", fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def with_other(summary, name):
    """load(summary, name) plus what unhardened() needs."""
    S = load(summary, name)
    other = "modulation" if name == "aberration" else "aberration"
    S.update(vel_other=load(summary, other)["vel"],
             K_full=load(summary, "system")["K"])
    return S


def plot_planes(S, path):
    """The sims in the xy, xz, yz planes with 1 and 2 sd ellipses."""
    vel, v_true = S["vel"], S["v_true"]
    col = COLOURS[S["name"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(vel[:, i], vel[:, j], s=6, color=col, alpha=0.35, lw=0)
        mu = vel[:, [i, j]].mean(axis=0)
        for ns in (1, 2):
            ellipse(ax, mu, np.cov(vel[:, [i, j]].T), ns, color=col, lw=1.5)
        ax.plot(*mu, "o", color=col, mec=INK, label="mean of sims")
        ax.plot(v_true[i], v_true[j], "*", ms=16, color=TRUTH, label="input")
        ax.set_xlabel(f"$v_{'xyz'[i]}$ [km/s]")
        ax.set_ylabel(f"$v_{'xyz'[j]}$ [km/s]")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend()
    fig.suptitle(TITLES[S["name"]])
    fig.savefig(path)
    plt.close(fig)


def plot_response_matrix(S, path):
    """The estimator's response to its own effect (its diagonal block of K)
    +- error on the mean, the inverse of that block, and the eigenvalues of
    its symmetric part."""
    R, Rerr = S["R"], S["Rerr"]
    Rinv_err = (jackknife_sd([np.linalg.inv(r) for r in S["R_jk"]])
                if "R_jk" in S else None)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for ax, M, E, name in ((axes[0], R, Rerr, "$R$ (own block of $K$)"),
                           (axes[1], np.linalg.inv(R), Rinv_err, "$R^{-1}$")):
        v = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
        for i in range(3):
            for j in range(3):
                txt = pm(M[i, j], None if E is None else E[i, j])
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5)
        ax.set_xticks(range(3), ["x", "y", "z"])
        ax.set_yticks(range(3), ["x", "y", "z"])
        # R maps a boost to the estimator's components; R^-1 the other way.
        ax.set_xlabel("boost along" if M is R else "estimator component")
        ax.set_ylabel("estimator component" if M is R else "boost along")
        ax.set_title(name)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    val, vec = np.linalg.eigh(0.5 * (R + R.T))
    labels = []
    for v in vec.T:
        ra, dec = lonlat(v if v[2] >= 0 else -v)    # an axis: name its north end
        labels.append(f"axis ra {ra:.0f}\ndec {dec:+.0f}")
    axes[2].bar(range(3), val, color=COLOURS[S["name"]])
    axes[2].set_xticks(range(3), labels, fontsize=8.5)
    axes[2].set_ylabel("eigenvalue of $(R+R^T)/2$")
    fig.suptitle(TITLES[S["name"]])
    fig.supxlabel(
        (f"$R$: ± error on the mean; $R^{{-1}}$: ± jackknife error over the "
         f"{len(S['R_jk'])} response sims.  The separate estimates apply rows "
         "of the full $K^{-1}$ (joint/inverse_response.png), not $R^{-1}$."
         if Rinv_err is not None else NO_JK), fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def plot_inverse_response(S, path):
    """What the estimates apply to the six estimator components (mean field
    removed): K^-1, whose first three rows give the separate u_aber and the
    last three u_mod, and the joint G, which gives the one u as a weighted
    average of the two.  Each is +- its jackknife error over the response
    sims; for G the weights are recomputed for each leave-one-out K."""
    comps = ["aber x", "aber y", "aber z", "mod x", "mod y", "mod z"]
    have = "K_jk" in S
    Kinv = np.linalg.inv(S["K"])
    panels = [(Kinv,
               jackknife_sd([np.linalg.inv(k) for k in S["K_jk"]]) if have
               else None,
               [rf"$u_{{\rm aber}}$ {a}" for a in "xyz"]
               + [rf"$u_{{\rm mod}}$ {a}" for a in "xyz"],
               "$K^{-1}$: separate estimates (rows: the u each one gives)")]
    if "G" in S:
        panels.append((S["G"], jackknife_sd(S["G_jk"]), [f"$u$ {a}" for a in "xyz"],
                       "$G$: joint estimate, weighted average of the two "
                       "separate estimates" if "w_joint" in S else
                       "$G$: joint estimate (older run, covariance-weighted)"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 4 + 2.2 * len(panels)),
                             height_ratios=[len(p[2]) for p in panels],
                             layout="constrained", squeeze=False)
    for ax, (M, E, rows, title) in zip(axes[:, 0], panels):
        v = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        for i in range(len(rows)):
            for j in range(6):
                txt = pm(M[i, j], None if E is None else E[i, j])
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5)
        ax.axvline(2.5, color="k", lw=1.5)
        if len(rows) == 6:
            ax.axhline(2.5, color="k", lw=1.5)
        ax.set_xticks(range(6), comps, rotation=45)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[-1, 0].set_xlabel("estimator component (mean field removed)")
    fig.supxlabel(
        (f"± jackknife error over the {len(S['K_jk'])} response sims; for $G$ "
         "the weights are recomputed for each leave-one-out $K$"
         if have else NO_JK), fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def mean_direction(S):
    """Galactic (l, b) of the mean velocity over the data sims, with its
    error, and its angle from the input.  The error on the mean vector,
    cov(v) / n_data, is carried to (l, b) by drawing from it.  Unlike the
    mean of the per-sim l and b, this is not pulled away from the input when
    single sims are noisy.  l is unwrapped around the input."""
    vel = S["vel"]
    vbar = vel.mean(axis=0)
    l_in = lonlat(S["d_true"] @ EQU2GAL.T)[0]
    lv, bv = lonlat(vbar @ EQU2GAL.T)
    lv = l_in + (lv - l_in + 180) % 360 - 180
    draws = np.random.default_rng(0).multivariate_normal(
        vbar, np.cov(vel, rowvar=False) / len(vel), 20000)
    ld, bd = lonlat(draws @ EQU2GAL.T)
    ld = lv + (ld - lv + 180) % 360 - 180
    angle = np.degrees(np.arccos(np.clip(
        vbar @ S["d_true"] / np.linalg.norm(vbar), -1, 1)))
    return lv, bv, ld.std(ddof=1), bd.std(ddof=1), angle


def plot_amplitude_direction(S, path):
    """Amplitude A, and the recovered directions in galactic coordinates:
    per-sim directions with 1 and 2 sd ellipses, their mean, the direction
    of the mean velocity with its error, and the input; the text box gives
    the numbers."""
    amp = S["amp"]
    col = COLOURS[S["name"]]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   layout="constrained")
    sd = amp.std(ddof=1)
    ax0.hist(amp, bins=30, color=col, alpha=0.4,
             label=f"$A$ = {amp.mean():+.3f} $\\pm$ {sd:.3f} (sd)\n"
                   f"N: {len(amp)} data, {int(S['n_mf'])} mf, "
                   f"{int(S['n_resp'])} response")
    for k in (-2, -1, 1, 2):
        ax0.axvline(amp.mean() + k * sd, color=col, ls=":", lw=1)
    ax0.axvline(amp.mean(), color=col, ls="--")
    ax0.axvline(1.0, color=TRUTH, lw=2, label="input")
    ax0.set_xlabel("amplitude $A$")
    ax0.set_ylabel("sims")
    ax0.legend(fontsize=9)

    # l is unwrapped around the input so the cloud never splits at l = 0,
    # and runs right to left as on the sky.  The aspect 1/cos b makes equal
    # distances on the sky look equal; sigma_l is a coordinate spread.
    l_in, b_in = lonlat(S["d_true"] @ EQU2GAL.T)
    l, b = lonlat(S["dir"] @ EQU2GAL.T)
    l = l_in + (l - l_in + 180) % 360 - 180
    lm, bm, sl, sb = l.mean(), b.mean(), l.std(ddof=1), b.std(ddof=1)
    ax1.scatter(l, b, s=14, color=col, alpha=0.55, lw=0, label="per sim")
    for ns in (1, 2):
        ellipse(ax1, (lm, bm), np.cov(l, b), ns, color=col, ls="--", lw=1)
    ax1.plot(lm, bm, "o", ms=8, color=col, mec=INK, zorder=5,
             label="mean of per-sim directions")
    lv, bv, elv, ebv, angle = mean_direction(S)
    ax1.errorbar(lv, bv, xerr=elv, yerr=ebv, fmt="D", ms=7, color=INK,
                 mfc="white", capsize=3, zorder=7,
                 label="mean velocity ± error")
    ax1.plot(l_in, b_in, "*", ms=16, color=TRUTH, zorder=6, label="input")
    ax1.invert_xaxis()
    ax1.set_aspect(1 / np.cos(np.radians(b_in)), adjustable="datalim")
    ax1.set_xlabel(r"galactic longitude $\ell$ [deg]")
    ax1.set_ylabel(r"galactic latitude $b$ [deg]")
    ax1.text(0.03, 0.03,
             f"input   $\\ell$ = {l_in:.2f}°,  $b$ = {b_in:+.2f}°\n"
             f"mean of sims   $\\ell$ = {lm:.2f}°,  $b$ = {bm:+.2f}°\n"
             f"sd       $\\sigma_\\ell$ = {sl:.2f}°,  $\\sigma_b$ = {sb:.2f}°\n"
             f"mean velocity   $\\ell$ = {lv:.2f} $\\pm$ {elv:.2f}°,  "
             f"$b$ = {bv:+.2f} $\\pm$ {ebv:.2f}°  "
             f"({angle:.2f}° from input)",
             transform=ax1.transAxes, fontsize=9, va="bottom",
             bbox=dict(fc="white", ec="0.8", alpha=0.9))
    ax1.legend(fontsize=9, loc="upper right")
    fig.suptitle(TITLES[S["name"]])
    fig.savefig(path)
    plt.close(fig)


def real_basis(A, pl, pm):
    """Packed complex alm (..., npack) -> real-harmonic r_LM, L >= 1,
    M = -L..L: r_L0 = a_L0, r_LM = sqrt2 (-1)^M Re a_LM,
    r_L,-M = -sqrt2 (-1)^M Im a_LM.  Times sqrt(3/4pi), L = 1 is (y, z, x)."""
    rows, cols = [], []
    for l in range(1, pl.max() + 1):
        for m in range(-l, l + 1):
            k = np.flatnonzero((pl == l) & (pm == abs(m)))[0]
            s = np.sqrt(2) * (-1) ** abs(m)
            cols.append(A[..., k].real if m == 0 else
                        s * A[..., k].real if m > 0 else -s * A[..., k].imag)
            rows.append((l, m))
    return rows, np.stack(cols, axis=-1)


def plot_alm_response(S, path):
    """(L, M) response to a unit boost of the estimator's own effect along
    x, y, z.  The boxed L = 1 block is R; everything below it is leakage.
    The linear band of the colour scale is the median MC error on the mean."""
    pl, pm = S["pack_l"], S["pack_m"]
    c = np.sqrt(3 / (4 * np.pi))
    rows, V = real_basis(S["Ralm"].T, pl, pm)
    V = c * V.T
    err = c * np.array([S["Ralm_sd"][(pl == l) & (pm == abs(m))][0]
                        for l, m in rows]) / np.sqrt(int(S["n_resp"]))
    thresh = float(np.median(err))
    vmax = max(np.abs(V).max(), 10 * thresh)
    fig, ax = plt.subplots(figsize=(5.4, 10), layout="constrained")
    im = ax.imshow(V, cmap="RdBu_r", aspect="auto",
                   norm=SymLogNorm(thresh, vmin=-vmax, vmax=vmax))
    for i in range(len(rows)):
        for j in range(3):
            ax.text(j, i, f"{V[i, j]:+.1e}", ha="center", va="center",
                    fontsize=7)
    lr = np.array([l for l, m in rows])
    for e in np.flatnonzero(np.diff(lr)) + 1:
        ax.axhline(e - 0.5, color="white", lw=1.6)
    ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=INK, lw=2))
    ax.set_xticks(range(3), ["x", "y", "z"])
    ax.set_yticks(range(len(rows)), [f"{l},{m:+d}" for l, m in rows],
                  fontsize=7.5)
    ax.set_xlabel("boost direction")
    ax.set_ylabel("$(L, M)$")
    ax.set_title(TITLES[S["name"]], fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035,
                 label=r"$\sqrt{3/4\pi}\,\langle r_{LM}\rangle$ per unit boost")
    fig.savefig(path)
    plt.close(fig)


def plot_leakage(S, path):
    """Coherent power the true boost (both effects) leaves at each L of this
    estimator, from the paired response sims (+- 1 and 2 jackknife sd), with
    the noisier data sims minus mean field as a cross-check."""
    L = np.arange(1, S["leak"].size)
    y, e, dat = S["leak"][1:], S["leakerr"][1:], S["datleak"][1:]
    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.errorbar(L, y, yerr=2 * e, fmt="none", ecolor=COR, alpha=0.4, capsize=4)
    ax.errorbar(L, y, yerr=e, fmt="o-", color=COR, lw=1.5, elinewidth=2.5,
                label="paired response sims")
    ax.plot(L, dat, "s", mfc="none", mec="0.4", label="data sims - mean field")
    vals = np.concatenate([y, dat])
    if np.all(vals > 0):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=max(np.abs(y).min(),
                                              1e-4 * np.abs(vals).max()))
        ax.axhline(0, color="0.5", lw=1)
    ax.set_xticks(L)
    ax.set_xlabel("reconstruction multipole $L$")
    ax.set_ylabel(r"$C_L$ in units of $u^2$")
    ax.set_title(TITLES[S["name"]])
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def write_report(summary, path):
    """report.md: the setup, where every number and error comes from, and
    the mean +- error of each quantity in the figures, so that a run can be
    read back long after it finished."""
    S, sy, lc = (load(summary, n) for n in (None, "system", "lensing_cases"))
    E = {n: load(summary, n) for n in TITLES}
    n_mf, n_r, n_d = (int(S[k]) for k in ("n_mf", "n_resp", "n_data"))
    req = [int(x) for x in S["requested"]]
    ra_in, dec_in = lonlat(S["d_true"])
    l_in, b_in = lonlat(S["d_true"] @ EQU2GAL.T)
    v_in = S["v_true"]
    comp = ["aber x", "aber y", "aber z", "mod x", "mod y", "mod z"]
    out = []
    w = out.append

    def corr_err(r, n):
        return (1 - r * r) / np.sqrt(n - 1)

    w("# Doppler boost simulations: report\n")
    w(f"Written {time.strftime('%Y-%m-%d %H:%M')} from `summary.npz` (cache tag "
      f"`{S['cache_tag']}`).  Simulations: {n_mf} mean-field, {n_r} response, "
      f"{n_d} data.")
    if [n_mf, n_r, n_d] != req:
        w(f"\n**Partial snapshot.**  {req[0]} / {req[1]} / {req[2]} were "
          f"requested; the rest were not in the cache yet.")

    w("\n## Setup (`coverage.png`, `noise.png`)\n")
    w("| | |\n|---|---|")
    w(f"| multipoles | {int(S['lmin'])} ≤ ℓ ≤ {int(S['lmax'])} |")
    mask = str(S["mask_kind"])
    w(f"| maps | {float(S['res_arcmin']):g}′ CAR |")
    w("| mask | " + (f"ACT DR6 lensing (`{S['mask_file']}`), f_sky = "
                     f"{float(S['fsky']):.4f}, w2 = {float(S['w2']):.4f} |"
                     if mask == "dr6" else "none (full sky) |"))
    w(f"| noise (`--noise {S['noise_name']}`) | {float(S['noise_t']):g} "
      f"μK-arcmin in T, "
      f"×√2 in P; beam {float(S['beam']):g}′ FWHM; 1/f knees "
      f"{S['knee'][0]:g} (T), {S['knee'][1]:g} (P); slopes {S['alpha'][0]:g} "
      f"(T), {S['alpha'][1]:g} (P); zero below ℓ = {int(S['lmin'])} |")
    w(f"| input boost | β = {float(S['beta']):.4e} = "
      f"{np.linalg.norm(v_in):.1f} km/s toward galactic ℓ = {l_in:.2f}°, "
      f"b = {b_in:+.2f}° (ra {ra_in:.2f}°, dec {dec_in:+.2f}°) |")
    w(f"| input velocity | v_x = {v_in[0]:+.1f}, v_y = {v_in[1]:+.1f}, "
      f"v_z = {v_in[2]:+.1f} km/s (equatorial axes) |")
    w(f"| modulation factor | b = {float(S['b_nu']):.4f} at "
      f"{float(S['freq']) / 1e9:g} GHz |")
    w(f"| aberration estimator | lensing quadratic estimator, {S['aber_case']} |")
    w("| modulation estimator | TT amplitude modulation (falafel `qe_mask`, "
      "darby_birefringence_patchytau), tempura `amp` normalisation "
      "(darbys_tau_edits2) |")

    w("\n## Where the numbers come from\n")
    w("Each estimate turns the reconstructed dipole into a boost vector "
      "u = β d.  The amplitude A = u·u_in / |u_in|² is 1 when unbiased.  "
      "Velocities are c u in km/s along the equatorial axes; ℓ and b are "
      "the galactic coordinates of each sim's direction, ℓ unwrapped around "
      "the input.\n")
    w("**Three estimates.**  *Aberration* and *modulation* are solved together "
      "from the 6×6 response K, so neither assumes the other has the same "
      "velocity.  *Joint* assumes one velocity for both: per axis, it is a "
      "weighted average of the aberration and modulation estimates, with "
      "weight w = s_mod² / (s_ab² + s_mod²) on aberration, where s is the sd "
      "of that estimate over the mean-field sims, so the less noisy one counts "
      "more.  Each separate estimate is unbiased for the same u, so any "
      "weights that sum to 1 are; these are the inverse-variance weights when "
      "the two are uncorrelated, and the correlation between them is not "
      "used.\n")
    w("**Response K.**  Each response sim is reconstructed unboosted and "
      "boosted by +β along x, y and z, with aberration only and with "
      "modulation only; (boosted − unboosted)/β is one column of K.  The "
      "mean field cancels within each sim, and K absorbs the mask and the "
      "full-sky normalisations at L = 1.  The difference is one-sided, so K "
      "has a truncation bias, expected at order β (~0.1%), that is not in "
      "the errors below.\n")
    w("**sd** is the spread of single sims: the error one measurement on one "
      "sky would have.  The boxes, whiskers, ellipses and histograms show it.\n")
    w("**Error on the mean** is how well the average over all the sims is "
      "known, so it sets how precisely this run tests for bias: compare "
      f"mean − input with it (last column of the tables).  It is the sd of "
      f"the {n_d} data sims divided by √{n_d}.  The mean field (averaged over "
      f"{n_mf} sims) and K (from {n_r} response sims) are shared by every "
      "data sim, and their Monte Carlo errors are not included.  The mean "
      "field's is about √(n_data/n_mf) times the quoted error, so the quoted "
      f"error is low by about a factor √(1 + n_data/n_mf) = "
      f"{np.sqrt(1 + n_d / n_mf):.2f} here; K's is small, because each "
      "response sim cancels most of its CMB.\n")
    w("**Other errors.**\n")
    w(f"- K and its diagonal blocks R: the sd of each element over the {n_r} "
      f"response sims divided by √{n_r}.  Eigenvalues of R: jackknife over the "
      f"response sims.")
    w("- Null test: the second half of the mean-field sims analysed with the "
      "first half as their mean field.  Expect 0; the error is "
      "sd · √(1/N₂ + 1/N₁).")
    w(f"- (L, M) response: the sd over the response sims divided by √{n_r}.")
    w("- Leakage C_L: the power of the mean response minus its Monte Carlo "
      "noise, with a jackknife error over the response sims.  The data-sim "
      "cross-check (data minus mean field) is far noisier; no error is quoted.")
    w("- Correlation coefficients r: approximate standard error "
      "(1 − r²)/√(N − 1).")

    w("\n## Response (`response_6x6.png`) and correlations\n")
    w("K per unit boost, mean ± error on the mean.  Rows: estimator "
      "components.  Columns: boost effect and axis.\n")
    w("| | " + " | ".join(comp) + " |\n|---|" + "---|" * 6)
    for i in range(6):
        w(f"| {comp[i]} QE | " + " | ".join(
            f"{sy['K'][i, j]:+.4f} ± {sy['Kerr'][i, j]:.4f}" for j in range(6))
          + " |")
    w(f"\nMean field relative to the signal, |mean field| / |K u_in|: "
      f"aberration QE {sy['mf_ratio'][0]:.2f}, modulation QE "
      f"{sy['mf_ratio'][1]:.2f}.\n")
    if "w_joint" in sy:
        w("Joint weight on the aberration estimate (modulation gets 1 − w), "
          "x, y, z: " + ", ".join(f"{v:.3f}" for v in sy["w_joint"]) + ".\n")
    w(f"Correlation of the six components over the {n_mf} mean-field sims "
      f"(error about {1 / np.sqrt(n_mf - 1):.3f} near zero):\n")
    w("| | " + " | ".join(f"{c} QE" for c in comp) + " |\n|---|" + "---|" * 6)
    for i in range(6):
        w(f"| {comp[i]} QE | " + " | ".join(f"{sy['corr'][i, j]:+.3f}"
                                           for j in range(6)) + " |")

    w("\n## The three estimates side by side (`comparison.png`)\n")
    w("| estimate | A mean | error on the mean | sd (one sim) | "
      "(mean − 1)/error |\n|---|---|---|---|---|")
    for n, e in E.items():
        a, er, sd = e["stat_mean"][0], e["stat_err"][0], e["stat_sd"][0]
        w(f"| {n} | {a:+.4f} | {er:.4f} | {sd:.4f} | {(a - 1) / er:+.2f} |")
    r = np.corrcoef(E["aberration"]["amp"], E["modulation"]["amp"])[0, 1]
    w(f"\nCorrelation of A between aberration and modulation, sim by sim: "
      f"r = {r:+.3f} ± {corr_err(r, n_d):.3f}.")

    label = ["A", "v_x [km/s]", "v_y [km/s]", "v_z [km/s]", "ℓ [deg]",
             "b [deg]"]
    truth = [1.0, *v_in, l_in, b_in]
    fmt = [".4f", ".1f", ".1f", ".1f", ".2f", ".2f"]
    for n, e in E.items():
        w(f"\n## {TITLES[n][0].upper() + TITLES[n][1:]} (`{n}/`)\n")
        w("Mean ± error on the mean (sd/√n_data), and the sd of one sim "
          "(`whisker.png`, `planes.png`, `amplitude_direction.png`).\n")
        w("| quantity | input | mean | error on the mean | sd (one sim) "
          "| (mean − input)/error |\n|---|---|---|---|---|---|")
        for k in range(6):
            f, mu, er = fmt[k], e["stat_mean"][k], e["stat_err"][k]
            w(f"| {label[k]} | {truth[k]:+{f}} | {mu:+{f}} | {er:{f}} | "
              f"{e['stat_sd'][k]:{f}} | {(mu - truth[k]) / er:+.2f} |")
        nl, ne = e["null"], float(e["null_err"])
        w(f"| null A | 0 | {nl.mean():+.4f} | {ne:.4f} | "
          f"{nl.std(ddof=1):.4f} | {nl.mean() / ne:+.2f} |")
        ubar = e["vel"].mean(axis=0) / np.linalg.norm(e["vel"].mean(axis=0))
        sep = np.degrees(np.arccos(np.clip(e["dir"] @ S["d_true"], -1, 1)))
        c = np.corrcoef(e["vel"].T)
        w(f"\nThe mean velocity points "
          f"{np.degrees(np.arccos(np.clip(ubar @ S['d_true'], -1, 1))):.2f}° "
          f"from the input; single sims are off by a median of "
          f"{np.median(sep):.1f}°.\n")
        w("Velocity correlations over the data sims (`planes.png`): " + ", ".join(
            f"r(v_{'xyz'[i]}, v_{'xyz'[j]}) = {c[i, j]:+.3f} ± "
            f"{corr_err(c[i, j], n_d):.3f}" for i, j in ((0, 1), (0, 2), (1, 2)))
          + ".")
        if "R" not in e:
            continue
        w("\nResponse to its own effect, the diagonal block R of K "
          "(`response_matrix.png`), mean ± error on the mean:\n")
        w("| | boost x | boost y | boost z |\n|---|---|---|---|")
        for i in range(3):
            w(f"| {'xyz'[i]} | " + " | ".join(
                f"{e['R'][i, j]:+.4f} ± {e['Rerr'][i, j]:.4f}"
                for j in range(3)) + " |")
        val, vec = np.linalg.eigh(0.5 * (e["R"] + e["R"].T))
        axes = [lonlat(v if v[2] >= 0 else -v) for v in vec.T]
        w("\nEigenvalues of (R + Rᵀ)/2 with jackknife errors: " + "; ".join(
            f"{val[k]:.4f} ± {e['eig_err'][k]:.4f} along ra {a[0]:.0f}°, "
            f"dec {a[1]:+.0f}°" for k, a in enumerate(axes)) + ".")
        w("\nLeakage (`leakage.png`): coherent power the input boost leaves "
          "at each L of this estimator, in units of u².  N_L is the "
          "reconstruction noise of one sim.\n")
        w("| L | C_L, response sims | jackknife error | data sims − mean "
          "field | N_L |\n|---|---|---|---|---|")
        for l in range(1, len(e["leak"])):
            w(f"| {l} | {e['leak'][l]:+.3e} | {e['leakerr'][l]:.1e} | "
              f"{e['datleak'][l]:+.3e} | {e['noise'][l]:.3e} |")
        pl, pm = S["pack_l"], S["pack_m"]
        cfac = np.sqrt(3 / (4 * np.pi))
        rows, V = real_basis(e["Ralm"].T, pl, pm)
        V = cfac * V.T
        err = cfac * np.array([e["Ralm_sd"][(pl == l) & (pm == abs(m))][0]
                               for l, m in rows]) / np.sqrt(n_r)
        w("\n(L, M) response to a unit boost of its own effect, real "
          "harmonics times √(3/4π), mean ± error on the mean "
          "(`alm_response.png`).  The L = 1 rows M = −1, 0, +1 are the y, z, "
          "x rows of R.\n")
        w("| (L, M) | boost x | boost y | boost z |\n|---|---|---|---|")
        for i, (l, m) in enumerate(rows):
            w(f"| {l}, {m:+d} | " + " | ".join(
                f"{V[i, j]:+.2e} ± {err[i, j]:.0e}" for j in range(3)) + " |")

    w("\n## Without hardening (`aberration/unhardened.png`, "
      "`modulation/unhardened.png`)\n")
    w("Each estimate corrected with its own 3×3 block R of K alone, as a "
      "single-effect analysis would, instead of K⁻¹ for both.  The other "
      "effect then leaks in, on average R⁻¹ K× v_in, where K× is the block of "
      "K coupling the estimator to the other effect.  These are computed "
      "exactly from the hardened estimates (y − y_MF = K (u_ab, u_mod), so R⁻¹ "
      "applied to one estimator's three components is that estimate plus "
      "R⁻¹ K× times the other's), not from new simulations.  The two are "
      "from the same sims, so their difference, sim by sim, is far less noisy "
      "than either.\n")
    w("| estimate | A hardened | A not hardened | difference, sim by sim "
      "| difference predicted |\n|---|---|---|---|---|")
    for name in ("aberration", "modulation"):
        Sx = with_other(summary, name)
        v_u, bias = unhardened(Sx)
        A_h = Sx["vel"] @ v_in / (v_in @ v_in)
        A_u = v_u @ v_in / (v_in @ v_in)
        w(f"| {TITLES[name]} | {A_h.mean():+.4f} ± "
          f"{A_h.std(ddof=1) / np.sqrt(n_d):.4f} | {A_u.mean():+.4f} ± "
          f"{A_u.std(ddof=1) / np.sqrt(n_d):.4f} | {(A_u - A_h).mean():+.4f} "
          f"± {(A_u - A_h).std(ddof=1) / np.sqrt(n_d):.4f} | "
          f"{bias @ v_in / (v_in @ v_in):+.4f} |")

    w("\n## Aberration amplitude for each lensing combination\n")
    w("The separate aberration estimate (solved with the modulation "
      "estimator) for each choice of lensing estimators.\n")
    w("| lensing estimators | A mean | error on the mean | sd (one sim) |\n"
      "|---|---|---|---|")
    for name, (a, er, sd) in zip(lc["names"], lc["A"]):
        w(f"| {name} | {a:+.4f} | {er:.4f} | {sd:.4f} |")

    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    os.replace(tmp, path)
    print("wrote", path)


PER_ESTIMATOR = [plot_whisker, plot_planes, plot_response_matrix,
                 plot_amplitude_direction, plot_alm_response, plot_leakage]


def make_all(summary, outdir=None):
    """Every figure, into plots/ beside summary.npz.  Each is drawn under a
    temporary name and renamed, so runs finishing together never leave a
    half-written file."""
    outdir = outdir or os.path.join(os.path.dirname(summary), "plots")

    def save(fn, S, path):
        tmp = f"{path[:-4]}.{os.getpid()}.png"
        fn(S, tmp)
        os.replace(tmp, path)
        print("wrote", path)

    jobs = [(outdir, None, [plot_coverage, plot_noise]),
            (outdir, "system", [plot_response_6x6]),
            (os.path.join(outdir, "aberration"), "aberration", PER_ESTIMATOR),
            (os.path.join(outdir, "modulation"), "modulation", PER_ESTIMATOR),
            (os.path.join(outdir, "joint"), "joint",
             [plot_whisker, plot_planes, plot_amplitude_direction]),
            (os.path.join(outdir, "joint"), "system", [plot_inverse_response])]
    for folder, name, fns in jobs:
        os.makedirs(folder, exist_ok=True)
        S = load(summary, name)
        for fn in fns:
            save(fn, S, os.path.join(folder, fn.__name__[5:] + ".png"))
    for name in ("aberration", "modulation"):
        save(plot_unhardened, with_other(summary, name),
             os.path.join(outdir, name, "unhardened.png"))
    save(plot_comparison, [load(summary, n) for n in TITLES],
         os.path.join(outdir, "comparison.png"))
    write_report(summary, os.path.join(outdir, "report.md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    make_all(a.summary, a.out)