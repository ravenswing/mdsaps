import gif
from matplotlib import pyplot as plt
import numpy as np
import sys
import subprocess

sys.path.append("/home/rhys/phd_tools/SAPS")
import load

# (Optional) Set the dots per inch resolution to 300
gif.options.matplotlib["dpi"] = 100

# Symbol for angstroms
ANG = "\u212b"

cv2 = "ligrmsd"
# Variables that define the time and time resolution
start = 0  # begin (steps)
end = 250001  # end (steps)
stride = 100  # no. of steps
scale = stride * 0.002  # value for scaling x-axis of time plot


def animate_fes(fes_path, colvar_path, out_path):
    data, lab = load.fes(fes_path, False)
    data[0] = np.multiply(data[0], 10)
    data[1] = np.multiply(data[1], 10)
    data[2] = data[2] / 4.184
    cmax = np.amax(data[2][np.isfinite(data[2])]) + 1

    conts = np.arange(0.0, cmax, 2.0)
    colvar = load.colvar(colvar_path)
    cut_xy = colvar[start:end:stride]
    print(cut_xy.iloc[0])

    # Decorate a plot function with @gif.frame
    @gif.frame
    def plot(i, xi, yi):
        if i > 0:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, height_ratios=[3, 2], figsize=[5.3, 8], layout="constrained"
            )
            ax1.contour(
                data[0],
                data[1],
                data[2],
                conts,
                colors="k",
                linewidths=0.5,
                linestyles="dashed",
                alpha=0.6,
                antialiased=True,
            )
            ax1.contourf(
                data[0],
                data[1],
                data[2],
                conts,
                cmap="RdYlBu",
                alpha=0.6,
                antialiased=True,
            )
            ax1.plot(xi, yi, alpha=0.7, linewidth=0.8, c="k")
            ax1.scatter(
                [xi.iloc[-1]],
                [yi.iloc[-1]],
                s=60,
                alpha=0.9,
                c="#ffffff",
                edgecolor="k",
                linewidths=0.5,
                zorder=3,
            )
            ax1.scatter(
                [xi.iloc[0]],
                [yi.iloc[0]],
                s=80,
                marker="X",
                alpha=0.9,
                c="#e05252",
                edgecolor="k",
                linewidths=0.5,
                zorder=3,
            )
            ax1.set_xlim((-1, 45))
            ax1.set_ylim((-1, 50))
            ax1.set_xlabel(f"Projection / {ANG}")
            ax1.set_ylabel(f"{cv2} / {ANG}")
            ax1.text(
                0.95,
                0.95,
                f"{len(xi)*scale:.1f} ns",
                horizontalalignment="right",
                verticalalignment="center",
                transform=ax1.transAxes,
                size=12,
            )

            ax2.plot(np.arange(0, len(xi)) * scale, xi, linewidth=0.5, c="#5252e0")
            ax2.scatter(
                [len(xi) * scale],
                [xi.iloc[-1]],
                s=40,
                alpha=0.9,
                c="#ffffff",
                edgecolor="k",
                linewidths=0.5,
                zorder=3,
            )
            ax2.set_xlim((0, (end - 1) * 0.002))
            ax2.set_ylim((-1, 45))
            ax2.set_xlabel("Time / ns")
            ax2.set_ylabel(f"Projection / {ANG}")

    # Construct "frames"
    frames = []
    for i in np.arange(len(cut_xy)):
        if i > 0:
            x = cut_xy.iloc[0:i]["pp.proj"] * 10
            y = cut_xy.iloc[0:i][cv2] * 10
            frames.append(plot(i, x, y))

    # Save "frames" to gif with a specified duration (ms) between each frame
    gif.save(frames, out_path, duration=20)


def gif_to_mp4(filename):
    cmd = (
        f"ffmpeg -i {filename}.gif "
        "-y -movflags faststart -pix_fmt yuv420p "
        '-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
        f"{filename}.mp4"
    )
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print(
            "Error code:", error.returncode, ". Output:", error.output.decode("utf-8")
        )


# For fun-metaD
for system in ["a2b1"]:
    for lig in ["A769", "PF739", "SC4", "MT47", "MK87"]:
        for i in [1, 2, 3]:
            wd = f"/media/rhys/Storage/ampk_metad_all_data/fun-RMSD/{system}+{lig}/R{i}"
            fes_path = f"{wd}/{system}+{lig}_R{i}_FES"
            colvar_path = f"{wd}/COLVAR"
            out_path = f"Figures/AMPK/fun-RMSD/{system}_{lig}_R{i}.gif"
            animate_fes(fes_path, colvar_path, out_path)
            gif_to_mp4(f"Figures/AMPK/fun-RMSD/{system}_{lig}_R{i}")

"""
for method in ['opes_metad_funnel', 'opes_explore_funnel']:
    fes_path = f"/media/rhys/Storage2/OPES/second_attempt/{method}/fes.dat"
    colvar_path = f"/media/rhys/Storage2/OPES/second_attempt/{method}/COLVAR"
    out_path = f"Figures/second_attempt/{method}.gif"
    animate_fes(fes_path, colvar_path, out_path)
    gif_to_mp4(f"Figures/second_attempt/{method}")
"""
