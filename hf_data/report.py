"""Downloadable PDF report — the AQUAH ARW (report-writer agent), integrated.

Same three-step flow as AQUAH_v0.3/tools/agent_report_writer.py, driven by the
VERBATIM agent/task/vision prompts copied to agents/report_config.yaml:
  1. make_report_for_figures : vision LLM reads the rendered figures
  2. generate_simulation_summary : metrics/metadata paragraph (incl. KGE)
  3. report-writer agent : role/goal/backstory + write_report task -> Markdown
then Markdown -> PDF via pypandoc + xelatex (both already in the Docker image,
inherited from AQUAH). Differences from AQUAH: figures are rendered here from
the live SimJob (matplotlib hydrograph, peak 2-D streamflow frame, DEM/FACC
panels from the basin clip store), and the LLM calls go through hf_data.llm
(vLLM -> OpenAI router) instead of crewai — same prompts, same agent.

Fallbacks so the button always yields a file: no LLM -> deterministic report
from the metrics; no pandoc/xelatex (local dev) -> the Markdown itself.
"""
from __future__ import annotations

import base64
import io
import math
import os
import re
import shutil
import threading
from datetime import datetime

from hf_data.statecache import CACHE_DIR

REPORT_DIR = os.path.join(CACHE_DIR, "reports")
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _config() -> dict:
    import yaml
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "agents", "report_config.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


# ---- figures (matplotlib, headless) -----------------------------------------
def _fig_results(rows: list[dict], meta: dict, metrics: dict, path: str):
    """AQUAH 'results.png': sim vs obs hydrograph + precipitation bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [datetime.strptime(r["time"], "%Y-%m-%d %H:%M") for r in rows]
    sim = [r.get("sim_q") for r in rows]
    obs = [r.get("obs_q") for r in rows]
    pr = [r.get("precip") or 0.0 for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    ax2 = ax.twinx()
    ax2.bar(t, pr, width=0.036, color="#5b9bd5", alpha=0.55, label="Precip")
    ax2.set_ylim(0, max(0.1, max(pr)) * 3.2)
    ax2.invert_yaxis()
    ax2.set_ylabel("Precip (mm/h)")
    if any(o is not None and not (isinstance(o, float) and math.isnan(o)) for o in obs):
        ax.plot(t, obs, color="#555", lw=1.4, label="Observed (USGS)")
    ax.plot(t, sim, color="#0a8f6a", lw=2.0, label="Simulated (CREST/EF5)")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title(f"USGS {meta.get('id')} · {meta.get('name', '')} — simulated vs observed")
    skill = "  ".join(f"{lbl}={metrics[k]}" for k, lbl in
                      (("nsce", "NSCE"), ("cc", "CC"), ("bias_pct", "%bias"), ("rmse", "RMSE"))
                      if metrics.get(k) is not None)
    if skill:
        ax.text(0.01, 0.98, skill, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#999"))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_peak_map(job, gid: str, meta: dict, path: str) -> bool:
    """AQUAH 'combined_maps.png': 2-D streamflow at the simulated peak + gauge."""
    frames = job.frames.get(gid) or []
    if not frames:
        return False
    rows = job.hydro.get(gid) or []
    peak_t, peak_q = None, None
    for r in rows:
        q = r.get("sim_q")
        if q is not None and (peak_q is None or q > peak_q):
            peak_q, peak_t = q, r.get("time")
    idx = len(frames) - 1
    if peak_t:
        for i, f in enumerate(frames):
            if (f[2] or "")[:16] == peak_t[:16]:
                idx = i
                break
    png, bounds, label = frames[idx]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open(io.BytesIO(png))
    (s, w), (n, e) = bounds
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=130)
    ax.imshow(img, extent=(w, e, s, n), origin="upper")
    if meta.get("lon") and meta.get("lat"):
        ax.plot(meta["lon"], meta["lat"], "^", color="#d0342c", ms=11,
                mec="white", mew=1.4, label=f"USGS {gid}")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_facecolor("#dfe6ec")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"2-D streamflow at the simulated peak ({label or 'last step'})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _fig_basic(meta: dict, path: str) -> bool:
    """AQUAH 'basic_data.png': DEM + flow accumulation from the basin clip store."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
        from hf_data import basic
        from hf_data.pipeline import basin_bbox
        bdir = basic.store_dir(basin_bbox(meta))
        panels = [("dem_clip.tif", "DEM (m)", "terrain"),
                  ("facc_clip.tif", "Flow accumulation (log10 cells)", "Blues")]
        if not all(os.path.exists(os.path.join(bdir, f)) for f, _, _ in panels):
            return False
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=130)
        for ax, (fname, title, cmap) in zip(axes, panels):
            with rasterio.open(os.path.join(bdir, fname)) as ds:
                a = ds.read(1).astype("float64")
                a[a == ds.nodata] = np.nan
                b = ds.bounds
            if "facc" in fname:
                a = np.log10(np.clip(a, 1, None))
            im = ax.imshow(a, extent=(b.left, b.right, b.bottom, b.top),
                           origin="upper", cmap=cmap)
            if meta.get("lon") and meta.get("lat"):
                ax.plot(meta["lon"], meta["lat"], "^", color="#d0342c", ms=9,
                        mec="white", mew=1.2)
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"Basin terrain — USGS {meta.get('id')} ({meta.get('name', '')})",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return True
    except Exception:
        return False


# ---- AQUAH step 2: simulation summary ---------------------------------------
def _kge(rows: list[dict]):
    import numpy as np
    sim, obs = [], []
    for r in rows:
        s, o = r.get("sim_q"), r.get("obs_q")
        if s is None or o is None or (isinstance(o, float) and math.isnan(o)):
            continue
        sim.append(s); obs.append(o)
    if len(sim) < 2:
        return None
    sim, obs = np.asarray(sim, float), np.asarray(obs, float)
    if obs.mean() == 0 or obs.std() == 0 or sim.std() == 0:
        return None
    cc = float(np.corrcoef(sim, obs)[0, 1])
    alpha = float(sim.std() / obs.std())
    beta = float(sim.mean() / obs.mean())
    return round(1 - math.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), 3)


def _summary(meta: dict, metrics: dict, rows: list, t_start, t_end,
             params: dict | None) -> str:
    """AQUAH generate_simulation_summary, adapted to this app's metrics dict."""
    kge = _kge(rows)
    lines = [f"Hydrological Simulation Summary for USGS {meta.get('id')} "
             f"({meta.get('name', 'unknown gauge')}):", "",
             f"The simulation covers {t_start:%Y-%m-%d %H:%M} to {t_end:%Y-%m-%d %H:%M} UTC "
             f"for the {meta.get('area', '?')} km² basin draining to USGS gauge "
             f"#{meta.get('id')} at ({meta.get('lat')}, {meta.get('lon')}). "
             f"Water-balance model: {str(meta.get('model', '?')).upper()} with "
             f"kinematic-wave routing (EF5), MRMS rainfall and hourly output."]
    m = metrics or {}
    perf = []
    if m.get("nsce") is not None:
        perf.append(f"NSCE {m['nsce']}")
    if kge is not None:
        perf.append(f"KGE {kge}")
    if m.get("cc") is not None:
        perf.append(f"correlation {m['cc']}")
    if m.get("bias_pct") is not None:
        perf.append(f"volume bias {m['bias_pct']}%")
    if m.get("rmse") is not None:
        perf.append(f"RMSE {m['rmse']} m³/s")
    if perf:
        lines.append("Performance vs USGS observations: " + ", ".join(perf) + ".")
    else:
        lines.append("No overlapping USGS observations were available for scoring.")
    if m.get("peak_sim") is not None:
        peak = (f"Simulated peak discharge {m['peak_sim']} m³/s at {m.get('peak_time')}")
        if m.get("peak_obs") is not None:
            peak += f"; observed peak {m['peak_obs']} m³/s"
        lines.append(peak + ".")
    if params:
        wb = params.get("wb") or {}
        kw = params.get("kw") or {}
        ps = ", ".join(f"{k}={round(float(v), 3)}" for k, v in {**wb, **kw}.items())
        if ps:
            lines.append(f"Effective parameter multipliers ({params.get('source', 'run')}): {ps}.")
    return "\n".join(lines)


# ---- AQUAH step 1: vision analysis of the figures ---------------------------
def _vision_markdown(fig_paths: list[str], cfg: dict) -> str:
    from hf_data import llm
    if not llm.available() or not fig_paths:
        return ""
    content = []
    for p in fig_paths:
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    prompts = cfg["report_writer_prompts"]
    content.append({"type": "text",
                    "text": prompts["system_prompt"] + prompts["user_suffix"]})
    try:
        txt, _prov = llm.chat([{"role": "user", "content": content}],
                              temperature=0.5, max_tokens=1500)
        return txt.strip()
    except Exception:
        return ""       # vision-incapable provider / transient — report still works


# ---- AQUAH step 3: the report-writer agent -----------------------------------
def _agent_markdown(cfg: dict, summary: str, figures_md: str, basin_name: str,
                    figure_path: str, fig_names: list[str]) -> str:
    from hf_data import llm
    task = cfg["write_report"]["description"].format(
        summary=summary, report_for_figures_md=figures_md or "(no figure analysis)",
        basin_name=basin_name, figure_path=figure_path)
    task += ("\n\nNOTE: the ONLY figure files that exist are: "
             + ", ".join(fig_names) + " — reference exactly these (with the given "
             "figure path) and no others.\n\n" + cfg["write_report"]["expected_output"])
    a = cfg["report_writer_agent"]
    sys_p = (f"ROLE: {a['role']}\nGOAL: {a['goal']}\nBACKSTORY: {a['backstory']}")
    txt, _prov = llm.chat([{"role": "system", "content": sys_p},
                           {"role": "user", "content": task}],
                          temperature=0.7, max_tokens=3500)
    md = txt.strip()
    if md.startswith("```markdown"):
        md = md[len("```markdown"):].strip()
    if md.startswith("```"):
        md = md[3:].strip()
    if md.endswith("```"):
        md = md[:-3].strip()
    return md


def _fallback_markdown(summary: str, meta: dict, metrics: dict,
                       figure_path: str, fig_names: list[str]) -> str:
    """No LLM configured — deterministic but complete report."""
    m = metrics or {}
    rows = [("NSCE", m.get("nsce")), ("CC", m.get("cc")),
            ("Bias (%)", m.get("bias_pct")), ("RMSE (m³/s)", m.get("rmse")),
            ("Peak sim (m³/s)", m.get("peak_sim")), ("Peak obs (m³/s)", m.get("peak_obs"))]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows if v is not None)
    figs = "\n\n".join(f"![]({figure_path}/{n})" for n in fig_names)
    return (f"# Flood Simulation Report — USGS {meta.get('id')} ({meta.get('name', '')})\n\n"
            f"{summary}\n\n## Figures\n\n{figs}\n\n"
            f"## Performance metrics\n\n| Metric | Value |\n|---|---|\n{table}\n")


# ---- public entry -------------------------------------------------------------
def generate(job, gid: str) -> str:
    """Build (or reuse) the report for one finished gauge. Returns the file path
    (.pdf normally; .md when pandoc/xelatex is unavailable)."""
    key = f"{job.id}_{gid}"
    with _lock(key):
        os.makedirs(REPORT_DIR, exist_ok=True)
        pdf = os.path.join(REPORT_DIR, key + ".pdf")
        md_fallback = os.path.join(REPORT_DIR, key + ".md")
        if os.path.exists(pdf):
            return pdf
        if os.path.exists(md_fallback):        # pandoc-less fallback is cached too
            return md_fallback

        meta = job.meta.get(gid) or {"id": gid}
        rows = job.hydro.get(gid) or []
        if not rows:
            raise ValueError("no simulated rows for this gauge yet")
        from hf_data import analysis
        metrics = analysis.compute_metrics(rows)
        cfg = _config()

        work = os.path.join(REPORT_DIR, key + "_work")
        os.makedirs(work, exist_ok=True)
        figp = work.replace(os.sep, "/")
        fig_names = []
        _fig_results(rows, meta, metrics, os.path.join(work, "results.png"))
        fig_names.append("results.png")
        if _fig_peak_map(job, gid, meta, os.path.join(work, "combined_maps.png")):
            fig_names.insert(0, "combined_maps.png")
        if _fig_basic(meta, os.path.join(work, "basic_data.png")):
            fig_names.append("basic_data.png")

        summary = _summary(meta, metrics, rows, job.t_start, job.t_end,
                           job.params.get(gid))
        basin = f"{meta.get('name', gid)} (USGS {gid})"
        try:
            from hf_data import llm
            has_llm = bool(llm.available())
        except Exception:
            has_llm = False
        if has_llm:
            figures_md = _vision_markdown([os.path.join(work, n) for n in fig_names], cfg)
            md = _agent_markdown(cfg, summary, figures_md, basin, figp, fig_names)
        else:
            md = _fallback_markdown(summary, meta, metrics, figp, fig_names)

        md_path = os.path.join(work, "Hydro_Report.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(md)
        try:
            import pypandoc
            pypandoc.convert_file(
                md_path, "pdf", outputfile=pdf,
                extra_args=["--pdf-engine=xelatex",
                            "--variable", "mainfont=Latin Modern Roman",
                            "--variable", "geometry:margin=2.2cm",
                            "--resource-path", work])
            shutil.rmtree(work, ignore_errors=True)
            return pdf
        except Exception:
            # local dev / missing pandoc: hand back the Markdown (figures inlined
            # as data URIs so the single file is still self-contained)
            def _inline(match):
                p = match.group(1)
                fp = p if os.path.isabs(p) else os.path.join(work, os.path.basename(p))
                try:
                    with open(fp, "rb") as fh:
                        return (f"![](data:image/png;base64,"
                                f"{base64.b64encode(fh.read()).decode()})")
                except Exception:
                    return match.group(0)
            md_inl = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", _inline, md)
            with open(md_fallback, "w", encoding="utf-8") as fh:
                fh.write(md_inl)
            shutil.rmtree(work, ignore_errors=True)
            return md_fallback
