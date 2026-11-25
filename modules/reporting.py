"""
PDF report for one analysis run: outputs/runs/<run_id>/GeoChange_report.pdf
"""

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from config import LULC_CLASSES

INK = colors.HexColor("#14213D")
MUTED = colors.HexColor("#5C677D")
RULE = colors.HexColor("#D5DAE1")


def _table(df, col_widths=None, font=8):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
        ("FONT", (0, 1), (-1, -1), "Helvetica", font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _img(path, width):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def build_report(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    a, b = meta["labels"][0], meta["labels"][-1]
    tag = f"{a}_{b}"
    stats = run_dir / "stats"
    ov = run_dir / "overlays"
    s = meta["summary"]

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=INK, fontSize=20, alignment=0)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=INK, fontSize=13, spaceBefore=10)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=9.5, leading=13, textColor=INK)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=MUTED)

    story = [
        Paragraph("LULC Change Report · NESFIC-D-21", h1),
        Paragraph(f"<b>{meta['name']}</b> · {meta['source'].title()} · "
                  f"periods {a} ({' to '.join(meta['periods'][a])}) and "
                  f"{b} ({' to '.join(meta['periods'][b])}) · generated {meta['created']} · "
                  f"prepared by Coarchitive Pvt. Ltd.", small),
        Spacer(1, 6),
    ]

    pc = s.get("priority_counts", {})
    story.append(Paragraph(
        f"Of {s['analysed_area_km2']} sq km analysed, <b>{s['significant_change_km2']} sq km "
        f"({s['significant_change_percent']}%)</b> shows significant land-cover change between "
        f"{a} and {b}, in {s['significant_patches']} patches of at least "
        f"{s['rules']['min_patch_ha']:.2f} ha. {pc.get('High', 0)} patches are rated high priority. "
        f"A further {s.get('doubtful_change_km2', 0)} sq km of change is low-confidence and is "
        f"listed for field verification.", body))

    if s["top_transitions"]:
        t = s["top_transitions"][0]
        story.append(Paragraph(
            f"The largest transition is <b>{t['from']} → {t['to']}</b> ({t['area_km2']} sq km).", body))

    story += [Paragraph("Land use / land cover maps", h2)]
    row = [[_img(ov / f"lulc_{a}.png", 85 * mm), _img(ov / f"lulc_{b}.png", 85 * mm)],
           [Paragraph(f"LULC {a}", small), Paragraph(f"LULC {b}", small)]]
    story.append(Table(row))
    legend = " &nbsp; ".join(
        f'<font color="{v["color"]}">■</font> {v["name"]}' for v in LULC_CLASSES.values())
    story.append(Paragraph(legend, small))

    story += [Paragraph("Class area and change", h2)]
    cc = pd.read_csv(stats / f"class_change_{tag}.csv")
    cc = cc[["class_name", f"area_{a}_km2", f"area_{b}_km2", "gain_km2", "loss_km2",
             "net_km2", "net_percent"]]
    cc.columns = ["Class", f"{a} sq km", f"{b} sq km", "Gain", "Loss", "Net", "Net %"]
    story.append(_table(cc))

    story += [Paragraph(f"Cross-matrix (sq km; rows = {a}, columns = {b})", h2)]
    tm = pd.read_csv(stats / f"transition_matrix_{tag}.csv")
    tm.columns = ["From \\ To"] + list(tm.columns[1:])
    story.append(_table(tm))

    story += [PageBreak(), Paragraph("Significant change map", h2),
              _img(ov / "change.png", 120 * mm),
              Paragraph("Only changes confident in both years and forming patches above the "
                        "minimum mapping unit are shown.", small)]

    pr_path = stats / f"priority_changes_{tag}.csv"
    if pr_path.exists():
        pr = pd.read_csv(pr_path).head(20)
        pr = pr[["rank", "transition", "area_ha", "confidence", "priority", "lat", "lon",
                 "field_verification"]].fillna("")
        pr.columns = ["#", "Transition", "ha", "Conf.", "Priority", "Lat", "Lon", "Field check"]
        story += [Paragraph("Priority list (top 20)", h2), _table(pr, font=7.5)]

    acc = stats / "accuracy_v2.json"
    bench = stats / "algorithm_benchmark.csv"
    if acc.exists() or bench.exists():
        story += [Paragraph("Model accuracy", h2)]
        if acc.exists():
            m = json.loads(acc.read_text())
            story.append(Paragraph(
                f"Validation: {m['validation']}. Overall accuracy {m['overall_accuracy']:.1%}, "
                f"kappa {m['kappa']:.2f}, macro F1 {m['macro_f1']:.2f} "
                f"({m['n_samples']} pixels, {m['n_polygons']} polygons).", body))
        if bench.exists():
            story.append(_table(pd.read_csv(bench)))

    story += [Paragraph("Method", h2), Paragraph(
        "Cloud-free median composites per period (Sentinel-2 L2A with SCL masking and "
        "processing-baseline offset correction, or Landsat 8/9 C2 L2) → 14 spectral, index and "
        "texture features → gradient-boosted tree classifier, applied to both dates jointly: an "
        "automatic (Otsu) spectral change-intensity threshold separates stable from changed land, "
        "stable pixels receive one joint class → 3×3 majority filter → comparison on a common 10 m grid → significant-change filter "
        f"(confidence ≥ {s['rules']['min_confidence']:.0%} in both years, patch ≥ "
        f"{s['rules']['min_patch_pixels']} px) → priority score = transition weight × size × "
        "confidence.", body)]

    out = run_dir / "GeoChange_report.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                      topMargin=14 * mm, bottomMargin=14 * mm,
                      title="NESFIC-D-21 LULC change report", author="Coarchitive Pvt. Ltd.").build(story)
    return out
