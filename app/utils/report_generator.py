# ════════════════════════════════════════════════════════
#  forecastpy_backend/app/utils/report_generator.py
#
#  Genera el PDF de reporte usando ReportLab + Matplotlib.
#  Incluye: encabezado, resumen clima, gráfico de línea,
#           tabla de KPIs mensuales y totales del período.
# ════════════════════════════════════════════════════════

import io
import logging
from datetime import datetime
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla (servidor)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image as RLImage,
)

from app.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# ── Colores corporativos ──────────────────────────────────
C_AZUL   = colors.HexColor("#0A1628")
C_ACENTO = colors.HexColor("#1E6FD9")
C_CLARO  = colors.HexColor("#4FC3F7")
C_VERDE  = colors.HexColor("#26A69A")
C_AMBER  = colors.HexColor("#FFA726")
C_GRIS   = colors.HexColor("#8EA8C3")
C_BG     = colors.HexColor("#F0F4F8")
C_ESTIM  = colors.HexColor("#7C4DFF")

NOMBRES = {
    "agua_500ml": "Agua Mineral 500 ml",
    "agua_2l":    "Agua Mineral 2 L",
    "agua_5l":    "Agua Mineral 5 L",
    "agua_20l":   "Agua Mineral 20 L",
}


def generar_pdf(resultado: Dict, clima: Dict) -> str:
    """
    Genera el PDF y lo guarda en REPORTS_DIR.
    Retorna el nombre del archivo generado.
    """
    pid        = resultado["producto_id"]
    nombre_pdf = f"reporte_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    ruta_pdf   = REPORTS_DIR / nombre_pdf

    doc   = SimpleDocTemplate(
        str(ruta_pdf), pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
    )
    estilos = getSampleStyleSheet()
    story   = []

    story.append(_encabezado(pid, resultado, estilos))
    story.append(Spacer(1, 0.4*cm))
    story.append(_seccion_clima(clima, resultado["ajuste_clima_pct"], estilos))
    story.append(Spacer(1, 0.4*cm))
    story.append(_grafico_linea(resultado))
    story.append(Spacer(1, 0.4*cm))
    story.append(_tabla_kpis(resultado, estilos))
    story.append(Spacer(1, 0.4*cm))
    story.append(_resumen_totales(resultado, estilos))
    story.append(Spacer(1, 0.4*cm))
    story.append(_pie_pagina(estilos))

    doc.build(story)
    logger.info(f"PDF generado: {ruta_pdf}")
    return nombre_pdf


# ══════════════════════════════════════════════════════════
#  COMPONENTES
# ══════════════════════════════════════════════════════════

def _encabezado(pid: str, resultado: Dict, estilos) -> Table:
    nombre   = NOMBRES.get(pid, pid)
    periodo  = "próximo mes" if resultado["meses_futuro"] == 1 else "próximos 6 meses"
    fecha_gn = datetime.now().strftime("%d/%m/%Y %H:%M")

    data = [
        [Paragraph("<b>ForecastPY — Agua San José</b>",
                   ParagraphStyle("t", fontSize=18, textColor=C_ACENTO,
                                  parent=estilos["Normal"]))],
        [Paragraph(f"Reporte de Predicción — {nombre}",
                   ParagraphStyle("s", fontSize=13, textColor=C_AZUL,
                                  parent=estilos["Normal"]))],
        [Paragraph(f"Período: {periodo}  |  Generado: {fecha_gn}  |  Cordillera, PY",
                   ParagraphStyle("i", fontSize=9, textColor=C_GRIS,
                                  parent=estilos["Normal"]))],
    ]
    t = Table(data, colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("LINEBELOW",     (0, -1), (-1, -1), 1.5, C_ACENTO),
    ]))
    return t


def _seccion_clima(clima: Dict, ajuste_pct: float, estilos) -> Table:
    color_map = {"normal": C_VERDE, "alto": C_AMBER,
                 "critico": colors.HexColor("#EF5350")}
    color = color_map.get(clima.get("nivel_alerta", "normal"), C_VERDE)

    data = [
        ["🌡 CONDICIÓN CLIMÁTICA DEL MES", "", "", ""],
        [
            f"Temp. máx. promedio: {clima.get('temp_max_promedio', '—')}°C",
            f"Temp. máx. absoluta: {clima.get('temp_max_absoluta', '—')}°C",
            f"Precipitación: {clima.get('precip_acumulada_mm', '—')} mm",
            f"Ajuste demanda: +{ajuste_pct:.0f}%",
        ],
    ]
    t = Table(data, colWidths=[4.25*cm] * 4)
    t.setStyle(TableStyle([
        ("SPAN",          (0, 0), (-1, 0)),
        ("BACKGROUND",    (0, 0), (-1, 0), color),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("BACKGROUND",    (0, 1), (-1, 1), C_BG),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_GRIS),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    return t


def _grafico_linea(resultado: Dict) -> RLImage:
    """Genera el gráfico matplotlib y lo devuelve como RLImage."""
    hist    = resultado["historico"]
    meses_p = resultado["meses"]
    cants   = resultado["cantidades"]

    fechas_h = [r["fecha"]    for r in hist]
    vals_h   = [r["cantidad"] for r in hist]

    fig, ax = plt.subplots(figsize=(14, 4), dpi=110)
    fig.patch.set_facecolor("white")

    # Histórico
    ax.plot(range(len(fechas_h)), vals_h,
            color="#4FC3F7", linewidth=2, marker="o", markersize=4,
            label="Ventas históricas")

    # Estimación
    offset    = len(fechas_h)
    idx_pred  = list(range(offset - 1, offset + len(cants)))
    vals_pred = [vals_h[-1]] + cants
    ax.plot(idx_pred, vals_pred,
            color="#7C4DFF", linewidth=2, marker="D", markersize=5,
            linestyle="--", label="Estimación LSTM")
    ax.fill_between(idx_pred, vals_pred, alpha=0.10, color="#7C4DFF")
    ax.axvline(x=offset - 0.5, color="#7C4DFF",
               linewidth=1, linestyle=":", alpha=0.7)
    ax.text(offset - 0.3, max(vals_h) * 0.95,
            "→ Estimación", color="#7C4DFF", fontsize=8)

    todas   = fechas_h + meses_p
    step    = max(1, len(todas) // 12)
    indices = list(range(0, len(todas), step))
    ax.set_xticks(indices)
    ax.set_xticklabels([todas[i] for i in indices],
                       rotation=45, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))
    ax.set_ylabel("Unidades", fontsize=9)
    ax.set_title("Historial de ventas y estimación de demanda",
                 fontsize=11, pad=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=16*cm, height=5*cm)


def _gs(v: int) -> str:
    return f"Gs. {v:,.0f}".replace(",", ".")


def _tabla_kpis(resultado: Dict, estilos) -> Table:
    enc = ["Mes", "Cant. (ud.)", "Precio Venta",
           "Costo Total", "Margen", "Rentabilidad"]
    filas = [enc]
    for i, mes in enumerate(resultado["meses"]):
        filas.append([
            mes,
            f"{resultado['cantidades'][i]:,}".replace(",", "."),
            _gs(resultado["precios_venta"][i]),
            _gs(resultado["costos_total"][i]),
            _gs(resultado["margenes"][i]),
            f"{resultado['rentabilidades'][i]:.1f}%",
        ])
    t = Table(filas, colWidths=[2.5*cm, 2.5*cm, 3*cm, 3*cm, 3*cm, 3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), C_ACENTO),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG]),
        ("GRID",           (0, 0), (-1, -1), 0.5, C_GRIS),
        ("ALIGN",          (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
    ]))
    return t


def _resumen_totales(resultado: Dict, estilos) -> Table:
    total_ud   = sum(resultado["cantidades"])
    total_ing  = sum(resultado["ingresos"])
    total_cost = sum(resultado["costos_total"])
    total_marg = sum(resultado["margenes"])
    rent_prom  = (total_marg / total_ing * 100) if total_ing > 0 else 0

    data = [
        ["RESUMEN DEL PERÍODO", ""],
        ["Total unidades estimadas:", f"{total_ud:,}".replace(",", ".")],
        ["Ingreso total estimado:",   _gs(total_ing)],
        ["Costo total estimado:",     _gs(total_cost)],
        ["Margen bruto total:",       _gs(total_marg)],
        ["Rentabilidad promedio:",    f"{rent_prom:.1f}%"],
    ]
    t = Table(data, colWidths=[8*cm, 9*cm])
    t.setStyle(TableStyle([
        ("SPAN",           (0, 0), (-1, 0)),
        ("BACKGROUND",     (0, 0), (-1, 0), C_VERDE),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG]),
        ("GRID",           (0, 0), (-1, -1), 0.5, C_GRIS),
        ("ALIGN",          (1, 1), (-1, -1), "RIGHT"),
        ("FONTNAME",       (1, 1), (-1, -1), "Helvetica-Bold"),
        ("TOPPADDING",     (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
        ("LEFTPADDING",    (0, 0), (-1, -1), 10),
    ]))
    return t


def _pie_pagina(estilos) -> Paragraph:
    return Paragraph(
        f"<i>ForecastPY · Agua San José · Caacupé, Cordillera, Paraguay · "
        f"{datetime.now().strftime('%d/%m/%Y')}</i>",
        ParagraphStyle("pie", fontSize=8, textColor=C_GRIS,
                       alignment=1, parent=estilos["Normal"]),
    )