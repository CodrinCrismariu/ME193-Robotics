"""Build v_fit_5_iterations.xlsx: 5 iterations of gradient descent on m and b, as live Excel formulas.

Sheets
  Data           Anna's V data (x, y), copied from data/annacode-v.tsv
  ReLU neuron    y_hat = ReLU(m*u + b)  - 5 iterations
  Linear neuron  y_hat = m*u + b        - 5 iterations (no activation, for comparison)

Every number on the model sheets is a formula, so changing the learning rate or the starting
m and b (blue cells) recalculates everything. The starting values match v_fit.py, so the
summary tables must match what v_fit.py prints.

Run:  python make_excel.py
"""
from openpyxl import Workbook
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as col

from common import HERE, load_v
from v_fit import B0, LR_NEURON, M0

OUT = HERE / "v_fit_5_iterations.xlsx"
ITERATIONS = 5

FONT = "Arial"
F_BASE = Font(name=FONT, size=10)
F_BOLD = Font(name=FONT, size=10, bold=True)
F_TITLE = Font(name=FONT, size=14, bold=True)
F_NOTE = Font(name=FONT, size=9, italic=True, color="52514E")
F_INPUT = Font(name=FONT, size=10, color="0000FF")          # blue = number you can change
F_LINK = Font(name=FONT, size=10, color="008000")           # green = pulled from the Data sheet
FILL_INPUT = PatternFill("solid", fgColor="FFF2CC")
FILL_HEAD = PatternFill("solid", fgColor="E7E6E1")
FILL_BLOCK = PatternFill("solid", fgColor="DDEBF7")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)

# Layout (same on both model sheets)
R_INPUT = 4               # rows 4-9: inputs and data statistics
R_SUM_HEAD = 12           # summary table header; iteration k is on row R_SUM_HEAD + 1 + k
R_BLOCK_TITLE = 21
R_HEAD = 22
R_FIRST = 23              # first data row in the grid
C_BLOCK0 = 6              # column F: first iteration block


def sum_row(k):
    return R_SUM_HEAD + 1 + k


def put(ws, ref, value, font=F_BASE, fmt=None, fill=None, border=None, align=None):
    c = ws[ref]
    c.value = value
    c.font = font
    if fmt:
        c.number_format = fmt
    if fill:
        c.fill = fill
    if border:
        c.border = border
    if align:
        c.alignment = align
    return c


def data_sheet(wb, x, y):
    ws = wb.active
    ws.title = "Data"
    for j, h in enumerate(["point", "x", "y"], start=1):
        put(ws, f"{col(j)}1", h, F_BOLD, fill=FILL_HEAD, border=BOX, align=CENTER)
    for i, (xi, yi) in enumerate(zip(x, y)):
        r = i + 2
        put(ws, f"A{r}", i + 1)
        put(ws, f"B{r}", float(xi), fmt="0.0000")
        put(ws, f"C{r}", float(yi), fmt="0.0000")
    put(ws, "E1", "Source: Anna's Code (AnnaCode builder page), Example data -> V, saved with the Save button "
                  "as annacode-v.tsv on 2026-11-08. The page makes it with y = 2|x - 60| + noise (std 2), "
                  "120 points, x random in [20, 100].", F_NOTE)
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 11
    ws.column_dimensions["C"].width = 11
    ws.freeze_panes = "A2"
    return len(x)


def model_sheet(wb, title, n, use_relu):
    ws = wb.create_sheet(title)
    last = R_FIRST + n - 1
    if use_relu:
        put(ws, "A1", "One neuron + ReLU: 5 iterations of gradient descent on Anna's V", F_TITLE)
        put(ws, "A2", "Model: y_hat = ReLU(z),  z = m*u + b,  u = (x - mean x) / std x.   "
                      "Loss = MSE = (1/N) * sum (y - y_hat)^2.", F_NOTE)
        put(ws, "A3", "dL/dm = mean of -2 (y - y_hat) ReLU'(z) u      dL/db = mean of -2 (y - y_hat) ReLU'(z)"
                      "      m_new = m - alpha * dL/dm,   b_new = b - alpha * dL/db", F_NOTE)
        block = ["z = m*u + b", "y_hat = ReLU(z)", "ReLU'(z)", "error y - y_hat",
                 "-2 err ReLU' u", "-2 err ReLU'"]
    else:
        put(ws, "A1", "One neuron (no activation): 5 iterations of gradient descent on Anna's V", F_TITLE)
        put(ws, "A2", "Model: y_hat = m*u + b,  u = (x - mean x) / std x.   "
                      "Loss = MSE = (1/N) * sum (y - y_hat)^2.", F_NOTE)
        put(ws, "A3", "dL/dm = mean of -2 (y - y_hat) u      dL/db = mean of -2 (y - y_hat)"
                      "      m_new = m - alpha * dL/dm,   b_new = b - alpha * dL/db", F_NOTE)
        block = ["y_hat = m*u + b", "error y - y_hat", "-2 err u", "-2 err"]
    width = len(block) + 1                 # one empty spacer column between blocks

    # ── inputs and data statistics ────────────────────────────────────────────
    r = R_INPUT
    rows = [
        ("learning rate (alpha)", LR_NEURON, True, "0.00"),
        ("starting m", M0, True, "0.0000"),
        ("starting b", B0, True, "0.0000"),
        ("N (number of points)", f"=COUNT(Data!B2:B{n + 1})", False, "0"),
        ("mean of x", f"=AVERAGE(Data!B2:B{n + 1})", False, "0.0000"),
        ("std of x (STDEVP, divide by N)", f"=STDEVP(Data!B2:B{n + 1})", False, "0.0000"),
    ]
    for i, (label, val, is_input, fmt) in enumerate(rows):
        put(ws, f"A{r + i}", label, F_BOLD if is_input else F_BASE)
        put(ws, f"C{r + i}", val, F_INPUT if is_input else F_BASE, fmt=fmt,
            fill=FILL_INPUT if is_input else None, border=BOX)
    put(ws, f"E{r}", "Blue cells on yellow are inputs - change them and the whole sheet updates. "
                     "Green numbers come from the Data sheet. Everything else is a formula.", F_NOTE)
    alpha, N, xmean, xstd = f"$C${r}", f"$C${r + 3}", f"$C${r + 4}", f"$C${r + 5}"

    # ── summary table: one row per iteration ──────────────────────────────────
    heads = ["iteration", "m", "b", "MSE", "dL/dm", "dL/db"] + (["points with ReLU on"] if use_relu else [])
    for j, h in enumerate(heads, start=1):
        put(ws, f"{col(j)}{R_SUM_HEAD}", h, F_BOLD, fill=FILL_HEAD, border=BOX, align=CENTER)
    put(ws, f"A{R_SUM_HEAD - 1}", "Summary: m and b after each iteration (iteration 0 = starting guess)", F_BOLD)

    for k in range(ITERATIONS + 1):
        sr = sum_row(k)
        c0 = C_BLOCK0 + k * width                      # first column of block k
        cols = {name: col(c0 + j) for j, name in enumerate(block)}
        yhat = cols["y_hat = ReLU(z)"] if use_relu else cols["y_hat = m*u + b"]
        dm_col = col(c0 + len(block) - 2)
        db_col = col(c0 + len(block) - 1)
        put(ws, f"A{sr}", k, F_BOLD, fmt="0", border=BOX, align=CENTER)
        if k == 0:
            put(ws, f"B{sr}", f"=C{R_INPUT + 1}", fmt="0.0000", border=BOX)
            put(ws, f"C{sr}", f"=C{R_INPUT + 2}", fmt="0.0000", border=BOX)
        else:
            put(ws, f"B{sr}", f"=B{sr - 1}-{alpha}*E{sr - 1}", fmt="0.0000", border=BOX)
            put(ws, f"C{sr}", f"=C{sr - 1}-{alpha}*F{sr - 1}", fmt="0.0000", border=BOX)
        put(ws, f"D{sr}", f"=SUMXMY2($C${R_FIRST}:$C${last},{yhat}{R_FIRST}:{yhat}{last})/{N}",
            fmt="0.00", border=BOX)
        put(ws, f"E{sr}", f"=AVERAGE({dm_col}{R_FIRST}:{dm_col}{last})", fmt="0.0000", border=BOX)
        put(ws, f"F{sr}", f"=AVERAGE({db_col}{R_FIRST}:{db_col}{last})", fmt="0.0000", border=BOX)
        if use_relu:
            g = cols["ReLU'(z)"]
            put(ws, f"G{sr}", f"=SUM({g}{R_FIRST}:{g}{last})", fmt="0", border=BOX, align=CENTER)

        # ── the per-point block for iteration k ──────────────────────────────
        ws.merge_cells(start_row=R_BLOCK_TITLE, start_column=c0, end_row=R_BLOCK_TITLE,
                       end_column=c0 + len(block) - 1)
        put(ws, f"{col(c0)}{R_BLOCK_TITLE}", f"Iteration {k}: uses m and b from summary row {sr}",
            F_BOLD, fill=FILL_BLOCK, align=CENTER)
        for j, h in enumerate(block):
            put(ws, f"{col(c0 + j)}{R_HEAD}", h, F_BOLD, fill=FILL_HEAD, border=BOX, align=CENTER)
            ws.column_dimensions[col(c0 + j)].width = 12
        ws.column_dimensions[col(c0 + len(block))].width = 3
        m, b = f"$B${sr}", f"$C${sr}"
        for i in range(n):
            rr = R_FIRST + i
            u = f"$D{rr}"
            if use_relu:
                z, a, gp, err, dm, db = (f"{cols[h]}{rr}" for h in block)
                put(ws, z, f"={m}*{u}+{b}", fmt="0.000")
                put(ws, a, f"=MAX(0,{z})", fmt="0.000")
                put(ws, gp, f"=IF({z}>0,1,0)", fmt="0")
                put(ws, err, f"=$C{rr}-{a}", fmt="0.000")
                put(ws, dm, f"=-2*{err}*{gp}*{u}", fmt="0.000")
                put(ws, db, f"=-2*{err}*{gp}", fmt="0.000")
            else:
                a, err, dm, db = (f"{cols[h]}{rr}" for h in block)
                put(ws, a, f"={m}*{u}+{b}", fmt="0.000")
                put(ws, err, f"=$C{rr}-{a}", fmt="0.000")
                put(ws, dm, f"=-2*{err}*{u}", fmt="0.000")
                put(ws, db, f"=-2*{err}", fmt="0.000")

    # ── data columns of the grid ─────────────────────────────────────────────
    for j, h in enumerate(["point", "x", "y", "u = (x - mean)/std"], start=1):
        put(ws, f"{col(j)}{R_HEAD}", h, F_BOLD, fill=FILL_HEAD, border=BOX, align=CENTER)
    for i in range(n):
        rr = R_FIRST + i
        put(ws, f"A{rr}", i + 1)
        put(ws, f"B{rr}", f"=Data!B{i + 2}", F_LINK, fmt="0.0000")
        put(ws, f"C{rr}", f"=Data!C{i + 2}", F_LINK, fmt="0.0000")
        put(ws, f"D{rr}", f"=(B{rr}-{xmean})/{xstd}", fmt="0.0000")

    ws.column_dimensions["A"].width = 30
    for c in "BCDEFG":
        ws.column_dimensions[c].width = max(ws.column_dimensions[c].width or 0, 12)
    ws.row_dimensions[R_SUM_HEAD].height = 28
    ws.row_dimensions[R_HEAD].height = 28
    ws.freeze_panes = f"E{R_FIRST}"

    # ── chart: data, the starting guess, and the model after 5 iterations ────
    ch = ScatterChart()
    ch.title = "Data, starting guess, and after 5 iterations"
    ch.style = 13
    ch.x_axis.title = "x"
    ch.y_axis.title = "y"
    ch.height, ch.width = 9, 15
    xs = Reference(ws, min_col=2, min_row=R_FIRST, max_row=last)
    ycol0 = C_BLOCK0 + (1 if use_relu else 0)
    ycol5 = ycol0 + ITERATIONS * width
    for ref_col, name, color in [(3, "data y", "9E9C95"), (ycol0, "iteration 0", "EB6834"),
                                 (ycol5, "iteration 5", "2A78D6")]:
        s = Series(Reference(ws, min_col=ref_col, min_row=R_FIRST, max_row=last), xs, title=name)
        s.marker.symbol = "circle"
        s.marker.size = 4
        s.marker.graphicalProperties.solidFill = color
        s.marker.graphicalProperties.line.solidFill = color
        s.graphicalProperties.line.noFill = True
        ch.series.append(s)
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.x_axis.number_format = "0"
    ch.y_axis.number_format = "0"
    ch.x_axis.scaling.min, ch.x_axis.scaling.max = 20, 100
    ch.y_axis.scaling.min, ch.y_axis.scaling.max = -10, 90
    ws.add_chart(ch, f"I{R_INPUT - 1}")
    return ws


def main():
    x, y = load_v()
    wb = Workbook()
    n = data_sheet(wb, x, y)
    model_sheet(wb, "ReLU neuron", n, use_relu=True)
    model_sheet(wb, "Linear neuron", n, use_relu=False)
    wb.active = 1                                   # open on the ReLU sheet
    for ws in wb.worksheets:
        ws.sheet_view.tabSelected = ws is wb.active
    wb.save(OUT)
    print(f"saved {OUT.name}")


if __name__ == "__main__":
    main()
