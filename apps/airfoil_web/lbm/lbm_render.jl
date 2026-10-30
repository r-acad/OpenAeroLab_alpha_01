# =====================================================================================================================
# lbm_render.jl -- the "LBM 3D (LUDWIG)" mid-plane fields: their app definitions (normalisation, fixed colour range,
# colour map) and the video frames rendered from the solver's numeric planes (27 Oct 2026). Base only: the worker runs
# without packages. A frame is a binary PPM (P6) image that ffmpeg encodes; its labels use the 5 x 7 bitmap font below.
#
# Colour maps (the same stops as static/lbm.js): "seq" is the app's field ramp (static/app.js ramp()); "div" a blue-white-
# red map whose white is the field's `mid` value (0), mapped piecewise so that an asymmetric range keeps white at 0.
# Solid cells are grey, cells outside every level (none on a single level) light grey.
# =====================================================================================================================
module AirfoilLBMRender

using Printf

export LBM_NUM_FIELDS, lbm_field, lbm_color, render_frame, write_ppm

"""The numeric mid-plane fields of the app, from the solver's .lbmslice arrays: `src` the array; `lin(h)` the linear
normalisation (offset, factor), value = (solver value - offset) * factor, from the file header h (U = h["u_ref_m_s"],
c = h["reference_chord_m"], the pressure references); the fixed default colour range and map. Vorticity is shown
counter-clockwise positive in the view (x right, z up): omega = dw/dx - du/dz = -omega_y."""
const LBM_NUM_FIELDS = [
    (key="umag", label="|u|/U", src="velocity_magnitude", cmap="seq", range=(0.0, 1.6), mid=NaN,
     lin=h -> (0.0, 1/h["u_ref_m_s"]), units="velocity magnitude over the freestream speed"),
    (key="vort", label="vorticity wc/U", src="vorticity_y", cmap="div", range=(-30.0, 30.0), mid=0.0,
     lin=h -> (0.0, -h["reference_chord_m"]/h["u_ref_m_s"]), units="spanwise vorticity x chord / U, counter-clockwise positive in this view"),
    (key="cp", label="Cp", src="pressure", cmap="div", range=(-2.5, 1.0), mid=0.0,
     lin=h -> (h["pressure_ref_lattice"], 1/h["dynamic_pressure_ref_lattice"]), units="pressure coefficient (solver lattice pressure against the inlet state)"),
    (key="u", label="u/U", src="u", cmap="seq", range=(-0.4, 1.6), mid=NaN,
     lin=h -> (0.0, 1/h["u_ref_m_s"]), units="streamwise velocity over U"),
    (key="w", label="w/U", src="w", cmap="div", range=(-0.8, 0.8), mid=0.0,
     lin=h -> (0.0, 1/h["u_ref_m_s"]), units="vertical velocity over U (domain z, normal to the freestream)"),
    (key="v", label="v/U spanwise", src="v", cmap="div", range=(-0.25, 0.25), mid=0.0,
     lin=h -> (0.0, 1/h["u_ref_m_s"]), units="spanwise velocity over U (zero in a two-dimensional flow)"),
]
lbm_field(key::AbstractString) = LBM_NUM_FIELDS[findfirst(f -> f.key == key, LBM_NUM_FIELDS)]

const SEQ_STOPS = ((23,62,114), (52,124,187), (91,192,201), (237,212,107), (233,139,77), (155,56,77))
const DIV_STOPS = ((40,70,140), (110,155,210), (245,245,245), (225,130,100), (150,35,45))
const SOLID_RGB = (128,134,140)
const NOLEVEL_RGB = (215,220,224)

function _stops(stops, t)
    t = clamp(t, 0.0, 1.0)
    n = length(stops)
    a = min(n - 1, floor(Int, t*(n - 1)) + 1)
    k = clamp(t*(n - 1) - (a - 1), 0.0, 1.0)
    ntuple(i -> round(UInt8, stops[a][i]*(1 - k) + stops[a+1][i]*k), 3)
end
"The colour of `v` on a field's map and range: seq linear over the range; div white at `mid`, each side linear."
function lbm_color(v::Real, cmap::AbstractString, lo::Real, hi::Real, mid::Real)
    isfinite(v) || return UInt8.(NOLEVEL_RGB)
    if cmap == "div" && isfinite(mid) && lo < mid < hi
        t = v < mid ? 0.5*(v - lo)/(mid - lo) : 0.5 + 0.5*(v - mid)/(hi - mid)
        return _stops(DIV_STOPS, t)
    end
    _stops(cmap == "div" ? DIV_STOPS : SEQ_STOPS, (v - lo)/(hi - lo))
end

# ------------------------------------------------------------------------------------------------- 5 x 7 bitmap font
# Five column bytes per glyph, bit 0 the top row (the classic 5x7 LCD font; g, j, p, q, y use row 7 as a descender).
const FONT = Dict{Char,NTuple{5,UInt8}}(
    ' '=>(0x00,0x00,0x00,0x00,0x00), '!'=>(0x00,0x00,0x5F,0x00,0x00), '%'=>(0x23,0x13,0x08,0x64,0x62),
    '?'=>(0x02,0x01,0x51,0x09,0x06), '#'=>(0x14,0x7F,0x14,0x7F,0x14),
    '('=>(0x00,0x1C,0x22,0x41,0x00), ')'=>(0x00,0x41,0x22,0x1C,0x00), '*'=>(0x14,0x08,0x3E,0x08,0x14),
    '+'=>(0x08,0x08,0x3E,0x08,0x08), ','=>(0x00,0x50,0x30,0x00,0x00), '-'=>(0x08,0x08,0x08,0x08,0x08),
    '.'=>(0x00,0x60,0x60,0x00,0x00), '/'=>(0x20,0x10,0x08,0x04,0x02), ':'=>(0x00,0x36,0x36,0x00,0x00),
    ';'=>(0x00,0x56,0x36,0x00,0x00), '<'=>(0x08,0x14,0x22,0x41,0x00), '='=>(0x14,0x14,0x14,0x14,0x14),
    '>'=>(0x00,0x41,0x22,0x14,0x08), '['=>(0x00,0x7F,0x41,0x41,0x00), ']'=>(0x00,0x41,0x41,0x7F,0x00),
    '_'=>(0x40,0x40,0x40,0x40,0x40), '|'=>(0x00,0x00,0x7F,0x00,0x00), '\''=>(0x00,0x05,0x03,0x00,0x00),
    '0'=>(0x3E,0x51,0x49,0x45,0x3E), '1'=>(0x00,0x42,0x7F,0x40,0x00), '2'=>(0x42,0x61,0x51,0x49,0x46),
    '3'=>(0x21,0x41,0x45,0x4B,0x31), '4'=>(0x18,0x14,0x12,0x7F,0x10), '5'=>(0x27,0x45,0x45,0x45,0x39),
    '6'=>(0x3C,0x4A,0x49,0x49,0x30), '7'=>(0x01,0x71,0x09,0x05,0x03), '8'=>(0x36,0x49,0x49,0x49,0x36),
    '9'=>(0x06,0x49,0x49,0x29,0x1E),
    'A'=>(0x7E,0x11,0x11,0x11,0x7E), 'B'=>(0x7F,0x49,0x49,0x49,0x36), 'C'=>(0x3E,0x41,0x41,0x41,0x22),
    'D'=>(0x7F,0x41,0x41,0x22,0x1C), 'E'=>(0x7F,0x49,0x49,0x49,0x41), 'F'=>(0x7F,0x09,0x09,0x09,0x01),
    'G'=>(0x3E,0x41,0x49,0x49,0x7A), 'H'=>(0x7F,0x08,0x08,0x08,0x7F), 'I'=>(0x00,0x41,0x7F,0x41,0x00),
    'J'=>(0x20,0x40,0x41,0x3F,0x01), 'K'=>(0x7F,0x08,0x14,0x22,0x41), 'L'=>(0x7F,0x40,0x40,0x40,0x40),
    'M'=>(0x7F,0x02,0x0C,0x02,0x7F), 'N'=>(0x7F,0x04,0x08,0x10,0x7F), 'O'=>(0x3E,0x41,0x41,0x41,0x3E),
    'P'=>(0x7F,0x09,0x09,0x09,0x06), 'Q'=>(0x3E,0x41,0x51,0x21,0x5E), 'R'=>(0x7F,0x09,0x19,0x29,0x46),
    'S'=>(0x46,0x49,0x49,0x49,0x31), 'T'=>(0x01,0x01,0x7F,0x01,0x01), 'U'=>(0x3F,0x40,0x40,0x40,0x3F),
    'V'=>(0x1F,0x20,0x40,0x20,0x1F), 'W'=>(0x3F,0x40,0x38,0x40,0x3F), 'X'=>(0x63,0x14,0x08,0x14,0x63),
    'Y'=>(0x07,0x08,0x70,0x08,0x07), 'Z'=>(0x61,0x51,0x49,0x45,0x43),
    'a'=>(0x20,0x54,0x54,0x54,0x78), 'b'=>(0x7F,0x48,0x44,0x44,0x38), 'c'=>(0x38,0x44,0x44,0x44,0x20),
    'd'=>(0x38,0x44,0x44,0x48,0x7F), 'e'=>(0x38,0x54,0x54,0x54,0x18), 'f'=>(0x08,0x7E,0x09,0x01,0x02),
    'g'=>(0x98,0xA4,0xA4,0xA4,0x7C), 'h'=>(0x7F,0x08,0x04,0x04,0x78), 'i'=>(0x00,0x44,0x7D,0x40,0x00),
    'j'=>(0x40,0x80,0x84,0x7D,0x00), 'k'=>(0x7F,0x10,0x28,0x44,0x00), 'l'=>(0x00,0x41,0x7F,0x40,0x00),
    'm'=>(0x7C,0x04,0x18,0x04,0x78), 'n'=>(0x7C,0x08,0x04,0x04,0x78), 'o'=>(0x38,0x44,0x44,0x44,0x38),
    'p'=>(0xFC,0x24,0x24,0x24,0x18), 'q'=>(0x18,0x24,0x24,0x24,0xFC), 'r'=>(0x7C,0x08,0x04,0x04,0x08),
    's'=>(0x48,0x54,0x54,0x54,0x20), 't'=>(0x04,0x3F,0x44,0x40,0x20), 'u'=>(0x3C,0x40,0x40,0x20,0x7C),
    'v'=>(0x1C,0x20,0x40,0x20,0x1C), 'w'=>(0x3C,0x40,0x30,0x40,0x3C), 'x'=>(0x44,0x28,0x10,0x28,0x44),
    'y'=>(0x9C,0xA0,0xA0,0xA0,0x7C), 'z'=>(0x44,0x64,0x54,0x4C,0x44))

"Draw `text` into the RGB image `img` (3 x W x H) with its top-left corner at (x, y), each font pixel `s` x `s`."
function draw_text!(img::Array{UInt8,3}, text::AbstractString, x::Integer, y::Integer, s::Integer, rgb)
    W, H = size(img, 2), size(img, 3)
    cx = x
    for ch in text
        g = get(FONT, ch, FONT['?'])
        for col in 1:5, row in 0:7
            (g[col] >> row) & 0x01 == 0x01 || continue
            for dy in 0:s-1, dx in 0:s-1
                px = cx + (col - 1)*s + dx; py = y + row*s + dy
                (1 <= px <= W && 1 <= py <= H) || continue
                img[1, px, py] = rgb[1]; img[2, px, py] = rgb[2]; img[3, px, py] = rgb[3]
            end
        end
        cx += 6s
    end
    cx
end
text_width(text::AbstractString, s::Integer) = 6s*length(text) - s

_fmt(v) = abs(v) >= 100 || v == round(v) ? @sprintf("%g", v) : @sprintf("%.2g", v)

"""One video frame: the field `vals` (n1 x n2, i along x, k along z; NaN where `mask` is not fluid) at `scale` pixels per
cell, z up, with a title line above and a colour bar below. `mask` 0 fluid, 1 solid, 2 no level. Returns 3 x W x H."""
function render_frame(vals::AbstractMatrix, mask::AbstractMatrix{UInt8}, fld, title::AbstractString, subtitle::AbstractString;
                      scale::Integer=1, lo::Real=fld.range[1], hi::Real=fld.range[2], levels=nothing)
    n1, n2 = size(vals)
    s = max(1, scale)
    fs = max(2, round(Int, 2 + n1*s/1200))         # font pixel size
    top = 21fs                                      # two text lines of 8 font rows
    bar = 15fs
    W = n1*s; H = top + n2*s + bar
    W += isodd(W); H += isodd(H)                    # yuv420p needs even dimensions
    img = fill(0xff, 3, W, H)
    # several levels: the nodes on both sides of a change of the supplying level are darkened (the level boundaries)
    edge = falses(n1, n2)
    if levels !== nothing && size(levels) == (n1, n2) && length(unique(filter(!iszero, vec(levels)))) > 1
        @inbounds for k in 1:n2, i in 1:n1
            v = levels[i, k]
            if i < n1 && levels[i+1, k] != v; edge[i, k] = true; edge[i+1, k] = true; end
            if k < n2 && levels[i, k+1] != v; edge[i, k] = true; edge[i, k+1] = true; end
        end
    end
    @inbounds for k in 1:n2, i in 1:n1
        m = mask[i, k]
        rgb = m == 0x01 ? UInt8.(SOLID_RGB) : m == 0x02 ? UInt8.(NOLEVEL_RGB) : lbm_color(vals[i, k], fld.cmap, lo, hi, fld.mid)
        (edge[i, k] && m != 0x01) && (rgb = map(c -> round(UInt8, 0.35c + 12), rgb))
        y0 = top + (n2 - k)*s
        x0 = (i - 1)*s
        for dy in 1:s, dx in 1:s
            img[1, x0+dx, y0+dy] = rgb[1]; img[2, x0+dx, y0+dy] = rgb[2]; img[3, x0+dx, y0+dy] = rgb[3]
        end
    end
    ink = (0x1f, 0x33, 0x45)
    draw_text!(img, title, 2fs, fs, fs, ink)
    draw_text!(img, subtitle, 2fs, 11fs, fs, (0x4c, 0x63, 0x73))
    # the colour bar: 40% of the width, labelled at its ends (and at the white of a diverging map)
    bw = max(100, round(Int, 0.4W)); bx = 2fs; by = top + n2*s + 2fs; bh = 4fs
    for px in 0:bw-1
        t = px/(bw - 1); v = lo + t*(hi - lo)
        rgb = lbm_color(v, fld.cmap, lo, hi, fld.mid)
        for py in 0:bh-1
            (bx + px <= W && by + py <= H) || continue
            img[1, bx+px, by+py] = rgb[1]; img[2, bx+px, by+py] = rgb[2]; img[3, bx+px, by+py] = rgb[3]
        end
    end
    ly = by + bh + fs
    draw_text!(img, _fmt(lo), bx, ly, fs, ink)
    hs = _fmt(hi); draw_text!(img, hs, bx + bw - text_width(hs, fs), ly, fs, ink)
    if fld.cmap == "div" && isfinite(fld.mid) && lo < fld.mid < hi
        ms = _fmt(fld.mid); mx = bx + round(Int, 0.5*(bw - 1)) - text_width(ms, fs) ÷ 2
        draw_text!(img, ms, mx, ly, fs, ink)
    end
    draw_text!(img, fld.label * (any(edge) ? "  (fixed scale; grey: solid; dark lines: level boundaries)" : "  (fixed scale; grey: solid)"), bx + bw + 3fs, by, fs, ink)
    img
end

"Write a 3 x W x H RGB image as a binary PPM (P6)."
function write_ppm(path::AbstractString, img::Array{UInt8,3})
    open(path, "w") do io
        write(io, "P6\n$(size(img, 2)) $(size(img, 3))\n255\n")
        write(io, img)
    end
    path
end

end # module AirfoilLBMRender
