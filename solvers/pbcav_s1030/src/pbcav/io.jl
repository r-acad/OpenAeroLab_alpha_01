# =====================================================================================================================
# io.jl -- receipts, outputs and checkpoints (plan 6.1, 7). Stage P0b: the receipts are built; the cavmov_ CSVs, the
# progress line and the state checkpoint are built with the march.
#
# MESH COORDINATE RECEIPT. Printed verbatim in version 1's form (tools/cmesh_cav_movie.jl:736-741), so every CSV-only
# reader and receipt binder runs unchanged: "MESH COORDINATE RECEIPT: x_sha256=<C block X> y_sha256=<C block Y>
# [strip_x_sha256=<strip X> strip_y_sha256=<strip Y>] foil_sha256=<the section file's bytes>", the SHA-256 of each node
# array's Float64 bytes in column-major order.
# =====================================================================================================================

using .PBRestartIO

"""
    mesh_receipt_line(m::ChannelMesh, foilpath) -> String

Version 1's MESH COORDINATE RECEIPT for a generated mesh, byte for byte.
"""
function mesh_receipt_line(m::ChannelMesh, foilpath::AbstractString)
    strip = m.strip === nothing ? "" :
        string(" strip_x_sha256=", h64(m.strip.X), " strip_y_sha256=", h64(m.strip.Y))
    string("MESH COORDINATE RECEIPT: x_sha256=", h64(m.X), " y_sha256=", h64(m.Y), strip,
           " foil_sha256=", bytes2hex(sha256(read(foilpath))))
end

"""
    mesh_level_line(b::CBlocks) -> String

The nested level's receipt (version 2 only): the subdivision level and the hashes of the subdivided node arrays.
"""
function mesh_level_line(b::CBlocks)
    h = node_hashes(b)
    string("PB MESH LEVEL: level=", b.level, " x_sha256=", h[1], " y_sha256=", h[2],
           b.NS > 0 ? string(" strip_x_sha256=", h[3], " strip_y_sha256=", h[4]) : "")
end

"""
    face_census_line(m::PBMesh) -> String

One line of the mesh's face census (counts by tag, the special lists, the lines) and its geometry extremes.
"""
function face_census_line(m::PBMesh)
    c = face_census(m)
    th = _host(m.theta); vol = _host(m.vol)
    # baffles (24 Oct 2026, opt-in): their face counts only on a mesh that has one, so every other mesh's line is unchanged
    bt = _host(m.btag); nbf = count(==(BT_BAFFLE_FRONT), bt); nbb = count(==(BT_BAFFLE_BACK), bt)
    string("PB FACE CENSUS: ", join((string(k, "=", v) for (k, v) in pairs(c)), " "),
           nbf + nbb > 0 ? string(" baffle_front=", nbf, " baffle_back=", nbb) : "",
           @sprintf(" vol_min=%.6e vol_sum=%.15e theta_max=%.4f", minimum(vol), sum(vol), isempty(th) ? 0.0 : maximum(th)))
end

"Write the cavmov_ CSVs of a frame (version 1's schemas, verbatim). Stage P0b, second half."
write_cavmov_csvs(args...) = _nyi(:write_cavmov_csvs)
"The progress line (version 1's form). Stage P0b, second half."
progress_line(args...) = _nyi(:progress_line)
"Write a state checkpoint as PBRestart/1 (the arrays of PBState and the metadata of plan 3.3). Stage P0b, second half."
write_checkpoint(path, mesh, state, meta) = _nyi(:write_checkpoint)
