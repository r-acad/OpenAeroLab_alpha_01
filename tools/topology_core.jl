# ============================================================================================
# SHARED CAVITY-TOPOLOGY CORE — the 8-connectivity flood fill on the alpha_v >= theta mask that
# defines the campaign's Gate-B "does it shed" measurement, plus a per-k-plane 3D wrapper.
#
# PROVENANCE. `analyse_topology` was lifted VERBATIM from tools/cmesh_shed_topology.jl (the certified
# 2D shedding arbiter, repaired 2026-07-20 — 5 bugs, see
# memory/project_docs/2026-07-20_TOPOLOGY_TOOL_REPAIR.md). This file is now the canonical implementation imported by
# both the 2D arbiter and the 3D side, so the two cannot drift onto different definitions.
#
# WHY THIS FILE EXISTS (2026-07-26). The 3D driver had no topology measure at all. It used a Cp-plateau
# length that was referenced to the wrong ambient (saturated — see
# memory/project_docs/2026-07-26_SUPERCAVITY_WAS_A_METRIC_ARTIFACT.md) and a near-wall-band alpha length that saturates
# on a fine wall mesh. Neither can see cloud DETACHMENT, which is the actual shedding signal, and neither
# matches how the 2D result was certified. Divergence between separately-maintained 2D and 3D
# measurement code is precisely what produced this session's two retractions, so this is the shared copy.
#
# OPEN PHYSICAL SEMANTICS. Component areas and `min_area` remain raw cell counts; the LE fallback can
# select the wrong side; and the wake cut is not connected. They are intentionally unchanged here:
# each needs a separately authorised physical rebaseline rather than a silent scorer rewrite.
# ============================================================================================

# ---- 8-connectivity flood fill on the alpha_v >= theta mask over the FOIL-region submesh ------------
# Returns: (ncomp, max_area, detached_area_not_touching_LE, LE_connected_downstream_extent_x,
#           total_area, nclouds, attached_area)
#
# ⚠ INDEX CONVENTIONS (getting these wrong silently anchors the sheet on the wrong end of the foil —
#   that was repair #5a of the 2026-07-20 arbiter repair):
#     A       ghost-sized alpha (NI+2g, NJ+2g); read as A[NIW+ii+g, jj+g]
#     xc      INTERIOR-sized (NI, NJ), NO ghost offset; read as xc[NIW+ii, jj]
#     ii_le   FOIL-RELATIVE leading-edge index in 1..NIF, i.e. argmin(xc[NIW+ii,1] for ii in 1:NIF).
#             NOT the global i index — subtract NIW if you computed it globally.
function analyse_topology(A, xc, g, NIW, NIF, NJ, theta, ii_le, theta_dense, min_area)
    ni = NIF; nj = NJ
    mask = falses(ni, nj)
    for jj in 1:nj, ii in 1:ni
        mask[ii,jj] = Float64(A[NIW+ii+g, jj+g]) >= theta   # alpha IS ghost-sized: +g is correct here
    end
    lab = zeros(Int, ni, nj); nc = 0
    stack = Tuple{Int,Int}[]
    for jj in 1:nj, ii in 1:ni
        (mask[ii,jj] && lab[ii,jj]==0) || continue
        nc += 1; push!(stack, (ii,jj)); lab[ii,jj]=nc
        while !isempty(stack)
            (ci,cj) = pop!(stack)
            for dj in -1:1, di in -1:1
                (di==0 && dj==0) && continue
                ni2 = ci+di; nj2 = cj+dj
                (1<=ni2<=ni && 1<=nj2<=nj) || continue
                if mask[ni2,nj2] && lab[ni2,nj2]==0
                    lab[ni2,nj2]=nc; push!(stack,(ni2,nj2))
                end
            end
        end
    end
    # per-component area AND peak alpha (the peak is what separates a real cloud from speckle)
    areas = zeros(Int, nc); peak = zeros(Float64, nc)
    for jj in 1:nj, ii in 1:ni
        l = lab[ii,jj]; l==0 && continue
        areas[l] += 1
        peak[l] = max(peak[l], Float64(A[NIW+ii+g, jj+g]))
    end
    # ATTACHED SHEET = the component holding the LE wall cell (ii_le from argmin(x) over the wall row).
    att = 0
    if mask[ii_le, 1]
        att = lab[ii_le, 1]
    else
        # LE wall cell momentarily below theta: take the nearest masked wall cell within +-10% chord
        wsearch = max(1, round(Int, 0.10*ni))
        for d in 1:wsearch
            for cand in (ii_le-d, ii_le+d)
                (1 <= cand <= ni) || continue
                mask[cand,1] && (att = lab[cand,1])
                att != 0 && break
            end
            att != 0 && break
        end
    end
    total_area = sum(areas)
    # DETACHED = components that are not the attached sheet AND clear both cloud filters
    # (dense-core peak >= theta_dense AND area >= min_area, which rejects the dilute speckle).
    detached = 0; nclouds = 0
    for l in 1:nc
        l == att && continue
        (areas[l] >= min_area && peak[l] >= theta_dense) || continue
        detached += areas[l]; nclouds += 1
    end
    # attached-sheet downstream extent (xc is INTERIOR-sized, no +g)
    ledx = 0.0
    if att > 0
        for jj in 1:nj, ii in 1:ni
            lab[ii,jj]==att && (ledx = max(ledx, Float64(xc[NIW+ii, jj])))
        end
    end
    (nc, nc>0 ? maximum(areas) : 0, detached, ledx, total_area, nclouds, att>0 ? areas[att] : 0)
end

"""
    analyse_topology_3d(A3, xc, g, NIW, NIF, NJ, NK, theta, ii_le, theta_dense, min_area)

Run the 2D `analyse_topology` INDEPENDENTLY on each spanwise k-plane of a ghosted 3D alpha array.

This is the discriminator the campaign actually needs: 3D cloud shedding is CELLULAR/STAGGERED, so
out-of-phase span cells cancel in any span-AVERAGED scalar and hide vigorous per-cell shedding behind a
weak span-mean mode. Per-plane topology sees each cell's detachment separately.

Returns a NamedTuple:
  `per_k`        — Vector of the 7-tuple `analyse_topology` returns, one per k-plane
  `ledx_mean`    — span-mean attached-sheet downstream extent (the honest cavity length l/c)
  `ledx_min/max` — its span extremes (a large spread = the sheet's closure is NOT span-uniform)
  `detfrac_mean` — span-mean detached_area / total_area, the Gate-B cloud-detachment fraction
  `nclouds_tot`  — total detached clouds summed over planes
  `nclouds_max`  — most clouds on any single plane
  `planes_shed`  — how many planes currently show >=1 detached cloud (cellular signature: 0 < n < NK)
"""
function analyse_topology_3d(A3, xc, g, NIW, NIF, NJ, NK, theta, ii_le, theta_dense, min_area)
    per_k = Vector{NTuple{7,Any}}(undef, NK)
    @views for k in 1:NK
        per_k[k] = analyse_topology(A3[:,:,k+g], xc, g, NIW, NIF, NJ, theta, ii_le, theta_dense, min_area)
    end
    ledx = [Float64(t[4]) for t in per_k]
    detf = [Float64(t[3])/max(Float64(t[5]), 1e-30) for t in per_k]
    ncl  = [Int(t[6]) for t in per_k]
    (per_k = per_k,
     ledx_mean = sum(ledx)/NK, ledx_min = minimum(ledx), ledx_max = maximum(ledx),
     detfrac_mean = sum(detf)/NK, detfrac_max = maximum(detf),
     nclouds_tot = sum(ncl), nclouds_max = maximum(ncl),
     planes_shed = count(>(0), ncl))
end
