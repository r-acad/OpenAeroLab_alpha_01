# Opt-in exact UInt32 admission reductions; default 0 retains the original reduction.
# Exact tested UInt32 tile kernels from bench_species_reduction_tiles.jl cffe165d7e... .
# Per-cache immutable execution choice: baseline 0 or tile 1024. No checkpoint/config tag changes.
_validate_reduction_tile(tile::Int)=tile in (0,1024) ? nothing : throw(ArgumentError("reduction_tile must be 0 or 1024"))
# Exact driver dispatch parser; execution-only option, never part of physical configuration.
function _species_reduction_setting(environment,conservative::Bool,admission::Symbol)
    raw=get(environment,"TOPO_SPECIES_REDUCTION_TILE","0")
    raw in ("0","1024") || throw(ArgumentError("TOPO_SPECIES_REDUCTION_TILE must be 0 or 1024"))
    tile=raw=="0" ? 0 : 1024
    tile!=0 && !(conservative && admission===:deferred) &&
        throw(ArgumentError("TOPO_SPECIES_REDUCTION_TILE=1024 requires conservative deferred admission"))
    tile
end
function _build_reduction_workspace(s,tile::Int)
    _validate_reduction_tile(tile)
    tile==0 && return nothing
    # Pressure/status are ghost-sized; reconstruction/clock/dtf are at most these or interior-sized.
    nmax=max(length(s.rho),s.NI*s.NJ,length(s.dtf),length(s.outlet_status))
    ntiles=Int32(max(1,cld(nmax,tile)))
    partial=similar(s.rho,UInt32,Int(ntiles),ADMISSION_SLOTS)
    (;tile=Int32(tile),ntiles,partial)
end
_reduction_tile(c)=c.banks.reduction===nothing ? Int32(0) : c.banks.reduction.tile
# Pure key is CPU-testable. The graph executor owns a const cache binding, so it cannot silently
# retarget a graph to another cache; this key also makes execution mode explicit.
_species_execution_key(c,cfl::Float32,wallmode::Int32)=(cfl,wallmode,LEDGER_ENABLED[],CLOCK_MARGIN[],_reduction_tile(c))

@kernel function _k_tile_partial!(partial,@Const(sbank),@Const(cbank),@Const(dtf),@Const(p),
        @Const(outlet),@Const(rstat),@Const(spec),ns::Int32,nc::Int32,ndtf::Int32,np::Int32,
        nout::Int32,nr::Int32,ntiles::Int32,tile_items::Int32)
    li=@index(Local,Linear); grp=@index(Group,Linear)
    buf=@localmem UInt32 (_BATCH_WG,)
    @inbounds begin
        tile=(grp-1)%ntiles+1; slot=(grp-1)÷ntiles+1
        kind=spec[2,slot]; slice=spec[3,slot]
        v=_batch_ismax(kind) ? 0x00000000 : 0xffffffff
        i=Int32((tile-1)*tile_items+li); stop=Int32(tile*tile_items)
        if kind==_BANK_KIND_STATUS
            off=(slice-Int32(1))*ns
            while i<=min(ns,stop); v=max(v,sbank[off+i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_CLOCK
            off=(slice-Int32(1))*nc
            while i<=min(nc,stop); v=min(v,_bank_cap_bits(cbank[off+i])); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_DTF
            while i<=min(ndtf,stop); v=min(v,_bank_cap_bits(dtf[i])); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_PRESSURE
            while i<=min(np,stop); v=min(v,_bank_cap_bits(p[i])); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_OUTLET
            while i<=min(nout,stop); v=max(v,outlet[i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=Int32(_BATCH_WG); end
        else
            while i<=min(nr,stop); v=max(v,rstat[i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=Int32(_BATCH_WG); end
        end
        buf[li]=v
    end
    @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=128 && (buf[li]=_batch_ismax(spec[2,(grp-1)÷ntiles+1]) ? max(buf[li],buf[li+128]) : min(buf[li],buf[li+128])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=64 && (buf[li]=_batch_ismax(spec[2,(grp-1)÷ntiles+1]) ? max(buf[li],buf[li+64]) : min(buf[li],buf[li+64])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=32 && (buf[li]=_batch_ismax(spec[2,(grp-1)÷ntiles+1]) ? max(buf[li],buf[li+32]) : min(buf[li],buf[li+32])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=16 && (buf[li]=_batch_ismax(spec[2,(grp-1)÷ntiles+1]) ? max(buf[li],buf[li+16]) : min(buf[li],buf[li+16])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=8 && (buf[li]=_batch_ismax(spec[2,(grp-1)÷ntiles+1]) ? max(buf[li],buf[li+8]) : min(buf[li],buf[li+8])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=4 && (buf[li]=_batch_ismax(spec[2,(grp-1)÷ntiles+1]) ? max(buf[li],buf[li+4]) : min(buf[li],buf[li+4])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=2 && (buf[li]=_batch_ismax(spec[2,(grp-1)÷ntiles+1]) ? max(buf[li],buf[li+2]) : min(buf[li],buf[li+2])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear)
    @inbounds if li==1
        tile=(grp-1)%ntiles+1; slot=(grp-1)÷ntiles+1
        partial[tile,slot]=_batch_ismax(spec[2,slot]) ? max(buf[1],buf[2]) : min(buf[1],buf[2])
    end
end

@kernel function _k_tile_finish!(A,@Const(partial),@Const(spec),ntiles::Int32)
    li=@index(Local,Linear); grp=@index(Group,Linear)
    buf=@localmem UInt32 (_BATCH_WG,)
    @inbounds begin
        high=_batch_ismax(spec[2,grp]); v=high ? 0x00000000 : 0xffffffff; i=Int32(li)
        while i<=ntiles; v=high ? max(v,partial[i,grp]) : min(v,partial[i,grp]); i+=Int32(_BATCH_WG); end
        buf[li]=v
    end
    @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=128 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+128]) : min(buf[li],buf[li+128])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=64 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+64]) : min(buf[li],buf[li+64])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=32 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+32]) : min(buf[li],buf[li+32])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=16 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+16]) : min(buf[li],buf[li+16])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=8 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+8]) : min(buf[li],buf[li+8])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=4 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+4]) : min(buf[li],buf[li+4])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=2 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+2]) : min(buf[li],buf[li+2])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear)
    @inbounds if li==1
        high=_batch_ismax(spec[2,grp]); id=spec[1,grp]
        A[high ? id : ADMISSION_SLOTS+id]=high ? max(buf[1],buf[2]) : min(buf[1],buf[2])
    end
end

function _slot_batch_tiled!(c,spec,rstat)
    s=c.solver;b=c.banks;w=b.reduction
    sizes=(Int32(length(s.rho)),Int32(s.NI*s.NJ),Int32(length(s.dtf)),Int32(length(c.pressure)),
        Int32(length(s.outlet_status)),Int32(length(rstat)))
    _k_tile_partial!(s.backend,_BATCH_WG)(w.partial,b.status,b.clock,s.dtf,c.pressure,s.outlet_status,rstat,spec.m,
        sizes...,w.ntiles,w.tile;ndrange=Int(w.ntiles)*Int(spec.n)*_BATCH_WG)
    _k_tile_finish!(s.backend,_BATCH_WG)(c.admission,w.partial,spec.m,w.ntiles;ndrange=Int(spec.n)*_BATCH_WG)
    nothing
end
