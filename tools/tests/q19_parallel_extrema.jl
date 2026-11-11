# Compare the parallel every-step safety scan with the original serial scan,
# including corrupt states which must cause the same guard to stop a run.
using Test,OctreeQ19
const G=OctreeQ19
function serial_extrema(st)
    lo=Inf;hi=-Inf;vmax=0.0
    for (l,L) in enumerate(st.mesh.levels),o in st.levels[l].real_octs,k in 1:8
        L.kind[k,o]==G.K_FLUID || continue
        S=st.levels[l];r=Float64(S.rho[k,o]);lo=min(lo,r);hi=max(hi,r)
        vmax=max(vmax,Float64(S.u[k,1,o])^2+Float64(S.u[k,2,o])^2+Float64(S.u[k,3,o])^2)
    end
    lo,hi,sqrt(vmax)
end
@testset "Threaded stability scan preserves safety behavior" begin
    for T in (Float32,Float64),levels in (1,2)
        regions=levels==1 ? G.Region[] : [G.Region(:box,2;lo=(8.,8.,8.),hi=(24.,24.,24.))]
        spec=G.MeshSpec(domain=(32.,32.,32.),periodic=(true,true,true),levels=levels,dx_finest=1.,regions=regions)
        st=G.State(G.generate(G.Tri[],spec),G.test_case_config(spec;precision=T),T)
        G.initialise_uniform!(st,1.,.04,0,0)
        S=st.levels[end];L=st.mesh.levels[end]
        @test Threads.nthreads()==1 || length(S.chunks)>1
        slots=Tuple{Int,Int}[]
        for c in S.chunks
            found=false
            for oi in c,k in 1:8
                o=Int(S.real_octs[oi]);L.kind[k,o]==G.K_FLUID || continue
                push!(slots,(k,o));found=true;break
            end
            found || continue
        end
        @test !isempty(slots)
        k0,o0=first(slots);S.rho[k0,o0]=T(.4);S.u[k0,2,o0]=T(.3)
        k1,o1=last(slots);S.rho[k1,o1]=T(1.7);S.u[k1,3,o1]=T(.7)
        @test isequal(G.state_extrema(st),serial_extrema(st))
        for (array,index,value) in ((S.rho,(k0,o0),T(NaN)),(S.rho,(k1,o1),T(Inf)),
                                   (S.u,(k1,1,o1),T(NaN)),(S.u,(k0,2,o0),T(Inf)))
            original=array[index...];array[index...]=value
            got=G.state_extrema(st);@test isequal(got,serial_extrema(st))
            @test !all(isfinite,got)
            array[index...]=original
        end
        if levels==2
            Lc=st.mesh.levels[1];Sc=st.levels[1];covered=nothing
            for o in Sc.real_octs,k in 1:8
                if Lc.kind[k,o]==G.K_COVERED;covered=(k,Int(o));break;end
            end
            @test covered!==nothing
            k,o=covered;before=G.state_extrema(st)
            Sc.rho[k,o]=T(NaN);Sc.u[k,1,o]=T(Inf)
            @test isequal(G.state_extrema(st),before)
        end
    end
end
