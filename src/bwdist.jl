using Base.Cartesian
using Distance

macro nvect(N,ex)
    _nvect(N,ex)
end

function _nvect(N::Int,ex)
    vars = [Base.Cartesian.inlineanonymous(ex,i) for i = 1:N]
    Expr(:escape, Expr(:vcat,vars...))
end

@ngenerate N (Array{Vector{Int}},Array{Int}) function bwdist{N}(I::AbstractArray{Bool,N})
    F = Array(Vector{Int},size(I))
    computeft!(F,I)
    d = [1:N]
    for i in d
        voronoift!(F,I)
        F = permutedims(F,circshift(d,1))
        for j in 1:length(F)
            F[j] = circshift(F[j],1)
        end
    end
    D = zeros(Int,size(I))
    @nloops N i F begin
        (@nref N D i) = evaluate(SqEuclidean(),(@nvect N i),(@nref N F i))
    end
    return (F,D)
end

@ngenerate N typeof(F) function computeft!{N}(F::Array{Vector{Int},N},I::AbstractArray{Bool,N})
    @nloops N i I begin
        (@nref N F i) = (@nref N I i) ? (@nvect N i) : (@nvect N j->0)
    end
end

function voronoift!(F::Array{Vector{Int}},I::AbstractArray{Bool})
    nfv = count(x->x,I)
    for d = 1:size(F,2)
        Fstar = slicedim(F,2,d)
        l = 0
        g = fill([0 for j = 1:ndims(I)],nfv+1)
        for i in 1:size(Fstar,1)
            f = Fstar[i]
            if f != [0,0]
                if l < 2
                    l += 1
                    g[l] = f
                else
                    while l>=2 && removeft2(g[l-1],g[l],f,d)
                        l -= 1
                    end
                    l += 1
                    g[l] = f
                end
            end
        end
        n_s = l
        if n_s == 0
        else            
            l = 1
            for i in 1:length(Fstar)
                x = [i,d]
                while l<n_s && (euclidean(x,g[l])>euclidean(x,g[l+1]))
                    l += 1
                end
                F[i,d] = g[l]
            end
        end
    end
end

function removeft2(u::Vector{Int},v::Vector{Int},w::Vector{Int},r::Int)
    a = v[1]-u[1]
    b = w[1]-v[1]
    c = a+b
    c*distance2(v,r)-b*distance2(u,r)-a*distance2(w,r)-a*b*c > 0
end

function distance(x1::Vector{Int},x2::Vector{Int})
    (x1[1]-x2[2])^2+(x1[1]-x2[2])^2
end    

function distance2(u::Vector{Int},i::Int)
    (u[2] - i)^2
end

