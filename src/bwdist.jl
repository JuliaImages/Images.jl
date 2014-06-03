using Base.Cartesian

# A function to permute a tuple of subscripts given a
# permutation vector
function permutesubs(subs::(Int...),perm::Vector{Int})
    n = length(subs)
    ntuple(n,x->subs[perm[x]])
end

# A function to both permute the dimensions of an array and
# those of the linear indices stored in that array
function permutedimsubs{N}(F::Array{Int,N},perm::Vector{Int})
    B = permutedims(F,perm)
    for i in 1:length(B)
        B[i] = B[i] == 0 ? 0 : sub2ind(size(B),permutesubs(ind2sub(size(F),B[i]),perm)...)
    end
    return B
end

# A Cartesian macro that we'll need
# We need to be able to generate a vector
# the same way ntuple works
macro nvect(N,ex)
    _nvect(N,ex)
end

function _nvect(N::Int,ex)
    vars = [Base.Cartesian.inlineanonymous(ex,i) for i = 1:N]
    Expr(:escape, Expr(:vcat,vars...))
end

# Relevant distance functions copied from Distance.jl
function get_common_len(a::AbstractVector, b::AbstractVector)
    n = length(a)
    length(b) == n || throw(DimensionMismatch("The lengths of a and b must match."))
    return n
end

function sumsqdiff(a::AbstractVector, b::AbstractVector)
    n = get_common_len(a, b)::Int
    s = 0.
    for i = 1:n
        @inbounds s += abs2(a[i] - b[i])
    end
    s
end

sqeuclidean(a::AbstractVector, b::AbstractVector) = sumsqdiff(a,b)
euclidean(a::AbstractVector,b::AbstractVector) = sqrt(sumsqdiff(a,b))

# Be careful with this: you really need an Array of 0s and 1s
bwdist(I::AbstractArray{Real}) = bwdist(convert(Bool,I))
# bwdist: call this one!
@ngenerate N (Array{Int},Array{Int}) function bwdist{N}(I::AbstractArray{Bool,N})
    F = zeros(Int,size(I))
    _computeft!(F,I)
    d = [1:N]
    for i in d
        _voronoift!(F,I)
        F = permutedimsubs(F,circshift(d,1))
    end
    D = zeros(Int,size(F))
    @nexprs N d->(stride_d = strides(F)[d])
    @nloops N i F begin
        x = 1
        (@nexprs N j->(x+=(i_j-1)*stride_j))
        (@nref N D i) = sqeuclidean((@nref N F i),x,size(F))
    end
    return (F,D)
end

# Generate F_0 in the parlance of Maurer et al. 2003
@ngenerate N typeof(F) function _computeft!{N}(F::Array{Int,N},I::AbstractArray{Bool,N})
    @nexprs N d->(stride_d = strides(F)[d])
    @nloops N i I begin
        ind = 1
        (@nref N F i) = (@nref N I i) ? (@nexprs N d->(ind+=(i_d-1)*stride_d)) : 0
    end
end

# Compute the partial Voronoi diagram along the first dimension of F 
# following Maurer et al. 2003
@ngenerate N typeof(F) function _voronoift!{N}(F::Array{Int,N},I::AbstractArray{Bool,N})
    @nexprs N d->(stride_d = strides(F)[d])
    @nloops N d j->(j==1?0:1:size(F,j)) begin
        Fstar = (@nref N F j->(j==1?(:):d_j))
        l = 0
        g = zeros(Int,length(Fstar)+1)
        for i in 1:size(Fstar,1)
            f = Fstar[i]
            if f != 0
                if l < 2
                    l += 1
                    g[l] = f
                else
                    while l>=2 && removeft2(g[l-1],g[l],f,d_2,size(F))
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
            for d_1 in 1:length(Fstar)
                # This makes a vector x containing the appropriate subscripts
                # x = @nvect N (j->(j==1)?i:d_j)
                # We now want it to make x a linear index in F; (i,d_2,d_3...)
                # sub2ind(size(F),(@ntuple N j->(j==1?i:d_j))...)
                # Or the Cartesianier way:
                x = 1
                (@nexprs N j->(x+=(d_j-1)*stride_j))
                while l<n_s && (euclidean(x,g[l],size(F))>euclidean(x,g[l+1],size(F)))
                    l += 1
                end
                (@nref N F d)= g[l]
            end
        end
    end
end

# Calculate whether we should remove a feature pixel from the Voronoi diagram
function removeft2(u::Int,v::Int,w::Int,r::Int,dims)
    u1 = ind2sub(dims,u)[1]
    v1 = ind2sub(dims,v)[1]
    w1 = ind2sub(dims,w)[1]
    a = v1-u1
    b = w1-v1
    c = a+b
    c*distance2(v,r,dims)-b*distance2(u,r,dims)-a*distance2(w,r,dims)-a*b*c > 0
end

# Squared Euclidean distance from a linear index to a row index given by i
function distance2(u::Int,r::Int,dims)
    u2 = ind2sub(dims,u)[2]
    (u2 - r)^2
end

# Squared Euclidean Distance between linear indices
function sqeuclidean(x::Int,g::Int,dims)
    x_subs = ind2sub(dims,x)
    g_subs = ind2sub(dims,g)
    s = 0
    for i in 1:length(dims)
        s += (x_subs[i]-g_subs[i])^2
    end
    return s
end

# Euclidean distance between linear indices
euclidean(x::Int,g::Int,dims) = sqrt(sqeuclidean(x,g,dims))


