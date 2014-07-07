import Base.push!  # for DisjointMinSets

label_components(A, connectivity = 1:ndims(A), bkg = 0) = label_components!(Array(Int, size(A)), A, connectivity, bkg)

#### 4-connectivity in 2d, 6-connectivity in 3d, etc.
# But in fact you can choose which dimensions are connected
let _label_components_cache = Dict{(Int, Vector{Int}), Function}()
global label_components!
function label_components!(Albl::Array{Int}, A::Array, region::Union(Dims, AbstractVector{Int}), bkg = 0)
    uregion = unique(region)
    if isempty(uregion)
        # Each pixel is its own component
        k = 0
        for i = 1:length(A)
            if A[i] != bkg
                k += 1
                Albl[i] = k
            else
                Albl[i] = 0
            end
        end
        return Albl
    end
    # We're going to compile a version specifically for the chosen region. This should make it very fast.
    key = (ndims(A), uregion)
    if !haskey(_label_components_cache, key)
        # Need to generate the function
        N = length(uregion)
        ND = ndims(A)
        iregion = [symbol(string("i_",d)) for d in uregion]
        f! = eval(quote
            local lc!
            function lc!(Albl::Array{Int}, sets, A::Array, bkg)
                offsets = strides(A)[$uregion]
                @nexprs $N d->(offsets_d = offsets[d])
                k = 0
                @nloops $ND i A begin
                    k += 1
                    val = A[k]
                    label = typemax(Int)
                    if val != bkg
                        @nexprs $N d->begin
                            if $iregion[d] > 1  # is this neighbor in-bounds?
                                if A[k-offsets_d] == val  # if the two have the same value...
                                    newlabel = Albl[k-offsets_d]
                                    if label != typemax(Int) && label != newlabel
                                        label = union!(sets, label, newlabel)  # ...merge labels...
                                    else
                                        label = newlabel  # ...and assign the smaller to current pixel
                                    end
                                end
                            end
                        end
                        if label == typemax(Int)
                            label = push!(sets)   # there were no neighbors, create a new label
                        end
                    end
                    Albl[k] = label
                end
                Albl
            end
        end)
        _label_components_cache[key] = f!
    else
        f! = _label_components_cache[key]
    end
    sets = DisjointMinSets()
    f!(Albl, sets, A, bkg)
    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(A)
        Albl[i] = (A[i] == bkg) ? 0 : newlabel[find_root!(sets, Albl[i])]
    end
    Albl
end
end # let
label_components!(Albl::Array{Int}, A::BitArray, region::Union(Dims, AbstractVector{Int}), bkg = 0) = label_components!(Albl, convert(Array{Bool}, A), region, bkg)

#### Arbitrary connectivity
for N = 1:4
    @eval begin
        function label_components!(Albl::Array{Int,$N}, A::AbstractArray, connectivity::Array{Bool,$N}, bkg = 0)
            if isempty(connectivity) || !any(connectivity)
                # Each pixel is its own component
                k = 0
                for i = 1:length(A)
                    if A[i] != bkg
                        k += 1
                        Albl[i] = k
                    else
                        Albl[i] = 0
                    end
                end
                return Albl
            end
            for d = 1:ndims(connectivity)
                (isodd(size(connectivity, d)) && connectivity == flipdim(connectivity, d)) || error("connectivity must be symmetric")
            end
            size(Albl) == size(A) || throw(DimensionMismatch("size(Albl) must be equal to size(A)"))
            @nexprs $N d->(halfwidth_d = size(connectivity,d)>>1)
            @nexprs $N d->(offset_d = 1+halfwidth_d)
            cstride_1 = 1
            @nexprs $N d->(cstride_{d+1} = cstride_d * size(connectivity,d))
            @nexprs $N d->(k_d = 0)
            sets = DisjointMinSets()
            @nloops $N i A begin
                val = @nref($N, A, i)
                label = typemax(Int)
                if val != bkg
                    @nloops($N, j,
                            d->(max(-halfwidth_d, 1-i_d):min(halfwidth_d, size(A,d)-i_d)),  # loop range
                            d->(k_{d-1} = k_d + j_d*cstride_d; # break if linear index is >= 0
                                d > 1 ? (if k_{d-1} > 0 break end) : (if k_0 >= 0 break end);
                                jj_d = j_d+offset_d; ii_d = i_d+j_d),    # pre-expression
                            begin
                        if @nref($N, connectivity, jj)
                            if @nref($N, A, ii) == val
                                newlabel = @nref($N, Albl, ii)
                                if label != typemax(Int) && label != newlabel
                                    label = union!(sets, label, newlabel)
                                else
                                    label = newlabel
                                end
                            end
                        end
                    end)
                    if label == typemax(Int)
                        label = push!(sets)
                    end
                end
                @nref($N, Albl, i) = label
            end
            # Now parse sets to find the labels
            newlabel = minlabel(sets)
            for i = 1:length(A)
                Albl[i] = (A[i] == bkg) ? 0 : newlabel[find_root!(sets, Albl[i])]
            end
            Albl
        end
        label_components!(Albl::Array{Int,$N}, A::AbstractArray, connectivity::BitArray{$N}, bkg = 0) =
            label_components!(Albl, A, convert(Array{Bool}, connectivity), bkg)
    end
end

# Copied directly from DataStructures.jl, but specialized
# to always make the parent be the smallest label
immutable DisjointMinSets
    parents::Vector{Int}
    
    DisjointMinSets(n::Integer) = new([1:n])
end
DisjointMinSets() = DisjointMinSets(0)

function find_root!(sets::DisjointMinSets, m::Integer)
    p = sets.parents[m]   # don't use @inbounds here, it might not be safe
    @inbounds if sets.parents[p] != p
        sets.parents[m] = p = find_root_unsafe!(sets, p)
    end
    p
end

# an unsafe variant of the above
function find_root_unsafe!(sets::DisjointMinSets, m::Int)
    @inbounds p = sets.parents[m]
    @inbounds if sets.parents[p] != p
        sets.parents[m] = p = find_root_unsafe!(sets, p)
    end
    p
end

function union!(sets::DisjointMinSets, m::Integer, n::Integer)
    mp = find_root!(sets, m)
    np = find_root!(sets, n)
    if mp < np
        sets.parents[np] = mp
        return mp
    elseif np < mp
        sets.parents[mp] = np
        return np
    end
    mp
end

function push!(sets::DisjointMinSets)
    m = length(sets.parents) + 1
    push!(sets.parents, m)
    m
end

function minlabel(sets::DisjointMinSets)
    out = Array(Int, length(sets.parents))
    k = 0
    for i = 1:length(sets.parents)
        if sets.parents[i] == i
            k += 1
        end
        out[i] = k
    end
    out
end
