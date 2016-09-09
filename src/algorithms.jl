#### Math with images ####


(+)(img::AbstractImageDirect{Bool}, n::Bool) = img .+ n
(+)(n::Bool, img::AbstractImageDirect{Bool}) = n .+ img
(+)(img::AbstractImageDirect, n::Number) = img .+ n
(+)(img::AbstractImageDirect, n::AbstractRGB) = img .+ n
(+)(n::Number, img::AbstractImageDirect) = n .+ img
(+)(n::AbstractRGB, img::AbstractImageDirect) = n .+ img
(.+)(img::AbstractImageDirect, n::Number) = shareproperties(img, data(img).+n)
(.+)(n::Number, img::AbstractImageDirect) = shareproperties(img, data(img).+n)
if isdefined(:UniformScaling)
    (+){Timg,TA<:Number}(img::AbstractImageDirect{Timg,2}, A::UniformScaling{TA}) = shareproperties(img, data(img)+A)
    (-){Timg,TA<:Number}(img::AbstractImageDirect{Timg,2}, A::UniformScaling{TA}) = shareproperties(img, data(img)-A)
end
(+)(img::AbstractImageDirect, A::BitArray) = shareproperties(img, data(img)+A)
(+)(img::AbstractImageDirect, A::AbstractImageDirect) = shareproperties(img, data(img)+data(A))
(+)(img::AbstractImageDirect, A::AbstractArray) = shareproperties(img, data(img)+data(A))
(+){S,T}(A::Range{S}, img::AbstractImageDirect{T}) = shareproperties(img, data(A)+data(img))
(+)(A::AbstractArray, img::AbstractImageDirect) = shareproperties(img, data(A)+data(img))
(.+)(img::AbstractImageDirect, A::BitArray) = shareproperties(img, data(img).+A)
(.+)(img::AbstractImageDirect, A::AbstractArray) = shareproperties(img, data(img).+data(A))
(-)(img::AbstractImageDirect{Bool}, n::Bool) = img .- n
(-)(img::AbstractImageDirect, n::Number) = img .- n
(-)(img::AbstractImageDirect, n::AbstractRGB) = img .- n
(.-)(img::AbstractImageDirect, n::Number) = shareproperties(img, data(img).-n)
(-)(n::Bool, img::AbstractImageDirect{Bool}) = n .- img
(-)(n::Number, img::AbstractImageDirect) = n .- img
(-)(n::AbstractRGB, img::AbstractImageDirect) = n .- img
(.-)(n::Number, img::AbstractImageDirect) = shareproperties(img, n.-data(img))
(-)(img::AbstractImageDirect, A::BitArray) = shareproperties(img, data(img)-A)
(-){T}(img::AbstractImageDirect{T,2}, A::Diagonal) = shareproperties(img, data(img)-A) # fixes an ambiguity warning
(-)(img::AbstractImageDirect, A::Range) = shareproperties(img, data(img)-A)
(-)(img::AbstractImageDirect, A::AbstractImageDirect) = shareproperties(img, data(img)-data(A))
(-)(img::AbstractImageDirect, A::AbstractArray) = shareproperties(img, data(img)-data(A))
(-){S,T}(A::Range{S}, img::AbstractImageDirect{T}) = shareproperties(img, data(A)-data(img))
(-)(A::AbstractArray, img::AbstractImageDirect) = shareproperties(img, data(A)-data(img))
(-)(img::AbstractImageDirect) = shareproperties(img, -data(img))
(.-)(img::AbstractImageDirect, A::BitArray) = shareproperties(img, data(img).-A)
(.-)(img::AbstractImageDirect, A::AbstractArray) = shareproperties(img, data(img).-data(A))
(*)(img::AbstractImageDirect, n::Number) = (.*)(img, n)
(*)(n::Number, img::AbstractImageDirect) = (.*)(n, img)
(.*)(img::AbstractImageDirect, n::Number) = shareproperties(img, data(img).*n)
(.*)(n::Number, img::AbstractImageDirect) = shareproperties(img, data(img).*n)
(/)(img::AbstractImageDirect, n::Number) = shareproperties(img, data(img)/n)
(.*)(img1::AbstractImageDirect, img2::AbstractImageDirect) = shareproperties(img1, data(img1).*data(img2))
(.*)(img::AbstractImageDirect, A::BitArray) = shareproperties(img, data(img).*A)
(.*)(A::BitArray, img::AbstractImageDirect) = shareproperties(img, data(img).*A)
(.*)(img::AbstractImageDirect{Bool}, A::BitArray) = shareproperties(img, data(img).*A)
(.*)(A::BitArray, img::AbstractImageDirect{Bool}) = shareproperties(img, data(img).*A)
(.*)(img::AbstractImageDirect, A::AbstractArray) = shareproperties(img, data(img).*A)
(.*)(A::AbstractArray, img::AbstractImageDirect) = shareproperties(img, data(img).*A)
(./)(img::AbstractImageDirect, A::BitArray) = shareproperties(img, data(img)./A)  # needed to avoid ambiguity warning
(./)(img1::AbstractImageDirect, img2::AbstractImageDirect) = shareproperties(img1, data(img1)./data(img2))
(./)(img::AbstractImageDirect, A::AbstractArray) = shareproperties(img, data(img)./A)
(.^)(img::AbstractImageDirect, p::Number) = shareproperties(img, data(img).^p)
sqrt(img::AbstractImageDirect) = @compat shareproperties(img, sqrt.(data(img)))
atan2(img1::AbstractImageDirect, img2::AbstractImageDirect) =
    @compat shareproperties(img1, atan2.(data(img1),data(img2)))
hypot(img1::AbstractImageDirect, img2::AbstractImageDirect) =
    @compat shareproperties(img1, hypot.(data(img1),data(img2)))
real(img::AbstractImageDirect) = @compat shareproperties(img,real.(data(img)))
imag(img::AbstractImageDirect) = @compat shareproperties(img,imag.(data(img)))
abs(img::AbstractImageDirect) = @compat shareproperties(img,abs.(data(img)))

Compat.@dep_vectorize_2arg Gray atan2
Compat.@dep_vectorize_2arg Gray hypot

function sum(img::AbstractImageDirect, region::Union{AbstractVector,Tuple,Integer})
    f = prod(size(img)[[region...]])
    out = copyproperties(img, sum(data(img), region))
    if in(colordim(img), region)
        out["colorspace"] = "Unknown"
    end
    out
end

"""
`M = meanfinite(img, region)` calculates the mean value along the dimensions listed in `region`, ignoring any non-finite values.
"""
meanfinite{T<:Real}(A::AbstractArray{T}, region) = _meanfinite(A, T, region)
meanfinite{CT<:Colorant}(A::AbstractArray{CT}, region) = _meanfinite(A, eltype(CT), region)
function _meanfinite{T<:AbstractFloat}(A::AbstractArray, ::Type{T}, region)
    sz = Base.reduced_dims(A, region)
    K = zeros(Int, sz)
    S = zeros(eltype(A), sz)
    sumfinite!(S, K, A)
    S./K
end
_meanfinite(A::AbstractArray, ::Type, region) = mean(A, region)  # non floating-point

function meanfinite{T<:AbstractFloat}(img::AbstractImageDirect{T}, region)
    r = meanfinite(data(img), region)
    out = copyproperties(img, r)
    if in(colordim(img), region)
        out["colorspace"] = "Unknown"
    end
    out
end
meanfinite(img::AbstractImageIndexed, region) = meanfinite(convert(Image, img), region)
# Note that you have to zero S and K upon entry
@generated function sumfinite!{T,N}(S, K, A::AbstractArray{T,N})
    quote
        isempty(A) && return S, K
        @nexprs $N d->(sizeS_d = size(S,d))
        sizeA1 = size(A, 1)
        if size(S, 1) == 1 && sizeA1 > 1
            # When we are reducing along dim == 1, we can accumulate to a temporary
            @inbounds @nloops $N i d->(d>1? (1:size(A,d)) : (1:1)) d->(j_d = sizeS_d==1 ? 1 : i_d) begin
                s = @nref($N, S, j)
                k = @nref($N, K, j)
                for i_1 = 1:sizeA1
                    tmp = @nref($N, A, i)
                    if isfinite(tmp)
                        s += tmp
                        k += 1
                    end
                end
                @nref($N, S, j) = s
                @nref($N, K, j) = k
            end
        else
            # Accumulate to array storage
            @inbounds @nloops $N i A d->(j_d = sizeS_d==1 ? 1 : i_d) begin
                tmp = @nref($N, A, i)
                if isfinite(tmp)
                    @nref($N, S, j) += tmp
                    @nref($N, K, j) += 1
                end
            end
        end
        S, K
    end
end

# Entropy for grayscale (intensity) images
function _log(kind::Symbol)
    @compat if kind == :shannon
        x -> log2.(x)
    elseif kind == :nat
        x -> log.(x)
    elseif kind == :hartley
        x -> log10.(x)
    else
        throw(ArgumentError("Invalid entropy unit. (:shannon, :nat or :hartley)"))
    end
end

"""
`entropy(img, kind)` is the entropy of a grayscale image defined as -sum(p.*logb(p)).
The base b of the logarithm (a.k.a. entropy unit) is one of the following:
  `:shannon ` (log base 2, default)
  `:nat` (log base e)
  `:hartley` (log base 10)
"""
function entropy(img::AbstractArray; kind=:shannon)
    logᵦ = _log(kind)

    hist = StatsBase.fit(Histogram, vec(img), nbins=256)
    counts = hist.weights
    p = counts / length(img)
    logp = logᵦ(p)

    # take care of empty bins
    logp[Bool[isinf(v) for v in logp]] = 0

    -sum(p .* logp)
end

function entropy(img::AbstractArray{Bool}; kind=:shannon)
    logᵦ = _log(kind)

    p = sum(img) / length(img)

    (0 < p < 1) ? - p*logᵦ(p) - (1-p)*logᵦ(1-p) : zero(p)
end

entropy{C<:AbstractGray}(img::AbstractArray{C}; kind=:shannon) = entropy(raw(img), kind=kind)

# Logical operations
(.<)(img::AbstractImageDirect, n::Number) = data(img) .< n
(.>)(img::AbstractImageDirect, n::Number) = data(img) .> n
(.<)(img::AbstractImageDirect{Bool}, A::AbstractArray{Bool}) = data(img) .< A
(.<)(img::AbstractImageDirect, A::AbstractArray) = data(img) .< A
(.>)(img::AbstractImageDirect, A::AbstractArray) = data(img) .> A
(.==)(img::AbstractImageDirect, n::Number) = data(img) .== n
(.==)(img::AbstractImageDirect{Bool}, A::AbstractArray{Bool}) = data(img) .== A
(.==)(img::AbstractImageDirect, A::AbstractArray) = data(img) .== A

# functions red, green, and blue
for (funcname, fieldname) in ((:red, :r), (:green, :g), (:blue, :b))
    fieldchar = string(fieldname)[1]
    @eval begin
        function $funcname{CV<:Color}(img::AbstractArray{CV})
            T = eltype(CV)
            out = Array(T, size(img))
            for i = 1:length(img)
                out[i] = convert(RGB{T}, img[i]).$fieldname
            end
            out
        end

        function $funcname(img::AbstractArray)
            pos = search(lowercase(colorspace(img)), $fieldchar)
            pos == 0 && error("channel $fieldchar not found in colorspace $(colorspace(img))")
            sliceim(img, "color", pos)
        end
    end
end

"`r = red(img)` extracts the red channel from an RGB image `img`" red
"`g = green(img)` extracts the green channel from an RGB image `img`" green
"`b = blue(img)` extracts the blue channel from an RGB image `img`" blue

"""
`m = minfinite(A)` calculates the minimum value in `A`, ignoring any values that are not finite (Inf or NaN).
"""
function minfinite{T}(A::AbstractArray{T})
    ret = sentinel_min(T)
    for a in A
        ret = minfinite_scalar(a, ret)
    end
    ret
end
function minfinite(f, A::AbstractArray)
    ret = sentinel_min(typeof(f(first(A))))
    for a in A
        ret = minfinite_scalar(f(a), ret)
    end
    ret
end

"""
`m = maxfinite(A)` calculates the maximum value in `A`, ignoring any values that are not finite (Inf or NaN).
"""
function maxfinite{T}(A::AbstractArray{T})
    ret = sentinel_max(T)
    for a in A
        ret = maxfinite_scalar(a, ret)
    end
    ret
end
function maxfinite(f, A::AbstractArray)
    ret = sentinel_max(typeof(f(first(A))))
    for a in A
        ret = maxfinite_scalar(f(a), ret)
    end
    ret
end

"""
`m = maxabsfinite(A)` calculates the maximum absolute value in `A`, ignoring any values that are not finite (Inf or NaN).
"""
function maxabsfinite{T}(A::AbstractArray{T})
    ret = sentinel_min(typeof(abs(A[1])))
    for a in A
        ret = maxfinite_scalar(abs(a), ret)
    end
    ret
end

# Issue #232. FIXME: really should return a Gray here?
for f in (:minfinite, :maxfinite, :maxabsfinite)
    @eval $f{T}(A::AbstractArray{Gray{T}}) = $f(reinterpret(T, data(A)))
end

minfinite_scalar{T}(a::T, b::T) = isfinite(a) ? (b < a ? b : a) : b
maxfinite_scalar{T}(a::T, b::T) = isfinite(a) ? (b > a ? b : a) : b
minfinite_scalar{T<:Union{Integer,FixedPoint}}(a::T, b::T) = b < a ? b : a
maxfinite_scalar{T<:Union{Integer,FixedPoint}}(a::T, b::T) = b > a ? b : a
minfinite_scalar(a, b) = minfinite_scalar(promote(a, b)...)
maxfinite_scalar(a, b) = maxfinite_scalar(promote(a, b)...)

function minfinite_scalar{C<:AbstractRGB}(c1::C, c2::C)
    C(minfinite_scalar(c1.r, c2.r),
      minfinite_scalar(c1.g, c2.g),
      minfinite_scalar(c1.b, c2.b))
end
function maxfinite_scalar{C<:AbstractRGB}(c1::C, c2::C)
    C(maxfinite_scalar(c1.r, c2.r),
      maxfinite_scalar(c1.g, c2.g),
      maxfinite_scalar(c1.b, c2.b))
end

sentinel_min{T<:Union{Integer,FixedPoint}}(::Type{T}) = typemax(T)
sentinel_max{T<:Union{Integer,FixedPoint}}(::Type{T}) = typemin(T)
sentinel_min{T<:AbstractFloat}(::Type{T}) = convert(T, NaN)
sentinel_max{T<:AbstractFloat}(::Type{T}) = convert(T, NaN)
sentinel_min{C<:AbstractRGB}(::Type{C}) = _sentinel_min(C, eltype(C))
_sentinel_min{C<:AbstractRGB,T}(::Type{C},::Type{T}) = (s = sentinel_min(T); C(s,s,s))
sentinel_max{C<:AbstractRGB}(::Type{C}) = _sentinel_max(C, eltype(C))
_sentinel_max{C<:AbstractRGB,T}(::Type{C},::Type{T}) = (s = sentinel_max(T); C(s,s,s))
sentinel_min{C<:AbstractGray}(::Type{C}) = _sentinel_min(C, eltype(C))
_sentinel_min{C<:AbstractGray,T}(::Type{C},::Type{T}) = C(sentinel_min(T))
sentinel_max{C<:AbstractGray}(::Type{C}) = _sentinel_max(C, eltype(C))
_sentinel_max{C<:AbstractGray,T}(::Type{C},::Type{T}) = C(sentinel_max(T))


# fft & ifft
fft(img::AbstractImageDirect) = shareproperties(img, fft(data(img)))
function fft(img::AbstractImageDirect, region, args...)
    F = fft(data(img), region, args...)
    props = copy(properties(img))
    props["region"] = region
    Image(F, props)
end
fft{CV<:Colorant}(img::AbstractImageDirect{CV}) = fft(img, 1:ndims(img))
function fft{CV<:Colorant}(img::AbstractImageDirect{CV}, region, args...)
    imgr = reinterpret(eltype(CV), img)
    if ndims(imgr) > ndims(img)
        newregion = ntuple(i->region[i]+1, length(region))
    else
        newregion = ntuple(i->region[i], length(region))
    end
    F = fft(data(imgr), newregion, args...)
    props = copy(properties(imgr))
    props["region"] = newregion
    Image(F, props)
end

function ifft(img::AbstractImageDirect)
    region = get(img, "region", 1:ndims(img))
    A = ifft(data(img), region)
    props = copy(properties(img))
    haskey(props, "region") && delete!(props, "region")
    Image(A, props)
end
ifft(img::AbstractImageDirect, region, args...) = ifft(data(img), region, args...)

# average filter
"""
`kern = imaverage(filtersize)` constructs a boxcar-filter of the specified size.
"""
function imaverage(filter_size=[3,3])
    if length(filter_size) != 2
        error("wrong filter size")
    end
    m, n = filter_size
    if mod(m, 2) != 1 || mod(n, 2) != 1
        error("filter dimensions must be odd")
    end
    f = ones(Float64, m, n)/(m*n)
end

# laplacian filter kernel
"""
`kern = imlaplacian(filtersize)` returns a kernel for laplacian filtering.
"""
function imlaplacian(diagonals::AbstractString="nodiagonals")
    if diagonals == "diagonals"
        return [1.0 1.0 1.0; 1.0 -8.0 1.0; 1.0 1.0 1.0]
    elseif diagonals == "nodiagonals"
        return [0.0 1.0 0.0; 1.0 -4.0 1.0; 0.0 1.0 0.0]
    else
        error("Expected \"diagnoals\" or \"nodiagonals\" or Number, got: \"$diagonals\"")
    end
end

# more general version
function imlaplacian(alpha::Number)
    lc = alpha/(1 + alpha)
    lb = (1 - alpha)/(1 + alpha)
    lm = -4/(1 + alpha)
    return [lc lb lc; lb lm lb; lc lb lc]
end

# 2D gaussian filter kernel
"""
`kern = gaussian2d(sigma, filtersize)` returns a kernel for FIR-based Gaussian filtering.

See also: `imfilter_gaussian`.
"""
function gaussian2d(sigma::Number=0.5, filter_size=[])
    if length(filter_size) == 0
        # choose 'good' size
        m = 4*ceil(Int, sigma)+1
        n = m
    elseif length(filter_size) != 2
        error("wrong filter size")
    else
        m, n = filter_size[1], filter_size[2]
    end
    if mod(m, 2) != 1 || mod(n, 2) != 1
        error("filter dimensions must be odd")
    end
    g = Float64[exp(-(X.^2+Y.^2)/(2*sigma.^2)) for X=-floor(Int,m/2):floor(Int,m/2), Y=-floor(Int,n/2):floor(Int,n/2)]
    return g/sum(g)
end

# difference of gaussian
"""
`kern = imdog(sigma)` creates a difference-of-gaussians kernel (`sigma`s differing by a factor of
`sqrt(2)`).
"""
function imdog(sigma::Number=0.5)
    m = 4*ceil(sqrt(2)*sigma)+1
    return gaussian2d(sqrt(2)*sigma, [m m]) - gaussian2d(sigma, [m m])
end

# laplacian of gaussian
"""
`kern = imlog(sigma)` returns a laplacian-of-gaussian kernel.
"""
function imlog(sigma::Number=0.5)
    m = ceil(8.5sigma)
    m = m % 2 == 0 ? m + 1 : m
    return [(1/(2pi*sigma^4))*(2 - (x^2 + y^2)/sigma^2)*exp(-(x^2 + y^2)/(2sigma^2))
            for x=-floor(m/2):floor(m/2), y=-floor(m/2):floor(m/2)]
end

# Sum of squared differences and sum of absolute differences
for (f, op) in ((:ssd, :(abs2(x))), (:sad, :(abs(x))))
    @eval begin
        function ($f)(A::AbstractArray, B::AbstractArray)
            size(A) == size(B) || throw(DimensionMismatch("A and B must have the same size"))
            T = promote_type(difftype(eltype(A)), difftype(eltype(B)))
            s = zero(accum(eltype(T)))
            for i = 1:length(A)
                x = convert(T, A[i]) - convert(T, B[i])
                s += $op
            end
            s
        end
    end
end

"`s = ssd(A, B)` computes the sum-of-squared differences over arrays/images A and B" ssd
"`s = sad(A, B)` computes the sum-of-absolute differences over arrays/images A and B" sad

difftype{T<:Integer}(::Type{T}) = Int
difftype{T<:Real}(::Type{T}) = Float32
difftype(::Type{Float64}) = Float64
difftype{CV<:Colorant}(::Type{CV}) = difftype(CV, eltype(CV))
difftype{CV<:RGBA,T<:Real}(::Type{CV}, ::Type{T}) = RGBA{Float32}
difftype{CV<:RGBA}(::Type{CV}, ::Type{Float64}) = RGBA{Float64}
difftype{CV<:BGRA,T<:Real}(::Type{CV}, ::Type{T}) = BGRA{Float32}
difftype{CV<:BGRA}(::Type{CV}, ::Type{Float64}) = BGRA{Float64}
difftype{CV<:AbstractGray,T<:Real}(::Type{CV}, ::Type{T}) = Gray{Float32}
difftype{CV<:AbstractGray}(::Type{CV}, ::Type{Float64}) = Gray{Float64}
difftype{CV<:AbstractRGB,T<:Real}(::Type{CV}, ::Type{T}) = RGB{Float32}
difftype{CV<:AbstractRGB}(::Type{CV}, ::Type{Float64}) = RGB{Float64}

accum{T<:Integer}(::Type{T}) = Int
accum(::Type{Float32})    = Float32
accum{T<:Real}(::Type{T}) = Float64
accum{C<:Colorant}(::Type{C}) = base_colorant_type(C){accum(eltype(C))}

graytype{T<:Number}(::Type{T}) = T
graytype{C<:AbstractGray}(::Type{C}) = C
graytype{C<:Colorant}(::Type{C}) = Gray{eltype(C)}

# normalized by Array size
"`s = ssdn(A, B)` computes the sum-of-squared differences over arrays/images A and B, normalized by array size"
ssdn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = ssd(A, B)/length(A)

# normalized by Array size
"`s = sadn(A, B)` computes the sum-of-absolute differences over arrays/images A and B, normalized by array size"
sadn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = sad(A, B)/length(A)

# normalized cross correlation
"""
`C = ncc(A, B)` computes the normalized cross-correlation of `A` and `B`.
"""
function ncc{T}(A::AbstractArray{T}, B::AbstractArray{T})
    Am = (data(A).-mean(data(A)))[:]
    Bm = (data(B).-mean(data(B)))[:]
    return dot(Am,Bm)/(norm(Am)*norm(Bm))
end

# Simple image difference testing
macro test_approx_eq_sigma_eps(A, B, sigma, eps)
    quote
        if size($(esc(A))) != size($(esc(B)))
            error("Sizes ", size($(esc(A))), " and ",
                  size($(esc(B))), " do not match")
        end
        Af = imfilter_gaussian($(esc(A)), $(esc(sigma)))
        Bf = imfilter_gaussian($(esc(B)), $(esc(sigma)))
        diffscale = max(maxabsfinite($(esc(A))), maxabsfinite($(esc(B))))
        d = sad(Af, Bf)
        if d > length(Af)*diffscale*($(esc(eps)))
            error("Arrays A and B differ")
        end
    end
end

# image difference testing (@tbreloff's, based on the macro)
#   A/B: images/arrays to compare
#   sigma: tuple of ints... how many pixels to blur
#   eps: error allowance
# returns: percentage difference on match, error otherwise
function test_approx_eq_sigma_eps{T<:Real}(A::AbstractArray, B::AbstractArray,
                                  sigma::AbstractVector{T} = ones(ndims(A)),
                                  eps::AbstractFloat = 1e-2,
                                  expand_arrays::Bool = true)
    if size(A) != size(B)
        if expand_arrays
            newsize = map(max, size(A), size(B))
            if size(A) != newsize
                A = copy!(zeros(eltype(A), newsize...), A)
            end
            if size(B) != newsize
                B = copy!(zeros(eltype(B), newsize...), B)
            end
        else
            error("Arrays differ: size(A): $(size(A)) size(B): $(size(B))")
        end
    end
    if length(sigma) != ndims(A)
        error("Invalid sigma in test_approx_eq_sigma_eps. Should be ndims(A)-length vector of the number of pixels to blur.  Got: $sigma")
    end
    Af = imfilter_gaussian(A, sigma)
    Bf = imfilter_gaussian(B, sigma)
    diffscale = max(maxabsfinite(A), maxabsfinite(B))
    d = sad(Af, Bf)
    diffpct = d / (length(Af) * diffscale)
    if diffpct > eps
        error("Arrays differ.  Difference: $diffpct  eps: $eps")
    end
    diffpct
end

# Array padding
function padindexes{T,n}(img::AbstractArray{T,n}, prepad::Union{Vector{Int},Dims}, postpad::Union{Vector{Int},Dims}, border::AbstractString)
    I = Array(Vector{Int}, n)
    for d = 1:n
        I[d] = padindexes(img, d, prepad[d], postpad[d], border)
    end
    I
end

"""
```
imgpad = padarray(img, prepad, postpad, border, value)
```

For an `N`-dimensional array `img`, apply padding on both edges. `prepad` and
`postpad` are vectors of length `N` specifying the number of pixels used to pad
each dimension. `border` is a string, one of `"value"` (to pad with a specific
pixel value), `"replicate"` (to repeat the edge value), `"circular"` (periodic
boundary conditions), `"reflect"` (reflecting boundary conditions, where the
reflection is centered on edge), and `"symmetric"` (reflecting boundary
conditions, where the reflection is centered a half-pixel spacing beyond the
edge, so the edge value gets repeated). Arrays are automatically padded before
filtering. Use `"inner"` to avoid padding altogether; the output array will be
smaller than the input.
"""
function padarray{T,n}(img::AbstractArray{T,n}, prepad::Union{Vector{Int},Dims}, postpad::Union{Vector{Int},Dims}, border::AbstractString)
    img[padindexes(img, prepad, postpad, border)...]::Array{T,n}
end
function padarray{n}(img::Union{BitArray{n}, SubArray{Bool,n,BitArray{n}}}, prepad::Union{Vector{Int},Dims}, postpad::Union{Vector{Int},Dims}, border::AbstractString)
    img[padindexes(img, prepad, postpad, border)...]::BitArray{n}
end
function padarray{n,A<:BitArray}(img::Image{Bool,n,A}, prepad::Union{Vector{Int},Dims}, postpad::Union{Vector{Int},Dims}, border::AbstractString)
    img[padindexes(img, prepad, postpad, border)...]::BitArray{n}
end

function padarray{T,n}(img::AbstractArray{T,n}, prepad::Union{Vector{Int},Dims}, postpad::Union{Vector{Int},Dims}, border::AbstractString, value)
    if border != "value"
        return padarray(img, prepad, postpad, border)
    end
    A = Array(T, ntuple(d->size(img,d)+prepad[d]+postpad[d], n))
    fill!(A, value)
    I = Vector{Int}[1+prepad[d]:size(A,d)-postpad[d] for d = 1:n]
    A[I...] = img
    A::Array{T,n}
end

padarray{T,n}(img::AbstractArray{T,n}, padding::Union{Vector{Int},Dims}, border::AbstractString = "replicate") = padarray(img, padding, padding, border)
# Restrict the following to Number to avoid trouble when img is an Array{AbstractString}
padarray{T<:Number,n}(img::AbstractArray{T,n}, padding::Union{Vector{Int},Dims}, value::T) = padarray(img, padding, padding, "value", value)

function padarray{T,n}(img::AbstractArray{T,n}, padding::Union{Vector{Int},Dims}, border::AbstractString, direction::AbstractString)
    if direction == "both"
        return padarray(img, padding, padding, border)
    elseif direction == "pre"
        return padarray(img, padding, zeros(Int, n), border)
    elseif direction == "post"
        return padarray(img, zeros(Int, n), padding, border)
    end
end

function padarray{T<:Number,n}(img::AbstractArray{T,n}, padding::Vector{Int}, value::T, direction::AbstractString)
    if direction == "both"
        return padarray(img, padding, padding, "value", value)
    elseif direction == "pre"
        return padarray(img, padding, zeros(Int, n), "value", value)
    elseif direction == "post"
        return padarray(img, zeros(Int, n), padding, "value", value)
    end
end


function prep_kernel(img::AbstractArray, kern::AbstractArray)
    sc = coords_spatial(img)
    if ndims(kern) > length(sc)
        error("""The kernel has $(ndims(kern)) dimensions, more than the $(sdims(img)) spatial dimensions of img.
                 You'll need to set the dimensions and type of the kernel to be the same as the image.""")
    end
    sz = ones(Int, ndims(img))
    for i = 1:ndims(kern)
        sz[sc[i]] = size(kern,i)
    end
    reshape(kern, sz...)
end


###
### imfilter
###
"""
```
imgf = imfilter(img, kernel, [border, value])
```

filters the array `img` with the given `kernel`, using boundary conditions
specified by `border` and `value`. See `padarray` for an explanation of
the boundary conditions. Default is to use `"replicate"` boundary conditions.
This uses finite-impulse-response (FIR) filtering, and is fast only for
relatively small `kernel`s.

See also: `imfilter_fft`, `imfilter_gaussian`.
"""
imfilter(img, kern, border, value) = imfilter_inseparable(img, kern, border, value)
# Do not combine these with the previous using a default value (see the 2d specialization below)
imfilter(img, filter) = imfilter(img, filter, "replicate", zero(eltype(img)))
imfilter(img, filter, border) = imfilter(img, filter, border, zero(eltype(img)))

imfilter_inseparable{T,K,N,M}(img::AbstractArray{T,N}, kern::AbstractArray{K,M}, border::AbstractString, value) =
    imfilter_inseparable(img, prep_kernel(img, kern), border, value)

function imfilter_inseparable{T,K,N}(img::AbstractArray{T,N}, kern::AbstractArray{K,N}, border::AbstractString, value)
    if border == "inner"
        result = Array(typeof(one(T)*one(K)), ntuple(d->max(0, size(img,d)-size(kern,d)+1), N))
        imfilter!(result, img, kern)
    else
        prepad  = [div(size(kern,i)-1, 2) for i = 1:N]
        postpad = [div(size(kern,i),   2) for i = 1:N]
        A = padarray(img, prepad, postpad, border, convert(T, value))
        result = imfilter!(Array(typeof(one(T)*one(K)), size(img)), A, data(kern))
    end
    copyproperties(img, result)
end

# Special case for 2d kernels: check for separability
function imfilter{T}(img::AbstractArray{T}, kern::AbstractMatrix, border::AbstractString, value)
    sc = coords_spatial(img)
    if length(sc) < 2
        imfilter_inseparable(img, kern, border, value)
    end
    SVD = svdfact(kern)
    U, S, Vt = SVD[:U], SVD[:S], SVD[:Vt]
    separable = true
    EPS = sqrt(eps(eltype(S)))
    for i = 2:length(S)
        separable &= (abs(S[i]) < EPS)
    end
    if !separable
        return imfilter_inseparable(img, kern, border, value)
    end
    s = S[1]
    u,v = U[:,1],Vt[1,:]
    ss = sqrt(s)
    sz1 = ones(Int, ndims(img)); sz1[sc[1]] = size(kern, 1)
    sz2 = ones(Int, ndims(img)); sz2[sc[2]] = size(kern, 2)
    tmp = imfilter_inseparable(data(img), reshape(u*ss, sz1...), border, value)
    copyproperties(img, imfilter_inseparable(tmp, reshape(v*ss, sz2...), border, value))
end

for N = 1:5
    @eval begin
        function imfilter!{T,K}(B, A::AbstractArray{T,$N}, kern::AbstractArray{K,$N})
            for i = 1:$N
                if size(B,i)+size(kern,i) > size(A,i)+1
                    throw(DimensionMismatch("Output dimensions $(size(B)) and kernel dimensions $(size(kern)) do not agree with size of padded input, $(size(A))"))
                end
            end
            (isempty(A) || isempty(kern)) && return B
            p = A[1]*kern[1]
            TT = typeof(p+p)
            @nloops $N i B begin
                tmp = zero(TT)
                @inbounds begin
                    @nloops $N j kern d->(k_d = i_d+j_d-1) begin
                        tmp += (@nref $N A k)*(@nref $N kern j)
                    end
                    (@nref $N B i) = tmp
                end
            end
            B
        end
    end
end

"""
```
imfilter!(dest, img, kernel)
```

filters the image with the given `kernel`, storing the output in the
pre-allocated output `dest`. The size of `dest` must not be greater than the
size of the result of `imfilter` with `border = "inner"`, and it behaves
identically.  This uses finite-impulse-response (FIR) filtering, and is fast
only for relatively small `kernel`s.

No padding is performed; see `padarray` for options if you want to do
this manually.

See also: `imfilter`, `padarray`.
"""
imfilter!

###
### imfilter_fft
###
"""
```
imgf = imfilter_fft(img, kernel, [border, value])
```

filters `img` with the given `kernel` using an FFT algorithm.  This
is slower than `imfilter` for small kernels, but much faster for large
kernels. For Gaussian blur, an even faster choice is `imfilter_gaussian`.

See also: `imfilter`, `imfilter_gaussian`.
"""
imfilter_fft(img, kern, border, value) = copyproperties(img, imfilter_fft_inseparable(img, kern, border, value))
imfilter_fft(img, filter) = imfilter_fft(img, filter, "replicate", 0)
imfilter_fft(img, filter, border) = imfilter_fft(img, filter, border, 0)

imfilter_fft_inseparable{T,K,N,M}(img::AbstractArray{T,N}, kern::AbstractArray{K,M}, border::AbstractString, value) =
    imfilter_fft_inseparable(img, prep_kernel(img, kern), border, value)

function imfilter_fft_inseparable{T<:Colorant,K,N,M}(img::AbstractArray{T,N}, kern::AbstractArray{K,M}, border::AbstractString, value)
    A = reinterpret(eltype(T), data(img))
    kernrs = reshape(kern, tuple(1, size(kern)...))
    B = imfilter_fft_inseparable(A, prep_kernel(A, kernrs), border, value)
    reinterpret(base_colorant_type(T), B)
end

function imfilter_fft_inseparable{T<:Real,K,N}(img::AbstractArray{T,N}, kern::AbstractArray{K,N}, border::AbstractString, value)
    if border == "circular" && size(img) == size(kern)
        A = data(img)
        krn = reflect(kern)
        out = real(ifftshift(ifft(fft(A).*fft(krn))))
    elseif border != "inner"
        prepad  = [div(size(kern,i)-1, 2) for i = 1:N]
        postpad = [div(size(kern,i),   2) for i = 1:N]
        fullpad = Int[nextprod([2,3], size(img,i) + prepad[i] + postpad[i]) - size(img, i) - prepad[i] for i = 1:N]  # work around julia #15276
        A = padarray(img, prepad, fullpad, border, convert(T, value))
        krn = zeros(eltype(one(T)*one(K)), size(A))
        indexesK = ntuple(d->[size(krn,d)-prepad[d]+1:size(krn,d);1:size(kern,d)-prepad[d]], N)
        krn[indexesK...] = reflect(kern)
        AF = ifft(fft(A).*fft(krn))
        out = Array(realtype(eltype(AF)), size(img))
        indexesA = ntuple(d->postpad[d]+1:size(img,d)+postpad[d], N)
        copyreal!(out, AF, indexesA)
    else
        A = data(img)
        prepad  = [div(size(kern,i)-1, 2) for i = 1:N]
        postpad = [div(size(kern,i),   2) for i = 1:N]
        krn = zeros(eltype(one(T)*one(K)), size(A))
        indexesK = ntuple(d->[size(krn,d)-prepad[d]+1:size(krn,d);1:size(kern,d)-prepad[d]], N)
        krn[indexesK...] = reflect(kern)
        AF = ifft(fft(A).*fft(krn))
        out = Array(realtype(eltype(AF)), ([size(img)...] - prepad - postpad)...)
        indexesA = ntuple(d->postpad[d]+1:size(img,d)-prepad[d], N)
        copyreal!(out, AF, indexesA)
    end
    out
end

# flips the dimension specified by name instead of index
# it is thus independent of the storage order
Base.flipdim(img::AbstractImage, dimname::String) = shareproperties(img, flipdim(data(img), dimindex(img, dimname)))

flipx(img::AbstractImage) = flipdim(img, "x")
flipy(img::AbstractImage) = flipdim(img, "y")
flipz(img::AbstractImage) = flipdim(img, "z")

# Generalization of rot180
@generated function reflect{T,N}(A::AbstractArray{T,N})
    quote
        B = Array(T, size(A))
        @nexprs $N d->(n_d = size(A, d)+1)
        @nloops $N i A d->(j_d = n_d - i_d) begin
            @nref($N, B, j) = @nref($N, A, i)
        end
        B
    end
end


for N = 1:5
    @eval begin
        function copyreal!{T<:Real}(dst::Array{T,$N}, src, I::Tuple{Vararg{UnitRange{Int}}})
            @nexprs $N d->(I_d = I[d])
            @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                (@nref $N dst i) = real(@nref $N src j)
            end
            dst
        end
        function copyreal!{T<:Complex}(dst::Array{T,$N}, src, I::Tuple{Vararg{UnitRange{Int}}})
            @nexprs $N d->I_d = I[d]
            @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                (@nref $N dst i) = @nref $N src j
            end
            dst
        end
    end
end

realtype{R<:Real}(::Type{R}) = R
realtype{R<:Real}(::Type{Complex{R}}) = R


# IIR filtering with Gaussians
# See
#  Young, van Vliet, and van Ginkel, "Recursive Gabor Filtering",
#    IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 50: 2798-2805.
# and
#  Triggs and Sdika, "Boundary Conditions for Young - van Vliet
#    Recursive Filtering, IEEE TRANSACTIONS ON SIGNAL PROCESSING,
# Here we're using NA boundary conditions, so we set i- and i+
# (in Triggs & Sdika notation) to zero.
# Note these two papers use different sign conventions for the coefficients.

# Note: astype is ignored for AbstractFloat input
"""
```
imgf = imfilter_gaussian(img, sigma)
```

filters `img` with a Gaussian of the specified width. `sigma` should have
one value per array dimension (any number of dimensions are supported), 0
indicating that no filtering is to occur along that dimension. Uses the Young,
van Vliet, and van Ginkel IIR-based algorithm to provide fast gaussian filtering
even with large `sigma`. Edges are handled by "NA" conditions, meaning the
result is normalized by the number and weighting of available pixels, and
missing data (NaNs) are handled likewise.
"""
function imfilter_gaussian{CT<:Colorant}(img::AbstractArray{CT}, sigma; emit_warning = true, astype::Type=Float64)
    A = reinterpret(eltype(CT), data(img))
    newsigma = ndims(A) > ndims(img) ? [0;sigma] : sigma
    ret = imfilter_gaussian(A, newsigma; emit_warning=emit_warning, astype=astype)
    shareproperties(img, reinterpret(base_colorant_type(CT), ret))
end

function imfilter_gaussian{T<:AbstractFloat}(img::AbstractArray{T}, sigma::Vector; emit_warning = true, astype::Type=Float64)
    if all(sigma .== 0)
        return img
    end
    A = copy(data(img))
    nanflag = @compat isnan.(A)
    hasnans = any(nanflag)
    if hasnans
        A[nanflag] = zero(T)
        validpixels = convert(Array{T}, !nanflag)
        imfilter_gaussian!(A, validpixels, sigma; emit_warning=emit_warning)
        A[nanflag] = convert(T, NaN)
    else
        imfilter_gaussian_no_nans!(A, sigma; emit_warning=emit_warning)
    end
    shareproperties(img, A)
end

# For these types, you can't have NaNs
function imfilter_gaussian{T<:Union{Integer,UFixed},TF<:AbstractFloat}(img::AbstractArray{T}, sigma::Vector; emit_warning = true, astype::Type{TF}=Float64)
    A = copy!(Array(TF, size(img)), data(img))
    if all(sigma .== 0)
        return shareproperties(img, A)
    end
    imfilter_gaussian_no_nans!(A, sigma; emit_warning=emit_warning)
    shareproperties(img, A)
end

# This version is in-place, and destructive
# Any NaNs have to already be removed from data (and marked in validpixels)
function imfilter_gaussian!{T<:AbstractFloat}(data::Array{T}, validpixels::Array{T}, sigma::Vector; emit_warning = true)
    nd = ndims(data)
    if length(sigma) != nd
        error("Dimensionality mismatch")
    end
    _imfilter_gaussian!(data, sigma, emit_warning=emit_warning)
    _imfilter_gaussian!(validpixels, sigma, emit_warning=false)
    for i = 1:length(data)
        data[i] /= validpixels[i]
    end
    return data
end

# When there are no NaNs, the normalization is separable and hence can be computed far more efficiently
# This speeds the algorithm by approximately twofold
function imfilter_gaussian_no_nans!{T<:AbstractFloat}(data::Array{T}, sigma::Vector; emit_warning = true)
    nd = ndims(data)
    if length(sigma) != nd
        error("Dimensionality mismatch")
    end
    _imfilter_gaussian!(data, sigma, emit_warning=emit_warning)
    denom = Array(Vector{T}, nd)
    for i = 1:nd
        denom[i] = ones(T, size(data, i))
        if sigma[i] > 0
            _imfilter_gaussian!(denom[i], sigma[i:i], emit_warning=false)
        end
    end
    imfgnormalize!(data, denom)
    return data
end

for N = 1:5
    @eval begin
        function imfgnormalize!{T}(data::Array{T,$N}, denom)
            @nextract $N denom denom
            @nloops $N i data begin
                den = one(T)
                @nexprs $N d->(den *= denom_d[i_d])
                (@nref $N data i) /= den
            end
        end
    end
end

function iir_gaussian_coefficients(T::Type, sigma::Number; emit_warning::Bool = true)
    if sigma < 1 && emit_warning
        warn("sigma is too small for accuracy")
    end
    m0 = convert(T,1.16680)
    m1 = convert(T,1.10783)
    m2 = convert(T,1.40586)
    q = convert(T,1.31564*(sqrt(1+0.490811*sigma*sigma) - 1))
    scale = (m0+q)*(m1*m1 + m2*m2  + 2m1*q + q*q)
    B = m0*(m1*m1 + m2*m2)/scale
    B *= B
    # This is what Young et al call b, but in filt() notation would be called a
    a1 = q*(2*m0*m1 + m1*m1 + m2*m2 + (2*m0+4*m1)*q + 3*q*q)/scale
    a2 = -q*q*(m0 + 2m1 + 3q)/scale
    a3 = q*q*q/scale
    a = [-a1,-a2,-a3]
    Mdenom = (1+a1-a2+a3)*(1-a1-a2-a3)*(1+a2+(a1-a3)*a3)
    M = [-a3*a1+1-a3^2-a2      (a3+a1)*(a2+a3*a1)  a3*(a1+a3*a2);
          a1+a3*a2            -(a2-1)*(a2+a3*a1)  -(a3*a1+a3^2+a2-1)*a3;
          a3*a1+a2+a1^2-a2^2   a1*a2+a3*a2^2-a1*a3^2-a3^3-a3*a2+a3  a3*(a1+a3*a2)]/Mdenom;
    return a, B, M
end

function _imfilter_gaussian!{T<:AbstractFloat}(A::Array{T}, sigma::Vector; emit_warning::Bool = true)
    nd = ndims(A)
    szA = [size(A,i) for i = 1:nd]
    strdsA = [stride(A,i) for i = 1:nd]
    for d = 1:nd
        if sigma[d] == 0
            continue
        end
        if size(A, d) < 3
            error("All filtered dimensions must be of size 3 or larger")
        end
        a, B, M = iir_gaussian_coefficients(T, sigma[d], emit_warning=emit_warning)
        a1 = a[1]
        a2 = a[2]
        a3 = a[3]
        n1 = size(A,1)
        keepdims = [false;trues(nd-1)]
        if d == 1
            x = zeros(T, 3)
            vstart = zeros(T, 3)
            szhat = szA[keepdims]
            strdshat = strdsA[keepdims]
            if isempty(szhat)
                szhat = [1]
                strdshat = [1]
            end
            @inbounds @forcartesian c szhat begin
                coloffset = offset(c, strdshat)
                A[2+coloffset] -= a1*A[1+coloffset]
                A[3+coloffset] -= a1*A[2+coloffset] + a2*A[1+coloffset]
                for i = 4+coloffset:n1+coloffset
                    A[i] -= a1*A[i-1] + a2*A[i-2] + a3*A[i-3]
                end
                copytail!(x, A, coloffset, 1, n1)
                A_mul_B!(vstart, M, x)
                A[n1+coloffset] = vstart[1]
                A[n1-1+coloffset] -= a1*vstart[1]   + a2*vstart[2] + a3*vstart[3]
                A[n1-2+coloffset] -= a1*A[n1-1+coloffset] + a2*vstart[1] + a3*vstart[2]
                for i = n1-3+coloffset:-1:1+coloffset
                    A[i] -= a1*A[i+1] + a2*A[i+2] + a3*A[i+3]
                end
            end
        else
            x = Array(T, 3, n1)
            vstart = similar(x)
            keepdims[d] = false
            szhat = szA[keepdims]
            szd = szA[d]
            strdshat = strdsA[keepdims]
            strdd = strdsA[d]
            if isempty(szhat)
                szhat = [1]
                strdshat = [1]
            end
            @inbounds @forcartesian c szhat begin
                coloffset = offset(c, strdshat)  # offset for the remaining dimensions
                for i = 1:n1 A[i+strdd+coloffset] -= a1*A[i+coloffset] end
                for i = 1:n1 A[i+2strdd+coloffset] -= a1*A[i+strdd+coloffset] + a2*A[i+coloffset] end
                for j = 3:szd-1
                    jj = j*strdd+coloffset
                    for i = jj+1:jj+n1 A[i] -= a1*A[i-strdd] + a2*A[i-2strdd] + a3*A[i-3strdd] end
                end
                copytail!(x, A, coloffset, strdd, szd)
                A_mul_B!(vstart, M, x)
                for i = 1:n1 A[i+(szd-1)*strdd+coloffset] = vstart[1,i] end
                for i = 1:n1 A[i+(szd-2)*strdd+coloffset] -= a1*vstart[1,i]   + a2*vstart[2,i] + a3*vstart[3,i] end
                for i = 1:n1 A[i+(szd-3)*strdd+coloffset] -= a1*A[i+(szd-2)*strdd+coloffset] + a2*vstart[1,i] + a3*vstart[2,i] end
                for j = szd-4:-1:0
                    jj = j*strdd+coloffset
                    for i = jj+1:jj+n1 A[i] -= a1*A[i+strdd] + a2*A[i+2strdd] + a3*A[i+3strdd] end
                end
            end
        end
        @inbounds for i = 1:length(A)
            A[i] *= B
        end
    end
    A
end

function offset(c::Vector{Int}, strds::Vector{Int})
    o = 0
    for i = 1:length(c)
        o += (c[i]-1)*strds[i]
    end
    o
end

function copytail!(dest, A, coloffset, strd, len)
    for j = 1:3
        for i = 1:size(dest, 2)
            tmp = A[i + coloffset + (len-j)*strd]
            dest[j,i] = tmp
        end
    end
    dest
end

"""
`blob_LoG(img, sigmas) -> Vector{Tuple}`

Find "blobs" in an N-D image using Lapacian of Gaussians at the specifed
sigmas.  Returned are the local maxima's heights, radii, and spatial coordinates.

See Lindeberg T (1998), "Feature Detection with Automatic Scale Selection",
International Journal of Computer Vision, 30(2), 79–116.

Note that only 2-D images are currently supported due to a limitation of `imfilter_LoG`.
"""
@generated function blob_LoG{T,N}(img::AbstractArray{T,N}, sigmas)
    quote
        img_LoG = Array(Float64, length(sigmas), size(img)...)
        @inbounds for isigma in eachindex(sigmas)
            img_LoG[isigma,:] = sigmas[isigma] * imfilter_LoG(img, sigmas[isigma])
        end

        radii = sqrt(2.0)*sigmas
        maxima = findlocalmaxima(img_LoG, 1:ndims(img_LoG), (true, falses(N)...))
        [(img_LoG[x...], radii[x[1]], (@ntuple $N d->x[d+1])...) for x in maxima]
    end
end

findlocalextrema{T,N}(img::AbstractArray{T,N}, region, edges::Bool, order) = findlocalextrema(img, region, ntuple(d->edges,N), order)

@generated function findlocalextrema{T,N}(img::AbstractArray{T,N}, region::Union{Tuple{Int},Vector{Int},UnitRange{Int},Int}, edges::NTuple{N,Bool}, order::Base.Order.Ordering)
    quote
        issubset(region,1:ndims(img)) || throw(ArgumentError("Invalid region."))
        extrema = Tuple{(@ntuple $N d->Int)...}[]
        @inbounds @nloops $N i d->((1+!edges[d]):(size(img,d)-!edges[d])) begin
            isextrema = true
            img_I = (@nref $N img i)
            @nloops $N j d->(in(d,region) ? (max(1,i_d-1):min(size(img,d),i_d+1)) : i_d) begin
                (@nall $N d->(j_d == i_d)) && continue
                if !Base.Order.lt(order, (@nref $N img j), img_I)
                    isextrema = false
                    break
                end
            end
            isextrema && push!(extrema, (@ntuple $N d->(i_d)))
        end
        extrema
    end
end

"""
`findlocalmaxima(img, [region, edges]) -> Vector{Tuple}`

Returns the coordinates of elements whose value is larger than all of
their immediate neighbors.  `region` is a list of dimensions to
consider.  `edges` is a boolean specifying whether to include the
first and last elements of each dimension, or a tuple-of-Bool
specifying edge behavior for each dimension separately.
"""
findlocalmaxima(img::AbstractArray, region=coords_spatial(img), edges=true) =
        findlocalextrema(img, region, edges, Base.Order.Forward)

"""
Like `findlocalmaxima`, but returns the coordinates of the smallest elements.
"""
findlocalminima(img::AbstractArray, region=coords_spatial(img), edges=true) =
        findlocalextrema(img, region, edges, Base.Order.Reverse)

# Laplacian of Gaussian filter
# Separable implementation from Huertas and Medioni,
# IEEE Trans. Pat. Anal. Mach. Int., PAMI-8, 651, (1986)
"""
```
imgf = imfilter_LoG(img, sigma, [border])
```

filters a 2D image with a Laplacian of Gaussian of the specified width. `sigma`
may be a vector with one value per array dimension, or may be a single scalar
value for uniform filtering in both dimensions.  Uses the Huertas and Medioni
separable algorithm.
"""
function imfilter_LoG{T}(img::AbstractArray{T,2}, σ::Vector, border="replicate")
    # Limited to 2D for now.
    # See Sage D, Neumann F, Hediger F, Gasser S, Unser M.
    # Image Processing, IEEE Transactions on. (2005) 14(9):1372-1383.
    # for 3D.

    # Set up 1D kernels
    @assert length(σ) == 2
    σx, σy = σ[1], σ[2]
    h1(ξ, σ) = sqrt(1/(2π*σ^4))*(1 - ξ^2/σ^2)exp(-ξ^2/(2σ^2))
    h2(ξ, σ) = sqrt(1/(2π*σ^4))*exp(-ξ^2/(2σ^2))

    w = 8.5σx
    kh1x = Float64[h1(i, σx) for i = -floor(w/2):floor(w/2)]
    kh2x = Float64[h2(i, σx) for i = -floor(w/2):floor(w/2)]

    w = 8.5σy
    kh1y = Float64[h1(i, σy) for i = -floor(w/2):floor(w/2)]
    kh2y = Float64[h2(i, σy) for i = -floor(w/2):floor(w/2)]

    # Set up padding index lists
    kernlenx = length(kh1x)
    prepad  = div(kernlenx - 1, 2)
    postpad = div(kernlenx, 2)
    Ix = padindexes(img, 1, prepad, postpad, border)

    kernleny = length(kh1y)
    prepad  = div(kernleny - 1, 2)
    postpad = div(kernleny, 2)
    Iy = padindexes(img, 2, prepad, postpad, border)

    sz = size(img)
    # Store intermediate result in a transposed array
    # Allows column-major second stage filtering
    img1 = Array(Float64, (sz[2], sz[1]))
    img2 = Array(Float64, (sz[2], sz[1]))
    for j in 1:sz[2]
        for i in 1:sz[1]
            tmp1, tmp2 = 0.0, 0.0
            for k in 1:kernlenx
                @inbounds tmp1 += kh1x[k] * img[Ix[i + k - 1], j]
                @inbounds tmp2 += kh2x[k] * img[Ix[i + k - 1], j]
            end
            @inbounds img1[j, i] = tmp1 # Note the transpose
            @inbounds img2[j, i] = tmp2
        end
    end

    img12 = Array(Float64, sz) # Original image dims here
    img21 = Array(Float64, sz)
    for j in 1:sz[1]
        for i in 1:sz[2]
            tmp12, tmp21 = 0.0, 0.0
            for k in 1:kernleny
                @inbounds tmp12 += kh2y[k] * img1[Iy[i + k - 1], j]
                @inbounds tmp21 += kh1y[k] * img2[Iy[i + k - 1], j]
            end
            @inbounds img12[j, i] = tmp12 # Transpose back to original dims
            @inbounds img21[j, i] = tmp21
        end
    end
    copyproperties(img, img12 + img21)
end

imfilter_LoG{T}(img::AbstractArray{T,2}, σ::Real, border="replicate") =
    imfilter_LoG(img::AbstractArray{T,2}, [σ, σ], border)

function padindexes{T,n}(img::AbstractArray{T,n}, dim, prepad, postpad, border::AbstractString)
    M = size(img, dim)
    I = Array(Int, M + prepad + postpad)
    I = [(1 - prepad):(M + postpad);]
    @compat if border == "replicate"
        I = min.(max.(I, 1), M)
    elseif border == "circular"
        I = 1 .+ mod.(I .- 1, M)
    elseif border == "symmetric"
        I = [1:M; M:-1:1][1 .+ mod.(I .- 1, 2 * M)]
    elseif border == "reflect"
        I = [1:M; M-1:-1:2][1 .+ mod.(I .- 1, 2 * M - 2)]
    else
        error("unknown border condition")
    end
    I
end

### restrict, for reducing the image size by 2-fold
# This properly anti-aliases. The only "oddity" is that edges tend towards zero under
# repeated iteration.
# This implementation is considerably faster than the one in Grid,
# because it traverses the array in storage order.
if isdefined(:restrict)
    import Grid.restrict
end

"""
`imgr = restrict(img[, region])` performs two-fold reduction in size
along the dimensions listed in `region`, or all spatial coordinates if
`region` is not specified.  It anti-aliases the image as it goes, so
is better than a naive summation over 2x2 blocks.
"""
restrict(img::AbstractImageDirect, ::Tuple{}) = img

function restrict(img::AbstractImageDirect, region::Union{Dims, Vector{Int}}=coords_spatial(img))
    A = data(img)
    for dim in region
        A = _restrict(A, dim)
    end
    props = copy(properties(img))
    ps = copy(pixelspacing(img))
    ind = findin(coords_spatial(img), region)
    ps[ind] *= 2
    props["pixelspacing"] = ps
    Image(A, props)
end
function restrict(A::AbstractArray, region::Union{Dims, Vector{Int}}=coords_spatial(A))
    for dim in region
        A = _restrict(A, dim)
    end
    A
end

function restrict{S<:String}(img::AbstractImageDirect, region::Union{Tuple{Vararg{String}}, Vector{S}})
    so = spatialorder(img)
    regioni = Int[]
    for i = 1:length(region)
        push!(regioni, require_dimindex(img, region[i], so))
    end
    restrict(img, regioni)
end

function _restrict(A, dim)
    if size(A, dim) <= 2
        return A
    end
    out = Array(typeof(A[1]/4+A[2]/2), ntuple(i->i==dim?restrict_size(size(A,dim)):size(A,i), ndims(A)))
    restrict!(out, A, dim)
    out
end

# out should have efficient linear indexing
for N = 1:5
    @eval begin
        function restrict!{T}(out::AbstractArray{T,$N}, A::AbstractArray, dim)
            if isodd(size(A, dim))
                half = convert(eltype(T), 0.5)
                quarter = convert(eltype(T), 0.25)
                indx = 0
                if dim == 1
                    @inbounds @nloops $N i d->(d==1 ? (1:1) : (1:size(A,d))) d->(j_d = d==1 ? i_d+1 : i_d) begin
                        nxt = convert(T, @nref $N A j)
                        out[indx+=1] = half*(@nref $N A i) + quarter*nxt
                        for k = 3:2:size(A,1)-2
                            prv = nxt
                            i_1 = k
                            j_1 = k+1
                            nxt = convert(T, @nref $N A j)
                            out[indx+=1] = quarter*(prv+nxt) + half*(@nref $N A i)
                        end
                        i_1 = size(A,1)
                        out[indx+=1] = quarter*nxt + half*(@nref $N A i)
                    end
                else
                    strd = stride(out, dim)
                    # Must initialize the i_dim==1 entries with zero
                    @nexprs $N d->sz_d=d==dim?1:size(out,d)
                    @nloops $N i d->(1:sz_d) begin
                        (@nref $N out i) = zero(T)
                    end
                    stride_1 = 1
                    @nexprs $N d->(stride_{d+1} = stride_d*size(out,d))
                    @nexprs $N d->offset_d = 0
                    ispeak = true
                    @inbounds @nloops $N i d->(d==1?(1:1):(1:size(A,d))) d->(if d==dim; ispeak=isodd(i_d); offset_{d-1} = offset_d+(div(i_d+1,2)-1)*stride_d; else; offset_{d-1} = offset_d+(i_d-1)*stride_d; end) begin
                        indx = offset_0
                        if ispeak
                            for k = 1:size(A,1)
                                i_1 = k
                                out[indx+=1] += half*(@nref $N A i)
                            end
                        else
                            for k = 1:size(A,1)
                                i_1 = k
                                tmp = quarter*(@nref $N A i)
                                out[indx+=1] += tmp
                                out[indx+strd] = tmp
                            end
                        end
                    end
                end
            else
                threeeighths = convert(eltype(T), 0.375)
                oneeighth = convert(eltype(T), 0.125)
                indx = 0
                if dim == 1
                    z = convert(T, zero(A[1]))
                    @inbounds @nloops $N i d->(d==1 ? (1:1) : (1:size(A,d))) d->(j_d = i_d) begin
                        c = d = z
                        for k = 1:size(out,1)-1
                            a = c
                            b = d
                            j_1 = 2*k
                            i_1 = j_1-1
                            c = convert(T, @nref $N A i)
                            d = convert(T, @nref $N A j)
                            out[indx+=1] = oneeighth*(a+d) + threeeighths*(b+c)
                        end
                        out[indx+=1] = oneeighth*c+threeeighths*d
                    end
                else
                    fill!(out, zero(T))
                    strd = stride(out, dim)
                    stride_1 = 1
                    @nexprs $N d->(stride_{d+1} = stride_d*size(out,d))
                    @nexprs $N d->offset_d = 0
                    peakfirst = true
                    @inbounds @nloops $N i d->(d==1?(1:1):(1:size(A,d))) d->(if d==dim; peakfirst=isodd(i_d); offset_{d-1} = offset_d+(div(i_d+1,2)-1)*stride_d; else; offset_{d-1} = offset_d+(i_d-1)*stride_d; end) begin
                        indx = offset_0
                        if peakfirst
                            for k = 1:size(A,1)
                                i_1 = k
                                tmp = @nref $N A i
                                out[indx+=1] += threeeighths*tmp
                                out[indx+strd] += oneeighth*tmp
                            end
                        else
                            for k = 1:size(A,1)
                                i_1 = k
                                tmp = @nref $N A i
                                out[indx+=1] += oneeighth*tmp
                                out[indx+strd] += threeeighths*tmp
                            end
                        end
                    end
                end
            end
        end
    end
end

restrict_size(len::Integer) = isodd(len) ? (len+1)>>1 : (len>>1)+1

## imresize
function imresize!(resized, original)
    assert2d(original)
    scale1 = (size(original,1)-1)/(size(resized,1)-0.999f0)
    scale2 = (size(original,2)-1)/(size(resized,2)-0.999f0)
    for jr = 0:size(resized,2)-1
        jo = scale2*jr
        ijo = trunc(Int, jo)
        fjo = jo - oftype(jo, ijo)
        @inbounds for ir = 0:size(resized,1)-1
            io = scale1*ir
            iio = trunc(Int, io)
            fio = io - oftype(io, iio)
            tmp = (1-fio)*((1-fjo)*original[iio+1,ijo+1] +
                           fjo*original[iio+1,ijo+2]) +
                  + fio*((1-fjo)*original[iio+2,ijo+1] +
                         fjo*original[iio+2,ijo+2])
            resized[ir+1,jr+1] = convertsafely(eltype(resized), tmp)
        end
    end
    resized
end

imresize(original, new_size) = size(original) == new_size ? copy!(similar(original), original) : imresize!(similar(original, new_size), original)

convertsafely{T<:AbstractFloat}(::Type{T}, val) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::Integer) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::AbstractFloat) = round(T, val)
convertsafely{T}(::Type{T}, val) = convert(T, val)


function imlineardiffusion{T}(img::Array{T,2}, dt::AbstractFloat, iterations::Integer)
    u = img
    f = imlaplacian()
    for i = dt:dt:dt*iterations
        u = u + dt*imfilter(u, f, "replicate")
    end
    u
end

function imgaussiannoise{T}(img::AbstractArray{T}, variance::Number, mean::Number)
    return img + sqrt(variance)*randn(size(img)) + mean
end

imgaussiannoise{T}(img::AbstractArray{T}, variance::Number) = imgaussiannoise(img, variance, 0)
imgaussiannoise{T}(img::AbstractArray{T}) = imgaussiannoise(img, 0.01, 0)

# image gradients

# forward and backward differences
# can be very helpful for discretized continuous models
forwarddiffy{T}(u::Array{T,2}) = [u[2:end,:]; u[end,:]] - u
forwarddiffx{T}(u::Array{T,2}) = [u[:,2:end] u[:,end]] - u
backdiffy{T}(u::Array{T,2}) = u - [u[1,:]; u[1:end-1,:]]
backdiffx{T}(u::Array{T,2}) = u - [u[:,1] u[:,1:end-1]]

"""
```
imgr = imROF(img, lambda, iterations)
```

Perform Rudin-Osher-Fatemi (ROF) filtering, more commonly known as Total
Variation (TV) denoising or TV regularization. `lambda` is the regularization
coefficient for the derivative, and `iterations` is the number of relaxation
iterations taken. 2d only.
"""
function imROF{T}(img::Array{T,2}, lambda::Number, iterations::Integer)
    # Total Variation regularized image denoising using the primal dual algorithm
    # Also called Rudin Osher Fatemi (ROF) model
    # lambda: regularization parameter
    s1, s2 = size(img)
    p = zeros(T, s1, s2, 2)
    u = zeros(T, s1, s2)
    grad_u = zeros(T, s1, s2, 2)
    div_p = zeros(T, s1, s2)
    dt = lambda/4
    for i = 1:iterations
        div_p = backdiffx(p[:,:,1]) + backdiffy(p[:,:,2])
        u = img + div_p/lambda
        grad_u = cat(3, forwarddiffx(u), forwarddiffy(u))
        grad_u_mag = sqrt(grad_u[:,:,1].^2 + grad_u[:,:,2].^2)
        tmp = 1 + grad_u_mag*dt
        p = (dt*grad_u + p)./cat(3, tmp, tmp)
    end
    return u
end

# ROF Model for color images
function imROF(img::AbstractArray, lambda::Number, iterations::Integer)
    cd = colordim(img)
    local out
    if cd != 0
        out = similar(img)
        for i = size(img, cd)
            imsl = img["color", i]
            outsl = slice(out, "color", i)
            copy!(outsl, imROF(imsl, lambda, iterations))
        end
    else
        out = shareproperties(img, imROF(data(img), lambda, iterations))
    end
    out
end


### Morphological operations

# Erode and dilate support 3x3 regions only (and higher-dimensional generalizations).
"""
```
imgd = dilate(img, [region])
```

perform a max-filter over nearest-neighbors. The
default is 8-connectivity in 2d, 27-connectivity in 3d, etc. You can specify the
list of dimensions that you want to include in the connectivity, e.g., `region =
[1,2]` would exclude the third dimension from filtering.
"""
dilate(img::AbstractImageDirect, region=coords_spatial(img)) = shareproperties(img, dilate!(copy(data(img)), region))
"""
```
imge = erode(img, [region])
```

perform a min-filter over nearest-neighbors. The
default is 8-connectivity in 2d, 27-connectivity in 3d, etc. You can specify the
list of dimensions that you want to include in the connectivity, e.g., `region =
[1,2]` would exclude the third dimension from filtering.
"""
erode(img::AbstractImageDirect, region=coords_spatial(img)) = shareproperties(img, erode!(copy(data(img)), region))
dilate(img::AbstractArray, region=coords_spatial(img)) = dilate!(copy(img), region)
erode(img::AbstractArray, region=coords_spatial(img)) = erode!(copy(img), region)

dilate!(maxfilt, region=coords_spatial(maxfilt)) = extremefilt!(data(maxfilt), Base.Order.Forward, region)
erode!(minfilt, region=coords_spatial(minfilt)) = extremefilt!(data(minfilt), Base.Order.Reverse, region)
function extremefilt!(extrfilt::AbstractArray, order::Ordering, region=coords_spatial(extrfilt))
    for d = 1:ndims(extrfilt)
        if size(extrfilt, d) == 1 || !in(d, region)
            continue
        end
        sz = [size(extrfilt,i) for i = 1:ndims(extrfilt)]
        s = stride(extrfilt, d)
        sz[d] = 1
        @forcartesian i sz begin
            k = cartesian_linear(extrfilt, i)
            a2 = extrfilt[k]
            a3 = extrfilt[k+s]
            extrfilt[k] = extr(order, a2, a3)
            for l = 2:size(extrfilt,d)-1
                k += s
                a1 = a2
                a2 = a3
                a3 = extrfilt[k+s]
                extrfilt[k] = extr(order, a1, a2, a3)
            end
            extrfilt[k+s] = extr(order, a2, a3)
        end
    end
    extrfilt
end

"""
`imgo = opening(img, [region])` performs the `opening` morphology operation, equivalent to `dilate(erode(img))`.
`region` allows you to control the dimensions over which this operation is performed.
"""
opening(img::AbstractArray, region=coords_spatial(img)) = opening!(copy(img), region)
opening!(img::AbstractArray, region=coords_spatial(img)) = dilate!(erode!(img, region),region)

"""
`imgc = closing(img, [region])` performs the `closing` morphology operation, equivalent to `erode(dilate(img))`.
`region` allows you to control the dimensions over which this operation is performed.
"""
closing(img::AbstractArray, region=coords_spatial(img)) = closing!(copy(img), region)
closing!(img::AbstractArray, region=coords_spatial(img)) = erode!(dilate!(img, region),region)

"""
`imgth = tophat(img, [region])` performs `top hat` of an image,
which is defined as the image minus its morphological opening.
`region` allows you to control the dimensions over which this operation is performed.
"""
tophat(img::AbstractArray, region=coords_spatial(img)) = img - opening(img, region)

"""
`imgbh = bothat(img, [region])` performs `bottom hat` of an image,
which is defined as its morphological closing minus the original image.
`region` allows you to control the dimensions over which this operation is performed.
"""
bothat(img::AbstractArray, region=coords_spatial(img)) = closing(img, region) - img

"""
`imgmg = morphogradient(img, [region])` returns morphological gradient of the image,
which is the difference between the dilation and the erosion of a given image.
`region` allows you to control the dimensions over which this operation is performed.
"""
morphogradient(img::AbstractArray, region=coords_spatial(img)) = dilate(img, region) - erode(img, region)

"""
`imgml = morpholaplace(img, [region])` performs `Morphological Laplacian` of an image,
which is defined as the arithmetic difference between the internal and the external gradient.
`region` allows you to control the dimensions over which this operation is performed.
"""
morpholaplace(img::AbstractArray, region=coords_spatial(img)) = dilate(img, region) + erode(img, region) - 2img

extr(order::ForwardOrdering, x::Real, y::Real) = max(x,y)
extr(order::ForwardOrdering, x::Real, y::Real, z::Real) = max(x,y,z)
extr(order::ReverseOrdering, x::Real, y::Real) = min(x,y)
extr(order::ReverseOrdering, x::Real, y::Real, z::Real) = min(x,y,z)

extr(order::Ordering, x::RGB, y::RGB) = RGB(extr(order, x.r, y.r), extr(order, x.g, y.g), extr(order, x.b, y.b))
extr(order::Ordering, x::RGB, y::RGB, z::RGB) = RGB(extr(order, x.r, y.r, z.r), extr(order, x.g, y.g, z.g), extr(order, x.b, y.b, z.b))

extr(order::Ordering, x::Color, y::Color) = extr(order, convert(RGB, x), convert(RGB, y))
extr(order::Ordering, x::Color, y::Color, z::Color) = extr(order, convert(RGB, x), convert(RGB, y), convert(RGB, z))



# Min max filter

# This is a port of the Lemire min max filter as implemented by Bruno Luong
# http://arxiv.org/abs/cs.DS/0610046
# http://lemire.me/
# http://www.mathworks.com/matlabcentral/fileexchange/24705-min-max-filter

type Wedge{A <: AbstractArray}
    buffer::A
    size::Int
    n::Int
    first::Int
    last::Int
    mxn::Int
end


for N = 2:4
    @eval begin
    function extrema_filter{T <: Number}(A::Array{T, $N}, window::Array{Int, 1})

        maxval_temp = copy(A); minval_temp = copy(A)

        for dim = 1:$N

            # For all but the last dimension
            @nloops $(N-1) i maxval_temp begin

                # Create index for full array (fa) length
                @nexprs $(N)   j->(fa_{j} = 1:size(maxval_temp)[j])
                @nexprs $(N-1) j->(fa_{j} = i_j)

                # Create index for short array (sa) length
                @nexprs $(N)   j->(sa_{j} = 1:size(maxval_temp)[j] - window[dim] + 1)
                @nexprs $(N-1) j->(sa_{j} = i_j)

                # Filter the last dimension
                (@nref $N minval_temp sa) = min_filter(vec( @nref $N minval_temp fa), window[dim])
                (@nref $N maxval_temp sa) = max_filter(vec( @nref $N maxval_temp fa), window[dim])

            end

            # Circular shift the dimensions
            maxval_temp = permutedims(maxval_temp, mod(collect(1:$N), $N)+1)
            minval_temp = permutedims(minval_temp, mod(collect(1:$N), $N)+1)

        end

        # The dimensions to extract
        @nexprs $N j->(a_{j} = 1:size(A, j)-window[j]+1)

        # Extract set dimensions
        maxval_out = @nref $N maxval_temp a
        minval_out = @nref $N minval_temp a

        return minval_out, maxval_out
    end
    end
end


function extrema_filter{T <: Number}(a::AbstractArray{T}, window::Int)

    n = length(a)

    # Initialise the output variables
    # This is the running minimum and maximum over the specified window length
    minval = zeros(T, 1, n-window+1)
    maxval = zeros(T, 1, n-window+1)

    # Initialise the internal wedges
    # U[1], L[1] are the location of the global maximum and minimum
    # U[2], L[2] are the maximum and minimum over (U1, inf)
    L = Wedge(zeros(Int,1,window+1), window+1, 0, 1, 0, 0)          # Min
    U = Wedge(zeros(Int,1,window+1), window+1, 0, 1, 0, 0)

    for i = 2:n
        if i > window
            if !wedgeisempty(U)
                maxval[i-window] = a[getfirst(U)]
            else
                maxval[i-window] = a[i-1]
            end
            if !wedgeisempty(L)
                minval[i-window] = a[getfirst(L)]
            else
                minval[i-window] = a[i-1]
            end
        end # window

        if a[i] > a[i-1]
            pushback!(L, i-1)
            if i==window+getfirst(L); L=popfront(L); end
            while !wedgeisempty(U)
                if a[i] <= a[getlast(U)]
                    if i == window+getfirst(U); U = popfront(U); end
                    break
                end
                U = popback(U)
            end

        else

            pushback!(U, i-1)
            if i==window+getfirst(U); U=popfront(U); end

            while !wedgeisempty(L)
                if a[i] >= a[getlast(L)]
                    if i == window+getfirst(L); L = popfront(L); end
                    break
                end
                L = popback(L)
            end

        end  # a>a-1

    end # for i

    i = n+1
    if !wedgeisempty(U)
        maxval[i-window] = a[getfirst(U)]
    else
        maxval[i-window] = a[i-1]
    end

    if !wedgeisempty(L)
        minval[i-window] = a[getfirst(L)]
    else
        minval[i-window] = a[i-1]
    end

    return minval, maxval
end


function min_filter(a::AbstractArray, window::Int)

    minval, maxval = extrema_filter(a, window)

    return minval
end


function max_filter(a::AbstractArray, window::Int)

    minval, maxval = extrema_filter(a, window)

    return maxval
end


function wedgeisempty(X::Wedge)
    X.n <= 0
end

function pushback!(X::Wedge, v)
    X.last += 1
    if X.last > X.size
        X.last = 1
    end
    X.buffer[X.last] = v
    X.n = X.n+1
    X.mxn = max(X.mxn, X.n)
end

function getfirst(X::Wedge)
    X.buffer[X.first]
end

function getlast(X::Wedge)
    X.buffer[X.last]
end

function popfront(X::Wedge)
    X.n = X.n-1
    X.first = mod(X.first, X.size) + 1
    return X
end

function popback(X::Wedge)
    X.n = X.n-1
    X.last = mod(X.last-2, X.size) + 1
    return X
end




# phantom images

"""
```
phantom = shepp_logan(N,[M]; highContrast=true)
```

output the NxM Shepp-Logan phantom, which is a standard test image usually used
for comparing image reconstruction algorithms in the field of computed
tomography (CT) and magnetic resonance imaging (MRI). If the argument M is
omitted, the phantom is of size NxN. When setting the keyword argument
``highConstrast` to false, the CT version of the phantom is created. Otherwise,
the high contrast MRI version is calculated.
"""
function shepp_logan(M,N; highContrast=true)
  # Initially proposed in Shepp, Larry; B. F. Logan (1974).
  # "The Fourier Reconstruction of a Head Section". IEEE Transactions on Nuclear Science. NS-21.

  P = zeros(M,N)

  x = linspace(-1,1,M)'
  y = linspace(1,-1,N)

  centerX = [0, 0, 0.22, -0.22, 0, 0, 0, -0.08, 0, 0.06]
  centerY = [0, -0.0184, 0, 0, 0.35, 0.1, -0.1, -0.605, -0.605, -0.605]
  majorAxis = [0.69, 0.6624, 0.11, 0.16, 0.21, 0.046, 0.046, 0.046, 0.023, 0.023]
  minorAxis = [0.92, 0.874, 0.31, 0.41, 0.25, 0.046, 0.046, 0.023, 0.023, 0.046]
  theta = [0, 0, -18.0, 18.0, 0, 0, 0, 0, 0, 0]

  # original (CT) version of the phantom
  grayLevel = [2, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

  if(highContrast)
    # high contrast (MRI) version of the phantom
    grayLevel = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  end

  for l=1:length(theta)
    P += grayLevel[l] * (
           ( (cos(theta[l] / 360*2*pi) * (x .- centerX[l]) .+
              sin(theta[l] / 360*2*pi) * (y .- centerY[l])) / majorAxis[l] ).^2 .+
           ( (sin(theta[l] / 360*2*pi) * (x .- centerX[l]) .-
              cos(theta[l] / 360*2*pi) * (y .- centerY[l])) / minorAxis[l] ).^2 .< 1 )
  end

  return P
end

shepp_logan(N;highContrast=true) = shepp_logan(N,N;highContrast=highContrast)

"""
```
integral_img = integral_image(img)
```

Returns the integral image of an image. The integral image is calculated by assigning
to each pixel the sum of all pixels above it and to its left, i.e. the rectangle from
(1, 1) to the pixel. An integral image is a data structure which helps in efficient
calculation of sum of pixels in a rectangular subset of an image. See `boxdiff` for more
information.
"""
function integral_image(img::AbstractArray)
    integral_img = Array{accum(eltype(img))}(size(img))
    sd = coords_spatial(img)
    cumsum!(integral_img, img, sd[1])
    for i = 2:length(sd)
        cumsum!(integral_img, integral_img, sd[i])
    end
    integral_img
end

"""
```
sum = boxdiff(integral_image, ytop:ybot, xtop:xbot)
sum = boxdiff(integral_image, CartesianIndex(tl_y, tl_x), CartesianIndex(br_y, br_x))
sum = boxdiff(integral_image, tl_y, tl_x, br_y, br_x)
```

An integral image is a data structure which helps in efficient calculation of sum of pixels in
a rectangular subset of an image. It stores at each pixel the sum of all pixels above it and to
its left. The sum of a window in an image can be directly calculated using four array
references of the integral image, irrespective of the size of the window, given the `yrange` and
`xrange` of the window. Given an integral image -

        A - - - - - - B -
        - * * * * * * * -
        - * * * * * * * -
        - * * * * * * * -
        - * * * * * * * -
        - * * * * * * * -
        C * * * * * * D -
        - - - - - - - - -

The sum of pixels in the area denoted by * is given by S = D + A - B - C.
"""
boxdiff{T}(int_img::AbstractArray{T, 2}, y::UnitRange, x::UnitRange) = boxdiff(int_img, y.start, x.start, y.stop, x.stop)
boxdiff{T}(int_img::AbstractArray{T, 2}, tl::CartesianIndex, br::CartesianIndex) = boxdiff(int_img, tl[1], tl[2], br[1], br[2])

function boxdiff{T}(int_img::AbstractArray{T, 2}, tl_y::Integer, tl_x::Integer, br_y::Integer, br_x::Integer)
    sum = int_img[br_y, br_x]
    sum -= tl_x > 1 ? int_img[br_y, tl_x - 1] : zero(T)
    sum -= tl_y > 1 ? int_img[tl_y - 1, br_x] : zero(T)
    sum += tl_y > 1 && tl_x > 1 ? int_img[tl_y - 1, tl_x - 1] : zero(T)
    sum
end

"""
```
P = bilinear_interpolation(img, r, c)
```

Bilinear Interpolation is used to interpolate functions of two variables
on a rectilinear 2D grid.

The interpolation is done in one direction first and then the values obtained
are used to do the interpolation in the second direction.

"""
function bilinear_interpolation{T}(img::AbstractArray{T, 2}, y::Number, x::Number)
    y_min = floor(Int, y)
    x_min = floor(Int, x)
    y_max = ceil(Int, y)
    x_max = ceil(Int, x)

    topleft = chkbounds(Bool, img, y_min, x_min) ? img[y_min, x_min] : zero(T)
    bottomleft = chkbounds(Bool, img, y_max, x_min) ? img[y_max, x_min] : zero(T)
    topright = chkbounds(Bool, img, y_min, x_max) ? img[y_min, x_max] : zero(T)
    bottomright = chkbounds(Bool, img, y_max, x_max) ? img[y_max, x_max] : zero(T)

    if x_max == x_min
        if y_max == y_min
            return T(topleft)
        end
        return T(((y_max - y) * topleft + (y - y_min) * bottomleft) / (y_max - y_min))
    elseif y_max == y_min
        return T(((x_max - x) * topleft + (x - x_min) * topright) / (x_max - x_min))
    end

    r1 = ((x_max - x) * topleft + (x - x_min) * topright) / (x_max - x_min)
    r2 = ((x_max - x) * bottomleft + (x - x_min) * bottomright) / (x_max - x_min)

    T(((y_max - y) * r1 + (y - y_min) * r2) / (y_max - y_min))

end

if VERSION < v"0.5.0-dev+4754"
    chkbounds(::Type{Bool}, img, x, y)  = checkbounds(Bool, size(img, 1), y) && checkbounds(Bool, size(img, 2), x)
else
    chkbounds(::Type{Bool}, img, x, y) = checkbounds(Bool, img, x, y)
end

"""
```
pyramid = gaussian_pyramid(img, n_scales, downsample, sigma)
```

Returns a  gaussian pyramid of scales `n_scales`, each downsampled
by a factor `downsample` and `sigma` for the gaussian kernel.

"""
function gaussian_pyramid{T}(img::AbstractArray{T, 2}, n_scales::Int, downsample::Real, sigma::Real)
    prev = img
    img_smoothed_main = imfilter_gaussian(prev, [sigma, sigma])
    pyramid = typeof(img_smoothed_main)[]
    push!(pyramid, img_smoothed_main)
    prev_h, prev_w = size(img)
    for i in 1:n_scales
        next_h = ceil(Int, prev_h / downsample)
        next_w = ceil(Int, prev_w / downsample)
        img_smoothed = imfilter_gaussian(prev, [sigma, sigma])
        img_scaled = imresize(img_smoothed, (next_h, next_w))
        push!(pyramid, img_scaled)
        prev = img_scaled
        prev_h, prev_w = size(img_scaled)
    end
    pyramid
end
