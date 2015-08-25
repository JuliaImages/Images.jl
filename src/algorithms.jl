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
(+)(img::AbstractImageDirect, A::AbstractArray) = shareproperties(img, data(img)+data(A))
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
(-)(img::AbstractImageDirect, A::AbstractArray) = shareproperties(img, data(img)-data(A))
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
sqrt(img::AbstractImageDirect) = shareproperties(img, sqrt(data(img)))
atan2(img1::AbstractImageDirect, img2::AbstractImageDirect) = shareproperties(img1, atan2(data(img1),data(img2)))
hypot(img1::AbstractImageDirect, img2::AbstractImageDirect) = shareproperties(img1, hypot(data(img1),data(img2)))

@vectorize_2arg Gray atan2
@vectorize_2arg Gray hypot

function sum(img::AbstractImageDirect, region::Union(AbstractVector,Tuple,Integer))
    f = prod(size(img)[[region...]])
    out = copyproperties(img, sum(data(img), region))
    if in(colordim(img), region)
        out["colorspace"] = "Unknown"
    end
    out
end

meanfinite{T<:Real}(A::AbstractArray{T}, region) = _meanfinite(A, T, region)
meanfinite{CT<:Colorant}(A::AbstractArray{CT}, region) = _meanfinite(A, eltype(CT), region)
function _meanfinite{T<:FloatingPoint}(A::AbstractArray, ::Type{T}, region)
    sz = Base.reduced_dims(A, region)
    K = zeros(Int, sz)
    S = zeros(eltype(A), sz)
    sumfinite!(S, K, A)
    S./K
end
_meanfinite(A::AbstractArray, ::Type, region) = mean(A, region)  # non floating-point

function meanfinite{T<:FloatingPoint}(img::AbstractImageDirect{T}, region)
    r = meanfinite(data(img), region)
    out = copyproperties(img, r)
    if in(colordim(img), region)
        out["colorspace"] = "Unknown"
    end
    out
end
meanfinite(img::AbstractImageIndexed, region) = meanfinite(convert(Image, img), region)
# Note that you have to zero S and K upon entry
@ngenerate N typeof((S,K)) function sumfinite!{T,N}(S, K, A::AbstractArray{T,N})
    isempty(A) && return S, K
    @nexprs N d->(sizeS_d = size(S,d))
    sizeA1 = size(A, 1)
    if size(S, 1) == 1 && sizeA1 > 1
        # When we are reducing along dim == 1, we can accumulate to a temporary
        @inbounds @nloops N i d->(d>1? (1:size(A,d)) : (1:1)) d->(j_d = sizeS_d==1 ? 1 : i_d) begin
            s = @nref(N, S, j)
            k = @nref(N, K, j)
            for i_1 = 1:sizeA1
                tmp = @nref(N, A, i)
                if isfinite(tmp)
                    s += tmp
                    k += 1
                end
            end
            @nref(N, S, j) = s
            @nref(N, K, j) = k
        end
    else
        # Accumulate to array storage
        @inbounds @nloops N i A d->(j_d = sizeS_d==1 ? 1 : i_d) begin
            tmp = @nref(N, A, i)
            if isfinite(tmp)
                @nref(N, S, j) += tmp
                @nref(N, K, j) += 1
            end
        end
    end
    S, K
end

# Logical operations
(.<)(img::AbstractImageDirect, n::Number) = data(img) .< n
(.>)(img::AbstractImageDirect, n::Number) = data(img) .> n
(.<)(img::AbstractImageDirect{Bool}, A::AbstractArray{Bool}) = data(img) .< A
(.<)(img::AbstractImageDirect, A::AbstractArray) = data(img) .< A
(.>)(img::AbstractImageDirect, A::AbstractArray) = data(img) .> A
(.==)(img::AbstractImageDirect, n::Number) = data(img) .== n
(.==)(img::AbstractImageDirect{Bool}, A::AbstractArray{Bool}) = data(img) .== A
(.==)(img::AbstractImageDirect, A::AbstractArray) = data(img) .== A



function imadjustintensity{T}(img::AbstractArray{T}, range)
    assert_scalar_color(img)
    if length(range) == 0
        range = [minimum(img) maximum(img)]
    elseif length(range) == 1
        error("incorrect range")
    end
    tmp = (img .- range[1])/(range[2] - range[1])
    tmp[tmp .> 1] = 1
    tmp[tmp .< 0] = 0
    out = tmp
end

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

function minfinite{T}(A::AbstractArray{T})
    ret = sentinel_min(T)
    for a in A
        ret = minfinite_scalar(a, ret)
    end
    ret
end

function maxfinite{T}(A::AbstractArray{T})
    ret = sentinel_max(T)
    for a in A
        ret = maxfinite_scalar(a, ret)
    end
    ret
end

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
minfinite_scalar{T<:Union(Integer,FixedPoint)}(a::T, b::T) = b < a ? b : a
maxfinite_scalar{T<:Union(Integer,FixedPoint)}(a::T, b::T) = b > a ? b : a
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

sentinel_min{T<:Union(Integer,FixedPoint)}(::Type{T}) = typemax(T)
sentinel_max{T<:Union(Integer,FixedPoint)}(::Type{T}) = typemin(T)
sentinel_min{T<:FloatingPoint}(::Type{T}) = convert(T, NaN)
sentinel_max{T<:FloatingPoint}(::Type{T}) = convert(T, NaN)
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
function imlaplacian(diagonals::String="nodiagonals")
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
function imdog(sigma::Number=0.5)
    m = 4*ceil(sqrt(2)*sigma)+1
    return gaussian2d(sqrt(2)*sigma, [m m]) - gaussian2d(sigma, [m m])
end

# laplacian of gaussian
function imlog(sigma::Number=0.5)
    m = ceil(8.5sigma)
    m = m % 2 == 0 ? m + 1 : m
    return [(1/(2pi*sigma^4))*(2 - (x^2 + y^2)/sigma^2)*exp(-(x^2 + y^2)/(2sigma^2))
            for x=-floor(m/2):floor(m/2), y=-floor(m/2):floor(m/2)]
end

# Sum of squared differences and sum of absolute differences
for (f, op) in ((:ssd, :(sumsq(x))), (:sad, :(abs(x))))
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

difftype{T<:Integer}(::Type{T}) = Int
difftype{T<:Real}(::Type{T}) = Float32
difftype(::Type{Float64}) = Float64
difftype{CV<:Colorant}(::Type{CV}) = difftype(CV, eltype(CV))
difftype{CV<:AbstractGray,T<:Real}(::Type{CV}, ::Type{T}) = Gray{Float32}
difftype{CV<:AbstractGray}(::Type{CV}, ::Type{Float64}) = Gray{Float64}
difftype{CV<:AbstractRGB,T<:Real}(::Type{CV}, ::Type{T}) = RGB{Float32}
difftype{CV<:AbstractRGB}(::Type{CV}, ::Type{Float64}) = RGB{Float64}

accum{T<:Integer}(::Type{T}) = Int
accum(::Type{Float32})    = Float32
accum{T<:Real}(::Type{T}) = Float64

# normalized by Array size
ssdn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = ssd(A, B)/length(A)

# normalized by Array size
sadn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = sad(A, B)/length(A)

# normalized cross correlation
function ncc{T}(A::AbstractArray{T}, B::AbstractArray{T})
    Am = (data(A).-mean(data(A)))[:]
    Bm = (data(B).-mean(data(B)))[:]
    return dot(Am,Bm)/(norm(Am)*norm(Bm))
end

# Array padding
function padindexes{T,n}(img::AbstractArray{T,n}, prepad::Union(Vector{Int},Dims), postpad::Union(Vector{Int},Dims), border::String)
    I = Array(Vector{Int}, n)
    for d = 1:n
        I[d] = padindexes(img, d, prepad[d], postpad[d], border)
    end
    I
end

function padarray{T,n}(img::AbstractArray{T,n}, prepad::Union(Vector{Int},Dims), postpad::Union(Vector{Int},Dims), border::String)
    img[padindexes(img, prepad, postpad, border)...]::Array{T,n}
end
function padarray{n}(img::BitArray{n}, prepad::Union(Vector{Int},Dims), postpad::Union(Vector{Int},Dims), border::String)
    img[padindexes(img, prepad, postpad, border)...]::BitArray{n}
end
function padarray{n,A<:BitArray}(img::Image{Bool,n,A}, prepad::Union(Vector{Int},Dims), postpad::Union(Vector{Int},Dims), border::String)
    img[padindexes(img, prepad, postpad, border)...]::BitArray{n}
end

function padarray{T,n}(img::AbstractArray{T,n}, prepad::Union(Vector{Int},Dims), postpad::Union(Vector{Int},Dims), border::String, value)
    if border != "value"
        return padarray(img, prepad, postpad, border)
    end
    A = Array(T, ntuple(d->size(img,d)+prepad[d]+postpad[d], n))
    fill!(A, value)
    I = Vector{Int}[1+prepad[d]:size(A,d)-postpad[d] for d = 1:n]
    A[I...] = img
    A::Array{T,n}
end

padarray{T,n}(img::AbstractArray{T,n}, padding::Union(Vector{Int},Dims), border::String = "replicate") = padarray(img, padding, padding, border)
# Restrict the following to Number to avoid trouble when img is an Array{String}
padarray{T<:Number,n}(img::AbstractArray{T,n}, padding::Union(Vector{Int},Dims), value::T) = padarray(img, padding, padding, "value", value)

function padarray{T,n}(img::AbstractArray{T,n}, padding::Union(Vector{Int},Dims), border::String, direction::String)
    if direction == "both"
        return padarray(img, padding, padding, border)
    elseif direction == "pre"
        return padarray(img, padding, zeros(Int, n), border)
    elseif direction == "post"
        return padarray(img, zeros(Int, n), padding, border)
    end
end

function padarray{T<:Number,n}(img::AbstractArray{T,n}, padding::Vector{Int}, value::T, direction::String)
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
imfilter(img, kern, border, value) = imfilter_inseparable(img, kern, border, value)
# Do not combine these with the previous using a default value (see the 2d specialization below)
imfilter(img, filter) = imfilter(img, filter, "replicate", zero(eltype(img)))
imfilter(img, filter, border) = imfilter(img, filter, border, zero(eltype(img)))

imfilter_inseparable{T,K,N,M}(img::AbstractArray{T,N}, kern::AbstractArray{K,M}, border::String, value) =
    imfilter_inseparable(img, prep_kernel(img, kern), border, value)

function imfilter_inseparable{T,K,N}(img::AbstractArray{T,N}, kern::AbstractArray{K,N}, border::String, value)
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
function imfilter{T}(img::AbstractArray{T}, kern::AbstractMatrix, border::String, value)
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


###
### imfilter_fft
###
imfilter_fft(img, kern, border, value) = copyproperties(img, imfilter_fft_inseparable(img, kern, border, value))
imfilter_fft(img, filter) = imfilter_fft(img, filter, "replicate", 0)
imfilter_fft(img, filter, border) = imfilter_fft(img, filter, border, 0)

imfilter_fft_inseparable{T,K,N,M}(img::AbstractArray{T,N}, kern::AbstractArray{K,M}, border::String, value) =
    imfilter_fft_inseparable(img, prep_kernel(img, kern), border, value)

function imfilter_fft_inseparable{T<:Colorant,K,N,M}(img::AbstractArray{T,N}, kern::AbstractArray{K,M}, border::String, value)
    A = reinterpret(eltype(T), data(img))
    kernrs = reshape(kern, tuple(1, size(kern)...))
    B = imfilter_fft_inseparable(A, prep_kernel(A, kernrs), border, value)
    reinterpret(base_colorant_type(T), B)
end

function imfilter_fft_inseparable{T<:Real,K,N}(img::AbstractArray{T,N}, kern::AbstractArray{K,N}, border::String, value)
    if border != "inner"
        prepad  = [div(size(kern,i)-1, 2) for i = 1:N]
        postpad = [div(size(kern,i),   2) for i = 1:N]
        fullpad = [nextprod([2,3], size(img,i) + prepad[i] + postpad[i]) - size(img, i) - prepad[i] for i = 1:N]
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

# Generalization of rot180
@ngenerate N Array{T,N} function reflect{T,N}(A::AbstractArray{T,N})
    B = Array(T, size(A))
    @nexprs N d->(n_d = size(A, d)+1)
    @nloops N i A d->(j_d = n_d - i_d) begin
        @nref(N, B, j) = @nref(N, A, i)
    end
    B
end

for N = 1:5
    @eval begin
        function copyreal!{T<:Real}(dst::Array{T,$N}, src, I::(@compat Tuple{Vararg{UnitRange{Int}}}))
            @nexprs $N d->(I_d = I[d])
            @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                (@nref $N dst i) = real(@nref $N src j)
            end
            dst
        end
        function copyreal!{T<:Complex}(dst::Array{T,$N}, src, I::(@compat Tuple{Vararg{UnitRange{Int}}}))
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

# Note: astype is ignored for FloatingPoint input
function imfilter_gaussian{CT<:Colorant}(img::AbstractArray{CT}, sigma; emit_warning = true, astype::Type=Float64)
    A = reinterpret(eltype(CT), data(img))
    newsigma = ndims(A) > ndims(img) ? [0;sigma] : sigma
    ret = imfilter_gaussian(A, newsigma; emit_warning=emit_warning, astype=astype)
    shareproperties(img, reinterpret(base_colorant_type(CT), ret))
end

function imfilter_gaussian{T<:FloatingPoint}(img::AbstractArray{T}, sigma::Vector; emit_warning = true, astype::Type=Float64)
    if all(sigma .== 0)
        return img
    end
    A = copy(data(img))
    nanflag = isnan(A)
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
function imfilter_gaussian{T<:Union(Integer,Ufixed),TF<:FloatingPoint}(img::AbstractArray{T}, sigma::Vector; emit_warning = true, astype::Type{TF}=Float64)
    A = convert(Array{TF}, data(img))
    if all(sigma .== 0)
        return shareproperties(img, A)
    end
    imfilter_gaussian_no_nans!(A, sigma; emit_warning=emit_warning)
    shareproperties(img, A)
end

# This version is in-place, and destructive
# Any NaNs have to already be removed from data (and marked in validpixels)
function imfilter_gaussian!{T<:FloatingPoint}(data::Array{T}, validpixels::Array{T}, sigma::Vector; emit_warning = true)
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
function imfilter_gaussian_no_nans!{T<:FloatingPoint}(data::Array{T}, sigma::Vector; emit_warning = true)
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

function _imfilter_gaussian!{T<:FloatingPoint}(A::Array{T}, sigma::Vector; emit_warning::Bool = true)
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

# Laplacian of Gaussian filter
# Separable implementation from Huertas and Medioni,
# IEEE Trans. Pat. Anal. Mach. Int., PAMI-8, 651, (1986)

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

function padindexes{T,n}(img::AbstractArray{T,n}, dim, prepad, postpad, border::String)
    M = size(img, dim)
    I = Array(Int, M + prepad + postpad)
    I = [(1 - prepad):(M + postpad);]
    if border == "replicate"
        I = min(max(I, 1), M)
    elseif border == "circular"
        I = 1 .+ mod(I .- 1, M)
    elseif border == "symmetric"
        I = [1:M; M:-1:1][1 .+ mod(I .- 1, 2 * M)]
    elseif border == "reflect"
        I = [1:M; M-1:-1:2][1 .+ mod(I .- 1, 2 * M - 2)]
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

restrict(img::AbstractImageDirect, ::(@compat Tuple{})) = img

function restrict(img::AbstractImageDirect, region::Union(Dims, Vector{Int})=coords_spatial(img))
    A = data(img)
    for dim in region
        A = _restrict(A, dim)
    end
    props = copy(properties(img))
    ps = pixelspacing(img)
    ind = findin(coords_spatial(img), region)
    ps[ind] *= 2
    props["pixelspacing"] = ps
    Image(A, props)
end
function restrict(A::AbstractArray, region::Union(Dims, Vector{Int})=coords_spatial(A))
    for dim in region
        A = _restrict(A, dim)
    end
    A
end

function restrict{S<:ByteString}(img::AbstractImageDirect, region::Union((@compat Tuple{Vararg{ByteString}}), Vector{S}))
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
                        nxt = @nref $N A j
                        out[indx+=1] = half*(@nref $N A i) + quarter*nxt
                        for k = 3:2:size(A,1)-2
                            prv = nxt
                            i_1 = k
                            j_1 = k+1
                            nxt = @nref $N A j
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
                    z = zero(A[1])
                    @inbounds @nloops $N i d->(d==1 ? (1:1) : (1:size(A,d))) d->(j_d = i_d) begin
                        c = d = z
                        for k = 1:size(out,1)-1
                            a = c
                            b = d
                            j_1 = 2*k
                            i_1 = j_1-1
                            c = @nref $N A i
                            d = @nref $N A j
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

imresize(original, new_size) = imresize!(similar(original, new_size), original)

convertsafely{T<:FloatingPoint}(::Type{T}, val) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::Integer) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::FloatingPoint) = trunc(T, val+oftype(val, 0.5))
convertsafely{T}(::Type{T}, val) = convert(T, val)


function imlineardiffusion{T}(img::Array{T,2}, dt::FloatingPoint, iterations::Integer)
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

function imcomplement{T}(img::AbstractArray{T})
    return 1 - img
end

function _imstretch{T}(img::AbstractArray{T}, m::Number, slope::Number)
    shareproperties(img, 1./(1 + (m./(data(img) .+ eps(T))).^slope))
end
imstretch(img::AbstractArray, m::Number, slope::Number) = _imstretch(float(img), m, slope)

# image gradients

# forward and backward differences
# can be very helpful for discretized continuous models
forwarddiffy{T}(u::Array{T,2}) = [u[2:end,:]; u[end,:]] - u
forwarddiffx{T}(u::Array{T,2}) = [u[:,2:end] u[:,end]] - u
backdiffy{T}(u::Array{T,2}) = u - [u[1,:]; u[1:end-1,:]]
backdiffx{T}(u::Array{T,2}) = u - [u[:,1] u[:,1:end-1]]

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
dilate(img::AbstractImageDirect, region=coords_spatial(img)) = shareproperties(img, dilate!(copy(data(img)), region))
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

opening(img::AbstractArray, region=coords_spatial(img)) = opening!(copy(img), region)
opening!(img::AbstractArray, region=coords_spatial(img)) = dilate!(erode!(img, region),region)
closing(img::AbstractArray, region=coords_spatial(img)) = closing!(copy(img), region)
closing!(img::AbstractArray, region=coords_spatial(img)) = erode!(dilate!(img, region),region)

extr(order::ForwardOrdering, x::Real, y::Real) = max(x,y)
extr(order::ForwardOrdering, x::Real, y::Real, z::Real) = max(x,y,z)
extr(order::ReverseOrdering, x::Real, y::Real) = min(x,y)
extr(order::ReverseOrdering, x::Real, y::Real, z::Real) = min(x,y,z)

extr(order::Ordering, x::RGB, y::RGB) = RGB(extr(order, x.r, y.r), extr(order, x.g, y.g), extr(order, x.b, y.b))
extr(order::Ordering, x::RGB, y::RGB, z::RGB) = RGB(extr(order, x.r, y.r, z.r), extr(order, x.g, y.g, z.g), extr(order, x.b, y.b, z.b))

extr(order::Ordering, x::Color, y::Color) = extr(order, convert(RGB, x), convert(RGB, y))
extr(order::Ordering, x::Color, y::Color, z::Color) = extr(order, convert(RGB, x), convert(RGB, y), convert(RGB, z))


# phantom images

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
