#### Math with images ####

(+)(img::AbstractImageDirect{Bool}, n::Bool) = img .+ n
(+)(img::AbstractImageDirect, n::Number) = img .+ n
(+)(n::Bool, img::AbstractImageDirect{Bool}) = n .+ img
(+)(n::Number, img::AbstractImageDirect) = n .+ img
(.+)(img::AbstractImageDirect, n::Number) = copy(img, data(img).+n)
(.+)(n::Number, img::AbstractImageDirect) = copy(img, data(img).+n)
(+)(img::AbstractImageDirect, A::BitArray) = copy(img, data(img)+A)
if isdefined(:UniformScaling)
    (+){Timg,TA<:Number}(img::AbstractImageDirect{Timg,2}, A::UniformScaling{TA}) = copy(img, data(img)+A)
    (-){Timg,TA<:Number}(img::AbstractImageDirect{Timg,2}, A::UniformScaling{TA}) = copy(img, data(img)-A)
end
(+)(img::AbstractImageDirect, A::AbstractArray) = copy(img, data(img)+data(A))
(.+)(img::AbstractImageDirect, A::BitArray) = copy(img, data(img).+A)
(.+)(img::AbstractImageDirect, A::AbstractArray) = copy(img, data(img).+data(A))
(-)(img::AbstractImageDirect{Bool}, n::Bool) = img .- n
(-)(img::AbstractImageDirect, n::Number) = img .- n
(.-)(img::AbstractImageDirect, n::Number) = copy(img, data(img).-n)
(-)(n::Bool, img::AbstractImageDirect{Bool}) = n .- img
(-)(n::Number, img::AbstractImageDirect) = n .- img
(.-)(n::Number, img::AbstractImageDirect) = copy(img, n.-data(img))
(-)(img::AbstractImageDirect, A::BitArray) = copy(img, data(img)-A)
(-){T}(img::AbstractImageDirect{T,2}, A::Diagonal) = copy(img, data(img)-A) # fixes an ambiguity warning
(-)(img::AbstractImageDirect, A::AbstractArray) = copy(img, data(img)-data(A))
# (-)(A::AbstractArray, img::AbstractImageDirect) = limadj(copy(img, data(A) - data(img)), limminus(limits(A), limits(img)))
(.-)(img::AbstractImageDirect, A::BitArray) = copy(img, data(img).-A)
(.-)(img::AbstractImageDirect, A::AbstractArray) = copy(img, data(img).-data(A))
(*)(img::AbstractImageDirect, n::Number) = (.*)(img, n)
(*)(n::Number, img::AbstractImageDirect) = (.*)(n, img)
(.*)(img::AbstractImageDirect, n::Number) = copy(img, data(img).*n)
(.*)(n::Number, img::AbstractImageDirect) = copy(img, data(img).*n)
(/)(img::AbstractImageDirect, n::Number) = copy(img, data(img)/n)
(.*)(img1::AbstractImageDirect, img2::AbstractImageDirect) = copy(img1, data(img1).*data(img2))
(.*)(img::AbstractImageDirect, A::BitArray) = copy(img, data(img).*A)
(.*)(A::BitArray, img::AbstractImageDirect) = copy(img, data(img).*A)
(.*)(img::AbstractImageDirect{Bool}, A::BitArray) = copy(img, data(img).*A)
(.*)(A::BitArray, img::AbstractImageDirect{Bool}) = copy(img, data(img).*A)
(.*)(img::AbstractImageDirect, A::AbstractArray) = copy(img, data(img).*A)
(.*)(A::AbstractArray, img::AbstractImageDirect) = copy(img, data(img).*A)
(./)(img::AbstractImageDirect, A::BitArray) = copy(img, data(img)./A)  # needed to avoid ambiguity warning
(./)(img1::AbstractImageDirect, img2::AbstractImageDirect) = copy(img1, data(img1)./data(img2))
(./)(img::AbstractImageDirect, A::AbstractArray) = copy(img, data(img)./A)
# (./)(A::AbstractArray, img::AbstractImageDirect) = limadj(copy(img, A./data(img))
(.^)(img::AbstractImageDirect, p::Number) = copy(img, data(img).^p)

function sum(img::AbstractImageDirect, region::Union(AbstractVector,Tuple,Integer))
    f = prod(size(img)[[region...]])
    out = copy(img, sum(data(img), region))
    if in(colordim(img), region)
        out["colorspace"] = "Unknown"
    end
    out
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

#### Overlay AbstractArray implementation ####
length(o::Overlay) = isempty(o.channels) ? 0 : length(o.channels[1])
size(o::Overlay) = isempty(o.channels) ? (0,) : size(o.channels[1])
size(o::Overlay, d::Integer) = isempty(o.channels) ? 0 : size(o.channels[1],d)
eltype(o::Overlay) = RGB
nchannels(o::Overlay) = length(o.channels)

similar(o::Overlay) = Array(RGB, size(o))
similar(o::Overlay, ::NTuple{0}) = Array(RGB, size(o))
similar{T}(o::Overlay, ::Type{T}) = Array(T, size(o))
similar{T}(o::Overlay, ::Type{T}, sz::Int64) = Array(T, sz)
similar{T}(o::Overlay, ::Type{T}, sz::Int64...) = Array(T, sz)
similar{T}(o::Overlay, ::Type{T}, sz) = Array(T, sz)

function getindex(o::Overlay, indexes::Integer...)
    rgb = RGB(0,0,0)
    for i = 1:nchannels(o)
        if o.visible[i]
            rgb += scale(o.scalei[i], o.channels[i][indexes...])*o.colors[i]
        end
    end
    clamp(rgb)
end

# Fix ambiguity warning
getindex(o::Overlay, i::Real) = getindex_overlay(o, i)
getindex(o::Overlay, indexes::Union(Real,AbstractVector)...) = getindex_overlay(o, indexes...)

function getindex_overlay(o::Overlay, indexes::Union(Real,AbstractVector)...)
    len = [length(i) for i in indexes]
    n = length(len)
    while len[n] == 1
        pop!(len)
        n -= 1
    end
    rgb = fill(RGB(0,0,0), len...)
    for i = 1:nchannels(o)
        if o.visible[i]
            tmp = scale(o.scalei[i], o.channels[i][indexes...])
            accumulate!(rgb, tmp, o.colors[i])
        end
    end
    clip!(rgb)
end

getindex(o::Overlay, indexes::(AbstractVector...)) = getindex(o, indexes...)

setindex!(o::Overlay, x, index::Real) = error("Overlays are read-only")
setindex!(o::Overlay, x, indexes...) = error("Overlays are read-only")

# Identical to getindex except it saves a call to each array's getindex
function convert(::Type{Array{RGB}}, o::Overlay)
    rgb = fill(RGB(0,0,0), size(o))
    for i = 1:length(o.channels)
        if o.visible[i]
            tmp = scale(o.scalei[i], o.channels[i])
            accumulate!(rgb, tmp, o.colors[i])
        end
    end
    clip!(rgb)
end

function accumulate!(rgb::Array{RGB}, A::Array{Float64}, color::RGB)
    for j = 1:length(rgb)
        rgb[j] += A[j]*color
    end
end

for N = 1:4
    @eval begin
        function accumulate!{T}(rgb::Array{RGB,$N}, A::AbstractArray{T,$N}, color::RGB, scalei::ScaleInfo)
            k = 0
            @inbounds @nloops $N i A begin
                rgb[k+=1] += scale(scalei, (@nref $N A i))*color
            end
        end
    end
end

(*)(f::FloatingPoint, c::RGB) = RGB(f*c.r, f*c.g, f*c.b)
(*)(f::Uint8, c::RGB) = (f/255)*c
(/)(c::RGB, f::Real) = (1.0/f)*c
(.*)(f::AbstractArray, c::RGB) = [x*c for x in f]
(+)(a::RGB, b::RGB) = RGB(a.r+b.r, a.g+b.g, a.b+b.b)
convert(::Type{Uint32}, c::ColorValue) = convert(RGB24, c).color




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
        function $funcname{CV<:ColorValue}(img::AbstractArray{CV})
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

for N = 1:4
    N1 = N-1
    @eval begin
        function rgb2gray{T}(img::AbstractArray{T,$N})
            cs = colorspace(img)
            if cs == "Gray"
                return img
            end
            if cs != "RGB"
                error("Color space of image is $cs, not RGB")
            end
            strds = strides(img)
            cd = colordim(img)
            cstrd = strds[cd]
            sz = [size(img,d) for d=1:$N]
            sz[cd] = 1
            szgray = sz[setdiff(1:$N,cd)]
            out = Array(T, szgray...)::Array{T,$N1}
            wr, wg, wb = 0.299, 0.587, 0.114    # Rec 601 luma conversion
            dat = data(img)
            indx = 0
            @nexprs $N d->(strd_d = strds[d])
            @nexprs $N d->(sz_d = sz[d])
            @nexprs $N d->(k_d = 1)
            @nloops $N i d->1:sz_d d->(k_{d-1}=k_d+strd_d*(i_d-1)) begin
                out[indx+=1] = truncround(T,wr*dat[k_0] + wg*dat[k_0+cstrd] + wb*dat[k_0+2cstrd])
            end
            p = copy(properties(img))
            p["colorspace"] = "Gray"
            p["colordim"] = 0
            Image(out, p)
        end
    end
end

function sobel()
    f = [1.0 2.0 1.0; 0.0 0.0 0.0; -1.0 -2.0 -1.0]
    return f, f'
end

function prewitt()
    f = [1.0 1.0 1.0; 0.0 0.0 0.0; -1.0 -1.0 -1.0]
    return f, f'
end

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
        m = 4*ceil(sigma)+1
        n = m
    elseif length(filter_size) != 2
        error("wrong filter size")
    else
        m, n = filter_size[1], filter_size[2]
    end
    if mod(m, 2) != 1 || mod(n, 2) != 1
        error("filter dimensions must be odd")
    end
    g = [exp(-(X.^2+Y.^2)/(2*sigma.^2)) for X=-floor(m/2):floor(m/2), Y=-floor(n/2):floor(n/2)]
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

# Sum of squared differences
ssd{T}(A::AbstractArray{T}, B::AbstractArray{T}) = sum((data(A)-data(B)).^2)

# normalized by Array size
ssdn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = ssd(A, B)/length(A)

# sum of absolute differences
sad{T}(A::AbstractArray{T}, B::AbstractArray{T}) = sum(abs(data(A)-data(B)))

# normalized by Array size
sadn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = sad(A, B)/length(A)

# normalized cross correlation
function ncc{T}(A::AbstractArray{T}, B::AbstractArray{T})
    Am = (data(A).-mean(data(A)))[:]
    Bm = (data(B).-mean(data(B)))[:]
    return dot(Am,Bm)/(norm(Am)*norm(Bm))
end

# Array padding
function padarray{T,n}(img::AbstractArray{T,n}, prepad::Union(Vector{Int},Dims), postpad::Union(Vector{Int},Dims), border::String)
    I = Array(Vector{Int}, n)
    for d = 1:n
        M = size(img, d)
        I[d] = [(1 - prepad[d]):(M + postpad[d])]
        if border == "replicate"
            I[d] = min(max(I[d], 1), M)
        elseif border == "circular"
            I[d] = 1 .+ mod(I[d] .- 1, M)
        elseif border == "symmetric"
            I[d] = [1:M, M:-1:1][1 .+ mod(I[d] .- 1, 2 * M)]
        elseif border == "reflect"
            I[d] = [1:M, M-1:-1:2][1 .+ mod(I[d] .- 1, 2 * M - 2)]
        else
            error("unknown border condition")
        end
    end
    img[I...]::Array{T,n}
end

function padarray{T,n}(img::AbstractArray{T,n}, prepad::Union(Vector{Int},Dims), postpad::Union(Vector{Int},Dims), border::String, value)
    if border != "value"
        return padarray(img, prepad, postpad, border)
    end
    A = Array(T, ntuple(n, d->size(img,d)+prepad[d]+postpad[d]))
    fill!(A, value)
    I = Vector{Int}[1+prepad[d]:size(A,d)-postpad[d] for d = 1:n]
    A[I...] = img
    A::typeof(img)
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
imfilter(img, kern, border, value) = copy(img, imfilter_inseparable(img, kern, border, value))
# Do not combine these with the previous using a default value (see the 2d specialization below)
imfilter(img, filter) = imfilter(img, filter, "replicate", 0)
imfilter(img, filter, border) = imfilter(img, filter, border, 0)

imfilter_inseparable{T,K,N,M}(img::AbstractArray{T,N}, kern::AbstractArray{K,M}, border::String, value) =
    imfilter_inseparable(img, prep_kernel(img, kern), border, value)

function imfilter_inseparable{T,K,N}(img::AbstractArray{T,N}, kern::AbstractArray{K,N}, border::String, value)
    prepad  = [div(size(kern,i)-1, 2) for i = 1:N]
    postpad = [div(size(kern,i),   2) for i = 1:N]
    A = padarray(img, prepad, postpad, border, convert(T, value))
    result = _imfilter!(Array(typeof(one(T)*one(K)), size(img)), A, data(kern))
    copy(img, result)
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
    tmp = imfilter_inseparable(img, reshape(u*ss, sz1...), border, value)
    copy(img, imfilter_inseparable(tmp, reshape(v*ss, sz2...), border, value))
end

for N = 1:5
    @eval begin
        function _imfilter!{T,K}(B, A::AbstractArray{T,$N}, kern::AbstractArray{K,$N})
            for i = 1:$N
                if size(B,i)+size(kern,i) > size(A,i)+1
                    throw(DimensionMismatch("Output dimensions $(size(B)) and kernel dimensions $(size(kern)) do not agree with size of padded input, $(size(A))"))
                end
            end
            @nloops $N i B begin
                tmp = zero(T)
                @inbounds begin
                    @nloops $N j kern begin
                        tmp += (@nref $N A d->(i_d+j_d-1))*(@nref $N kern j)
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
imfilter_fft(img, kern, border, value) = copy(img, imfilter_fft_inseparable(img, kern, border, value))
imfilter_fft(img, filter) = imfilter_fft(img, filter, "replicate", 0)
imfilter_fft(img, filter, border) = imfilter_fft(img, filter, border, 0)

imfilter_fft_inseparable{T,K,N,M}(img::AbstractArray{T,N}, kern::AbstractArray{K,M}, border::String, value) =
    imfilter_fft_inseparable(img, prep_kernel(img, kern), border, value)

function imfilter_fft_inseparable{T,K,N}(img::AbstractArray{T,N}, kern::AbstractArray{K,N}, border::String, value)
    prepad  = [div(size(kern,i)-1, 2) for i = 1:N]
    postpad = [div(size(kern,i),   2) for i = 1:N]
    fullpad = [nextprod([2,3], size(img,i) + prepad[i] + postpad[i]) - size(img, i) - prepad[i] for i = 1:N]
    A = padarray(img, prepad, fullpad, border, convert(T, value))
    krn = zeros(typeof(one(T)*one(K)), size(A))
    indexesK = ntuple(N, d->[size(krn,d)-prepad[d]+1:size(krn,d),1:size(kern,d)-prepad[d]])
    krn[indexesK...] = rot180(kern)
    AF = ifft(fft(A).*fft(krn))
    out = Array(T, size(img))
    indexesA = ntuple(N, d->postpad[d]+1:size(img,d)+postpad[d])
    copyreal!(out, AF, indexesA)
    out
end

for N = 1:5
    @eval begin
        function copyreal!{T<:Real}(dst::Array{T,$N}, src, I::(Range1{Int}...))
            @nexprs $N d->(I_d = I[d])
            @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                (@nref $N dst i) = real(@nref $N src j)
            end
            dst
        end
        function copyreal!{T<:Complex}(dst::Array{T,$N}, src, I::(Range1{Int}...))
            @nexprs $N d->I_d = I[d]
            @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                (@nref $N dst i) = @nref $N src j
            end
            dst
        end
    end
end

function imfilter_fft_old{T}(img::StridedMatrix{T}, filter::Matrix{T}, border::String, value)
    si, sf = size(img), size(filter)
    fw = iceil(([sf...] - 1) / 2)
    A = padarray(img, fw, fw, border, convert(T, value))
    # correlation instead of convolution
    filter = rot180(filter)
    # check if separable
    SVD = svdfact(filter)
    U, S, Vt = SVD[:U], SVD[:S], SVD[:Vt]
    separable = true
    for i = 2:length(S)
        separable &= (abs(S[i]) < sqrt(eps(T)))
    end
    if separable
        # conv2 isn't suitable for this (kernel center should be the actual center of the kernel)
        y = U[:,1]*sqrt(S[1])
        x = vec(Vt[1,:])*sqrt(S[1])
        sa = size(A)
        m = length(y)+sa[1]
        n = length(x)+sa[2]
        B = zeros(T, m, n)
        B[int(length(y)/2)+1:sa[1]+int(length(y)/2),int(length(x)/2)+1:sa[2]+int(length(x)/2)] = A
        yp = zeros(T, m)
        halfy = int((m-length(y)-1)/2)
        yp[halfy+1:halfy+length(y)] = y
        y = fft(yp)
        xp = zeros(T, n)
        halfx = int((n-length(x)-1)/2)
        xp[halfx+1:halfx+length(x)] = x
        x = fft(xp)
        C = fftshift(ifft(fft(B) .* (y * x.')))
        if T <: Real
            C = real(C)
        end
    else
        #C = conv2(A, filter)
        sa, sb = size(A), size(filter)
        At = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
        Bt = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
        halfa1 = ifloor((size(At,1)-sa[1])/2)
        halfa2 = ifloor((size(At,2)-sa[2])/2)
        halfb1 = ifloor((size(Bt,1)-sb[1])/2)
        halfb2 = ifloor((size(Bt,2)-sb[2])/2)
        At[halfa1+1:halfa1+sa[1], halfa2+1:halfa2+sa[2]] = A
        Bt[halfb1+1:halfb1+sb[1], halfb2+1:halfb2+sb[2]] = filter
        C = fftshift(ifft(fft(At).*fft(Bt)))
        if T <: Real
            C = real(C)
        end
    end
    sc = size(C)
    out = C[int(sc[1]/2-si[1]/2):int(sc[1]/2+si[1]/2)-1, int(sc[2]/2-si[2]/2):int(sc[2]/2+si[2]/2)-1]
end



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
        A[nanflag] = nan(T)
    else
        imfilter_gaussian_no_nans!(A, sigma; emit_warning=emit_warning)
    end
    share(img, A)
end

function imfilter_gaussian{T<:Integer,TF<:FloatingPoint}(img::AbstractArray{T}, sigma::Vector; emit_warning = true, astype::Type{TF}=Float64)
    A = convert(Array{astype}, data(img))
    if all(sigma .== 0)
        return share(img, A)
    end
    imfilter_gaussian_no_nans!(A, sigma; emit_warning=emit_warning)
    share(img, A)
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
        keepdims = [false,trues(nd-1)]
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
    Ix = padindexes(img, 2, prepad, postpad, border)

    kernleny = length(kh1y)
    prepad  = div(kernleny - 1, 2)
    postpad = div(kernleny, 2)
    Iy = padindexes(img, 1, prepad, postpad, border)

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
    img12 + img21
end

imfilter_LoG{T}(img::AbstractArray{T,2}, σ::Real, border="replicate") = 
imfilter_LoG(img::AbstractArray{T,2}, [σ, σ], border)

function padindexes{T,n}(img::AbstractArray{T,n}, dim, prepad, postpad, border::String)
    M = size(img, dim)
    I = Array(Int, M + prepad + postpad)
    I = [(1 - prepad):(M + postpad)]
    if border == "replicate"
        I = min(max(I, 1), M)
    elseif border == "circular"
        I = 1 .+ mod(I .- 1, M)
    elseif border == "symmetric"
        I = [1:M, M:-1:1][1 .+ mod(I .- 1, 2 * M)]
    elseif border == "reflect"
        I = [1:M, M-1:-1:2][1 .+ mod(I .- 1, 2 * M - 2)]
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

restrict(img::AbstractImageDirect, ::()) = img

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
function restrict(A::AbstractArray, region::Union(Dims, Vector{Int})=coords_spatial(img))
    for dim in region
        A = _restrict(A, dim)
    end
    A
end

function restrict{S<:ByteString}(img::AbstractImageDirect, region::Union((ByteString...), Vector{S}))
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
    out = Array(typeof(A[1]/4+A[2]/2), ntuple(ndims(A), i->i==dim?restrict_size(size(A,dim)):size(A,i)))
    restrict!(out, A, dim)
    out
end

# out should have efficient linear indexing
for N = 1:5
    @eval begin
        function restrict!{T}(out::AbstractArray{T,$N}, A::AbstractArray, dim)
            if isodd(size(A, dim))
                half = convert(T, 0.5)
                quarter = convert(T, 0.25)
                indx = 0
                if dim == 1
                    @nloops $N i d->(d==1 ? (1:1) : (1:size(A,d))) d->(j_d = d==1 ? i_d+1 : i_d) begin
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
                        (@nref $N out i) = 0
                    end
                    stride_1 = 1
                    @nexprs $N d->(stride_{d+1} = stride_d*size(out,d))
                    @nexprs $N d->offset_d = 0
                    ispeak = true
                    @nloops $N i d->(d==1?(1:1):(1:size(A,d))) d->(if d==dim; ispeak=isodd(i_d); offset_{d-1} = offset_d+(div(i_d+1,2)-1)*stride_d; else; offset_{d-1} = offset_d+(i_d-1)*stride_d; end) begin
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
                threeeighths = convert(T, 0.375)
                oneeighth = convert(T, 0.125)
                indx = 0
                if dim == 1
                    @nloops $N i d->(d==1 ? (1:1) : (1:size(A,d))) d->(j_d = i_d) begin
                        c = d = zero(T)
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
                    fill!(out, 0)
                    strd = stride(out, dim)
                    stride_1 = 1
                    @nexprs $N d->(stride_{d+1} = stride_d*size(out,d))
                    @nexprs $N d->offset_d = 0
                    peakfirst = true
                    @nloops $N i d->(d==1?(1:1):(1:size(A,d))) d->(if d==dim; peakfirst=isodd(i_d); offset_{d-1} = offset_d+(div(i_d+1,2)-1)*stride_d; else; offset_{d-1} = offset_d+(i_d-1)*stride_d; end) begin
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

function rgb2ntsc{T}(img::Array{T})
    trans = [0.299 0.587 0.114; 0.596 -0.274 -0.322; 0.211 -0.523 0.312]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = trans * vec(img[i,j,:])
    end
    return out
end

function ntsc2rgb{T}(img::Array{T})
    trans = [1 0.956 0.621; 1 -0.272 -0.647; 1 -1.106 1.703]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = trans * vec(img[i,j,:])
    end
    return out
end

function rgb2ycbcr{T}(img::Array{T})
    trans = [65.481 128.533 24.966; -37.797 -74.203 112; 112 -93.786 -18.214]
    offset = [16.0; 128.0; 128.0]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = offset + trans * vec(img[i,j,:])
    end
    return out
end

function ycbcr2rgb{T}(img::Array{T})
    trans = inv([65.481 128.533 24.966; -37.797 -74.203 112; 112 -93.786 -18.214])
    offset = [16.0; 128.0; 128.0]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = trans * (vec(img[i,j,:]) - offset)
    end
    return out
end

function imcomplement{T}(img::AbstractArray{T})
    return 1 - img
end

function rgb2hsi{T}(img::Array{T})
    R = img[:,:,1]
    G = img[:,:,2]
    B = img[:,:,3]
    H = acos((1/2*(2*R - G - B)) ./ (((R - G).^2 + (R - B).*(G - B)).^(1/2)+eps(T))) 
    H[B .> G] = 2*pi - H[B .> G]
    H /= 2*pi
    rgb_sum = R + G + B
    rgb_sum[rgb_sum .== 0] = eps(T)
    S = 1 - 3./(rgb_sum).*min(R, G, B)
    H[S .== 0] = 0
    I = 1/3*(R + G + B)
    return cat(3, H, S, I)
end

function hsi2rgb{T}(img::Array{T})
    H = img[:,:,1]*(2pi)
    S = img[:,:,2]
    I = img[:,:,3]
    R = zeros(T, size(img,1), size(img,2))
    G = zeros(T, size(img,1), size(img,2))
    B = zeros(T, size(img,1), size(img,2))
    RG = 0 .<= H .< 2*pi/3
    GB = 2*pi/3 .<= H .< 4*pi/3
    BR = 4*pi/3 .<= H .< 2*pi
    # RG sector
    B[RG] = I[RG].*(1 - S[RG])
    R[RG] = I[RG].*(1 + (S[RG].*cos(H[RG]))./cos(pi/3 - H[RG]))
    G[RG] = 3*I[RG] - R[RG] - B[RG]
    # GB sector
    R[GB] = I[GB].*(1 - S[GB])
    G[GB] = I[GB].*(1 + (S[GB].*cos(H[GB] - pi/3))./cos(H[GB]))
    B[GB] = 3*I[GB] - R[GB] - G[GB]
    # BR sector
    G[BR] = I[BR].*(1 - S[BR])
    B[BR] = I[BR].*(1 + (S[BR].*cos(H[BR] - 2*pi/3))./cos(-pi/3 - H[BR]))
    R[BR] = 3*I[BR] - G[BR] - B[BR]
    return cat(3, R, G, B)
end

function imstretch{T}(img::AbstractArray{T}, m::Number, slope::Number)
    share(img, 1./(1 + (m./(data(img) + eps(T))).^slope))
end

function imedge{T}(img::AbstractArray{T}, method::String, border::String)
    # needs more methods
    if method == "sobel"
        s1, s2 = sobel()
        img1 = imfilter(img, s1, border)
        img2 = imfilter(img, s2, border)
        return img1, img2, sqrt(img1.^2 + img2.^2), atan2(img2, img1)
    elseif method == "prewitt"
        s1, s2 = prewitt()
        img1 = imfilter(img, s1, border)
        img2 = imfilter(img, s2, border)
        return img1, img2, sqrt(img1.^2 + img2.^2), atan2(img2, img1)
    end
end

imedge{T}(img::AbstractArray{T}, method::String) = imedge(img, method, "replicate")
imedge{T}(img::AbstractArray{T}) = imedge(img, "sobel", "replicate")

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
        out = share(img, imROF(data(img), lambda, iterations))
    end
    out
end


### Morphological operations

# Erode and dilate support 3x3 regions only (and higher-dimensional generalizations).
dilate(img::AbstractArray, region=coords_spatial(img)) = dilate!(copy(img), region)
erode(img::AbstractArray, region=coords_spatial(img)) = erode!(copy(img), region)

dilate!(maxfilt, region=coords_spatial(maxfilt)) = extremefilt!(data(maxfilt), Base.Order.Forward, region)
erode!(minfilt, region=coords_spatial(minfilt)) = extremefilt!(data(minfilt), Base.Order.Reverse, region)
function extremefilt!(extrfilt::Array, order::Ordering, region=coords_spatial(extrfilt))
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

extr(order::Ordering, x::ColorValue, y::ColorValue) = extr(order, convert(RGB, x), convert(RGB, y))
extr(order::Ordering, x::ColorValue, y::ColorValue, z::ColorValue) = extr(order, convert(RGB, x), convert(RGB, y), convert(RGB, z))


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
