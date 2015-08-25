#### Types and constructors ####

# Plain arrays can be treated as images. Other types will have
# metadata associated, make yours a child of one of the following:
abstract AbstractImage{T,N} <: AbstractArray{T,N}         # image with metadata
abstract AbstractImageDirect{T,N} <: AbstractImage{T,N}   # each pixel has own value/color
abstract AbstractImageIndexed{T,N} <: AbstractImage{T,N}  # indexed images (i.e., lookup table)

# Direct image (e.g., grayscale, RGB)
type Image{T,N,A<:AbstractArray} <: AbstractImageDirect{T,N}
    data::A
    properties::Dict{ASCIIString,Any}
end
Image(data::AbstractArray, props::Dict) = Image{eltype(data),ndims(data),typeof(data)}(data,props)
Image(data::AbstractArray; kwargs...) = Image(data, kwargs2dict(kwargs))

# Indexed image (colormap)
type ImageCmap{T<:Colorant,N,A<:AbstractArray} <: AbstractImageIndexed{T,N}
    data::A
    cmap::Vector{T}
    properties::Dict{ASCIIString,Any}
end
ImageCmap(data::AbstractArray, cmap::AbstractVector, props::Dict) = ImageCmap{eltype(cmap),ndims(data),typeof(data)}(data, cmap, props)
ImageCmap(data::AbstractArray, cmap::AbstractVector; kwargs...) = ImageCmap(data, cmap, kwargs2dict(kwargs))

# Convenience constructors
grayim(A::AbstractImage) = A
grayim(A::AbstractArray{Uint8,2})  = grayim(reinterpret(Ufixed8, A))
grayim(A::AbstractArray{Uint16,2}) = grayim(reinterpret(Ufixed16, A))
grayim(A::AbstractArray{Uint8,3})  = grayim(reinterpret(Ufixed8, A))
grayim(A::AbstractArray{Uint16,3}) = grayim(reinterpret(Ufixed16, A))
grayim{T}(A::AbstractArray{T,2}) = Image(A; colorspace="Gray", spatialorder=["x","y"])
grayim{T}(A::AbstractArray{T,3}) = Image(A; colorspace="Gray", spatialorder=["x","y","z"])

colorim(A::AbstractImage) = A
function colorim{T}(A::AbstractArray{T,3})
    if size(A, 1) == 4 || size(A, 3) == 4
        error("The array looks like a 4-channel color image. Please specify the colorspace explicitly (e.g. \"ARGB\" or \"RGBA\".)")
    end

    colorim(A, "RGB")
end

function colorim{T}(A::AbstractArray{T,3}, colorspace)
    if 3 <= size(A, 1) <= 4 && 3 <= size(A, 3) <= 4
        error("Both first and last dimensions are of size 3 or 4; impossible to guess which is for color. Use the Image constructor directly.")
    elseif 3 <= size(A, 1) <= 4  # Image as returned by imread for regular 2D RGB images
        if T<:Fractional
            CT = getcolortype(colorspace, eltype(A))
            Image(reinterpret(CT, A); spatialorder=["x","y"])
        else
            Image(A; colorspace=colorspace, colordim=1, spatialorder=["x","y"])
        end
    elseif 3 <= size(A, 3) <= 4  # "Matlab"-style image, as returned by convert(Array, im).
        Image(A; colorspace=colorspace, colordim=3, spatialorder=["y","x"])
    else
        error("Neither the first nor the last dimension is of size 3. This doesn't look like an RGB image.")
    end
end

colorim(A::AbstractArray{Uint8,3})  = colorim(reinterpret(Ufixed8, A))
colorim(A::AbstractArray{Uint16,3}) = colorim(reinterpret(Ufixed16, A))
colorim(A::AbstractArray{Uint8,3},  colorspace) = colorim(reinterpret(Ufixed8, A), colorspace)
colorim(A::AbstractArray{Uint16,3}, colorspace) = colorim(reinterpret(Ufixed16, A), colorspace)


#### Core operations ####

eltype{T}(img::AbstractImage{T}) = T

size(img::AbstractImage) = size(img.data)
size(img::AbstractImage, i::Integer) = size(img.data, i)
size(img::AbstractImage, dimname::String) = size(img.data, dimindex(img, dimname))

ndims(img::AbstractImage) = ndims(img.data)

strides(img::AbstractImage) = strides(img.data)

copy(img::Image) = Image(copy(img.data), deepcopy(img.properties))
copy(img::ImageCmap) = ImageCmap(copy(img.data), copy(img.cmap), deepcopy(img.properties))

# Create a new "Image" (could be just an Array) copying the properties but replacing the data
copyproperties(img::AbstractArray, data::AbstractArray) = data

copyproperties(img::AbstractImageDirect, data::AbstractArray) = Image(data, deepcopy(img.properties))

copyproperties(img::AbstractImageIndexed, data::AbstractArray) = ImageCmap(data, copy(img.cmap), deepcopy(img.properties))

copyproperties(img::AbstractImageDirect, _data::AbstractImageDirect) = copyproperties(img, data(_data))

# Provide new data but reuse the properties & cmap
shareproperties(img::AbstractArray, data::AbstractArray) = data

shareproperties(img::AbstractImageDirect, data::AbstractArray) = Image(data, img.properties)

shareproperties(img::AbstractImageIndexed, data::AbstractArray) = ImageCmap(data, img.cmap, img.properties)

# similar
similar(img::AbstractImageDirect) = Image(similar(img.data), copy(img.properties))
similar(img::AbstractImageDirect, ::NTuple{0}) = Image(similar(img.data), copy(img.properties))

similar(img::AbstractImageDirect, dims::Dims) = Image(similar(img.data, dims), copy(img.properties))

similar{T}(img::AbstractImageDirect, ::Type{T}) = Image(similar(img.data, T), copy(img.properties))

similar{T}(img::AbstractImageDirect, ::Type{T}, dims::Dims) = Image(similar(img.data, T, dims), copy(img.properties))

similar(img::AbstractImageIndexed) = ImageCmap(similar(img.data), copy(img.cmap), copy(img.properties))
similar(img::AbstractImageIndexed, ::NTuple{0}) = ImageCmap(similar(img.data), copy(img.cmap), copy(img.properties))

similar(img::AbstractImageIndexed, dims::Dims) = ImageCmap(similar(img.data, dims), copy(img.cmap), copy(img.properties))

similar{T}(img::AbstractImageIndexed, ::Type{T}) = ImageCmap(similar(img.data, T), copy(img.cmap), copy(img.properties))

similar{T}(img::AbstractImageIndexed, ::Type{T}, dims::Dims) = ImageCmap(similar(img.data, T, dims), copy(img.cmap), copy(img.properties))

# copy properties
function copy!(imgdest::AbstractImage, imgsrc::AbstractImage, prop1::ASCIIString, props::ASCIIString...)
    imgdest[prop1] = imgsrc[prop1]
    for p in props
        imgdest[p] = imgsrc[p]
    end
    imgdest
end


## reinterpret: Color->T
# Arrays
reinterpret{CV1<:Colorant,CV2<:Colorant}(::Type{CV1}, A::Array{CV2,1}) = _reinterpret_cvarray(CV1, A)
reinterpret{CV1<:Colorant,CV2<:Colorant}(::Type{CV1}, A::Array{CV2})   = _reinterpret_cvarray(CV1, A)
reinterpret{T,CV<:Colorant}(::Type{T}, A::Array{CV,1}) = _reinterpret_cvarray(T, A)
reinterpret{T,CV<:Colorant}(::Type{T}, A::Array{CV})   = _reinterpret_cvarray(T, A)
reinterpret{T,CV<:Colorant}(::Type{T}, A::StridedArray{CV})   = slice(_reinterpret_cvarray(T, A.parent), A.indexes...)
function _reinterpret_cvarray{T,CV<:Colorant}(::Type{T}, A::Array{CV})
    if sizeof(T) == sizeof(CV)
        return reinterpret(T, A, size(A))
    elseif sizeof(T)*length(CV) == sizeof(CV)
        return reinterpret(T, A, tuple(length(CV), size(A)...))
    end
    error("result shape not specified")
end
reinterpret{CV<:Colorant}(A::StridedArray{CV}) = reinterpret(eltype(CV), A)

# Images
reinterpret{CV1<:Colorant,CV2<:Colorant}(::Type{CV1}, img::AbstractImageDirect{CV2}) =
    shareproperties(img, reinterpret(CV1, data(img)))
function reinterpret{CV<:Colorant}(::Type{Uint32}, img::AbstractImageDirect{CV})
    CV <: Union(RGB24, ARGB32) || (CV <: AbstractRGB && sizeof(CV) == 4) || error("Can't convert $CV to Uint32")
    A = reinterpret(Uint32, data(img))
    props = copy(properties(img))
    props["colorspace"] = colorspace(img)
    Image(A, props)
end
function reinterpret{T,CV2<:Colorant}(::Type{T}, img::AbstractImageDirect{CV2})
    A = reinterpret(T, data(img))
    props = copy(properties(img))
    props["colorspace"] = colorspace(img)
    if ndims(A) > ndims(img)
        props["colordim"] = 1
    end
    Image(A, props)
end

## reinterpret: T->Color
# We have to distinguish two forms of call:
#   form 1: reinterpret(RGB, img)
#   form 2: reinterpret(RGB{Ufixed8}, img)
# Arrays
reinterpret{T,CV<:Colorant}(::Type{CV}, A::Array{T,1}) = _reinterpret(CV, eltype(CV), A)
reinterpret{T,CV<:Colorant}(::Type{CV}, A::Array{T})   = _reinterpret(CV, eltype(CV), A)
_reinterpret{T,CV<:Colorant}(::Type{CV}, ::Type{Any}, A::Array{T}) =
    _reinterpret_array_cv(CV{T}, A)   # form 1 (turn into a form 2 call by filling in the element type of the array)
_reinterpret{T,CV<:Colorant}(::Type{CV}, TT::DataType, A::Array{T}) =
    _reinterpret_array_cv(CV, A)    # form 2
function _reinterpret_array_cv{T,CV<:Colorant}(::Type{CV}, A::Array{T})
    if sizeof(T) == sizeof(CV)
        return reinterpret(CV, A, size(A))
    elseif sizeof(T)*size(A,1) == sizeof(CV)
        return reinterpret(CV, A, size(A)[2:end])
    end
    error("result shape not specified")
end
# This version is used by the deserializer to convert UInt8 buffers back to their original type. Fixes #287.
_reinterpret_array_cv{CV<:Colorant}(::Type{CV}, A::Vector{UInt8}) =
    reinterpret(CV, A, (div(length(A), sizeof(CV)),))

# Images
function reinterpret{T,CV<:Colorant}(::Type{CV}, img::AbstractImageDirect{T})
    A = reinterpret(CV, data(img))
    props = copy(properties(img))
    haskey(props, "colorspace") && delete!(props, "colorspace")
    haskey(props, "colordim") && delete!(props, "colordim")
    Image(A, props)
end

# T->S
function reinterpret{T,S}(::Type{T}, img::AbstractImageDirect{S})
    if sizeof(S) != sizeof(T)
        error("result shape not specified")
    end
    shareproperties(img, reinterpret(T, data(img)))
end

## To get data in raw format, and unwrap UfixedBase if present
function raw(img::AbstractArray)
    elemType = eltype(eltype(data(img)))

    if (elemType <: None)  # weird fallback case
        data(img)
    elseif eltype(eltype(data(img))) <: FixedPointNumbers.UfixedBase
        reinterpret( FixedPointNumbers.rawtype(eltype(eltype(img))), data(img) )
    else
        data(img)
    end
end


## convert
convert{T}(::Type{Image{T}}, img::Image{T}) = img
convert(::Type{Image}, img::Image) = img
convert(::Type{Image}, A::AbstractArray) = Image(A, properties(A))
# Convert an indexed image (cmap) to a direct image
function convert(::Type{Image}, img::ImageCmap)
    data = reshape(img.cmap[vec(img.data)], size(img.data))
    Image(data, copy(properties(img)))
end
# Convert an Image to an array. We convert the image into the canonical storage order convention for arrays.
# We restrict this to 2d images because for plain arrays this convention exists only for 2d.
# In other cases---or if you don't want the storage order altered---just use data(img)
convert{T<:Real,N}(::Type{Array{T}}, img::AbstractImageDirect{T,N}) = convert(Array{T,N}, img)
convert{T<:Colorant,N}(::Type{Array{T}}, img::AbstractImageDirect{T,N}) = convert(Array{T,N}, img)
convert{T}(::Type{Vector{T}}, img::AbstractImageDirect{T,1}) = convert(Vector{T}, data(img))
function convert{T,N}(::Type{Array{T,N}}, img::AbstractImageDirect{T,N})
    assert2d(img)  # only well-defined in 2d
    p = permutation_canonical(img)
    dat = convert(Array{T}, data(img))
    if issorted(p)
        return dat
    else
        return permutedims(dat, p)
    end
end
convert(::Type{Array}, img::AbstractImage) = convert(Array{eltype(img)}, img)

convert{C<:Colorant}(::Type{Image{C}}, img::Image{C}) = img
if !(VERSION < v"0.4.0-dev")
    convert{Cdest<:Colorant,Csrc<:Colorant}(::Type{Image{Cdest}}, img::Image{Csrc}) =
        copyproperties(img, _convert(Array{Cdest}, data(img)))  # FIXME when Julia issue ?? is fixed
end
convert{Cdest<:Colorant,Csrc<:Colorant}(::Type{Image{Cdest}}, img::AbstractImageDirect{Csrc}) =
    copyproperties(img, _convert(Array{Cdest}, data(img)))  # FIXME when Julia issue ?? is fixed
_convert{Cdest<:Colorant,Csrc<:Colorant,N}(::Type{Array{Cdest}}, img::AbstractArray{Csrc,N}) =
    _convert(Array{Cdest}, eltype(Cdest), img)     # FIXME when Julia issue ?? is fixed
_convert{Cdest<:Colorant,Csrc<:Colorant}(::Type{Array{Cdest}}, ::Type{Any}, img::AbstractArray{Csrc}) =
    convert(Array{Cdest{eltype(Csrc)}}, img)
_convert{Cdest<:Colorant,Csrc<:Colorant}(::Type{Array{Cdest}}, ::DataType, img::AbstractArray{Csrc}) =
    convert(Array{Cdest}, img)
convert{Cdest<:Colorant,Csrc<:Colorant,N}(::Type{Array{Cdest}}, img::AbstractImageDirect{Csrc,N}) =
    _convert(Array{Cdest}, convert(Array{Csrc,N}, img))

# Image{Colorant} -> Image{Numbers}
function separate{CV<:Colorant}(img::AbstractImage{CV})
    p = permutation_canonical(img)
    so = spatialorder(img)[p]
    T = eltype(CV)
    if length(CV) > 1
        A = permutedims(reinterpret(T, data(img), tuple(length(CV), size(img)...)), [p+1;1])
    else
        A = permutedims(reinterpret(T, data(img), size(img)), p)
    end
    props = copy(properties(img))
    props["colorspace"] = colorspace(img)
    props["colordim"] = ndims(A)
    props["spatialorder"] = so
    Image(A, props)
end
function separate{CV<:Colorant}(A::AbstractArray{CV})
    T = eltype(CV)
    permutedims(reinterpret(T, A, tuple(length(CV), size(A)...)), [2:ndims(A)+1;1])
end
separate(A::AbstractArray) = A

# Image{Numbers} -> Image{Colorant} (the opposite of separate)
convert{C<:Colorant,T<:Fractional}(::Type{Image{C}}, img::Union(AbstractArray{T},AbstractImageDirect{T})) =
    _convert(Image{C}, eltype(C), img)
_convert{C<:Colorant,T<:Fractional}(::Type{Image{C}}, ::Type{Any}, img::Union(AbstractArray{T},AbstractImageDirect{T})) =
    _convert(Image{C{T}}, img)
_convert{C<:Colorant,T<:Fractional}(::Type{Image{C}}, ::DataType, img::Union(AbstractArray{T},AbstractImageDirect{T})) =
    _convert(Image{C}, img)
function _convert{C<:Colorant,T<:Fractional}(::Type{Image{C}}, img::Union(AbstractArray{T},AbstractImageDirect{T}))
    cd = colordim(img)
    if cd > 0
        p = [cd; setdiff(1:ndims(img), cd)]
        A = permutedims(data(img), p)
    else
        A = data(img)
    end
    CV = getcolortype(colorspace(img), T)
    ACV = convert(Array{C}, reinterpret(CV, A))
    props = copy(properties(img))
    haskey(props, "colordim") && delete!(props, "colordim")
    haskey(props, "colorspace") && delete!(props, "colorspace")
    Image(ACV, props)
end


# Indexing. In addition to conventional array indexing, support syntax like
#    img["x", 100:400, "t", 32]
# where anything not mentioned by name is taken to include the whole range

typealias RealIndex{T<:Real} Union(T, AbstractArray{T})

# setindex!
setindex!(img::AbstractImage, X, i::Real) = setindex!(img.data, X, i)
setindex!(img::AbstractImage, X, I::RealIndex) = setindex!(img.data, X, I)
setindex!(img::AbstractImage, X, I::RealIndex, J::RealIndex) = setindex!(img.data, X, I, J)
setindex!(img::AbstractImage, X, I::RealIndex, J::RealIndex, K::RealIndex) = setindex!(img.data, X, I, J, K)
setindex!(img::AbstractImage, X, I::RealIndex, J::RealIndex,
                   K::RealIndex, L::RealIndex) = setindex!(img.data, X, I, J, K, L)
setindex!(img::AbstractImage, X, I::RealIndex...) = setindex!(img.data, X, I...)
setindex!(img::AbstractImage, X, dimname::String, ind::RealIndex, nameind...) = setindex!(img.data, X, coords(img, dimname, ind, nameind...)...)

# Adding a new property via setindex!
setindex!(img::AbstractImage, X, propname::String) = setindex!(img.properties, X, propname)

# Delete a property!
delete!(img::AbstractImage, propname::String) = delete!(img.properties, propname)


# getindex, sub, and slice return a value or AbstractArray, not an Image
getindex(img::AbstractImage, i::Real) = getindex(img.data, i)
getindex(img::AbstractImage, I::RealIndex) = getindex(img.data, I)
getindex(img::AbstractImage, I::RealIndex, J::RealIndex) = getindex(img.data, I, J)
getindex(img::AbstractImage, I::RealIndex, J::RealIndex, K::RealIndex) = getindex(img.data, I, J, K)
getindex(img::AbstractImage, I::RealIndex, J::RealIndex,
            K::RealIndex, L::RealIndex) = getindex(img.data, I, J, K, L)
getindex(img::AbstractImage, I::RealIndex...) = getindex(img.data, I...)

# getindex(img::AbstractImage, dimname::String, ind::RealIndex, nameind...) = getindex(img.data, coords(img, dimname, ind, nameind...)...)
getindex(img::AbstractImage, dimname::ASCIIString, ind, nameind...) = getindex(img.data, coords(img, dimname, ind, nameind...)...)

getindex(img::AbstractImage, propname::ASCIIString) = getindex(img.properties, propname)

sub(img::AbstractImage, I::RangeIndex...) = sub(img.data, I...)

sub(img::AbstractImage, dimname::ASCIIString, ind::RangeIndex, nameind::RangeIndex...) = sub(img.data, coords(img, dimname, ind, nameind...)...)

slice(img::AbstractImage, I::RangeIndex...) = slice(img.data, I...)

slice(img::AbstractImage, dimname::ASCIIString, ind::RangeIndex, nameind::RangeIndex...) = slice(img.data, coords(img, dimname, ind, nameind...)...)

# getindexim, subim, and sliceim return an Image. The first two share properties, the last requires a copy.
function getindexim(img::AbstractImage, I::RealIndex...)
    ret = copyproperties(img, data(img)[I...])
    cd = colordim(img)
    nd = ndims(ret)
    if cd > nd || (cd > 0 && length(I[cd]) < size(img, cd))
        ret["colorspace"] = "Unknown"
        if cd > nd
            ret["colordim"] = 0
        end
    end
    td = timedim(img)
    if td > nd
        ret["timedim"] = 0
    end
    sp = spatialproperties(img)
    for p in sp
        val = ret[p]
        if length(val) > nd
            ret[p] = val[1:nd]
        end
    end
    ret
end

getindexim(img::AbstractImage, dimname::ASCIIString, ind::Union(Real,AbstractArray), nameind...) = getindexim(img, coords(img, dimname, ind, nameind...)...)

subim(img::AbstractImage, I::RangeIndex...) = _subim(img, I)
_subim{TT}(img, I::TT) = shareproperties(img, sub(img.data, I...))  # work around #8504

subim(img::AbstractImage, dimname::ASCIIString, ind::RangeIndex, nameind...) = subim(img, coords(img, dimname, ind, nameind...)...)

sliceim(img::AbstractImage, I::RangeIndex...) = _sliceim(img, I)
function _sliceim{IT}(img::AbstractImage, I::IT)
    dimmap = Array(Int, ndims(img))
    n = 0
    for j = 1:ndims(img)
        if !isa(I[j], Int); n += 1; end;
        dimmap[j] = n
    end
    S = slice(img.data, I...)
    ret = copyproperties(img, S)
    cd = colordim(img)
    if cd > 0
        if isa(I[cd], Int)
            ret.properties["colordim"] = 0
            ret.properties["colorspace"] = "Unknown"
        else
            ret.properties["colordim"] = dimmap[cd]
            if I[cd] != 1:size(img, cd)
                ret.properties["colorspace"] = "Unknown"
            end
        end
    end
    td = timedim(img)
    if td > 0
        ret.properties["timedim"] = isa(I[td], Int) ? 0 : dimmap[td]
    end
    sp = spatialproperties(img)
    if !isempty(sp)
        c = coords_spatial(img)
        keep = Bool[map(x -> isa(x, AbstractVector), I[c])...]
        if !all(keep)
            for pname in sp
                p = img.properties[pname]
                if isa(p, Vector)
                    ret.properties[pname] = p[keep]
                elseif isa(p, Matrix)
                    ret.properties[pname] = p[keep, keep]
                else
                    error("Do not know how to handle property ", pname)
                end
            end
        end
    end
    ret
end

sliceim(img::AbstractImage, dimname::String, ind::RangeIndex, nameind...) = subim(img, coords(img, dimname, ind, nameind...)...)

sliceim(img::AbstractImage, dimname::String, ind::RangeIndex, nameind...) = sliceim(img, coords(img, dimname, ind, nameind...)...)

subim(img::AbstractImage, I::AbstractVector...) = error("Indexes must be integers or ranges")
sliceim(img::AbstractImage, I::AbstractVector...) = error("Indexes must be integers or ranges")

# Support colon indexes
getindexim(img::AbstractImage, I...) = getindexim(img, ntuple(i-> isa(I[i], Colon) ? (1:size(img,i)) : I[i], length(I))...)
subim(img::AbstractImage, I...) = subim(img, ntuple(i-> isa(I[i], Colon) ? (1:size(img,i)) : I[i], length(I))...)
sliceim(img::AbstractImage, I...) = sliceim(img, ntuple(i-> isa(I[i], Colon) ? (1:size(img,i)) : I[i], length(I))...)


# Iteration
# Defer to the array object in case it has special iteration defined
if VERSION >= v"0.4.0-dev+1623"
    next{T,N}(img::AbstractImage{T,N}, s::(@compat Tuple{Bool,Base.IteratorsMD.CartesianIndex{N}})) = next(data(img), s)
    done{T,N}(img::AbstractImage{T,N}, s::(@compat Tuple{Bool,Base.IteratorsMD.CartesianIndex{N}})) = done(data(img), s)
end
start(img::AbstractImage) = start(data(img))
next(img::AbstractImage, s) = next(data(img), s)
done(img::AbstractImage, s) = done(data(img), s)


# We'll frequently want to pull out different 2d slices from the same image, so here's a type and set of functions making that easier.
# We deliberately do not require the user to specify the full list of new slicing/ranging parameters, as often we'll want to change some aspects (e.g., z-slice) but not others (e.g., color coordinates)
type SliceData
    slicedims::(@compat Tuple{Vararg{Int}})
    slicestrides::(@compat Tuple{Vararg{Int}})
    rangedims::(@compat Tuple{Vararg{Int}})

    function SliceData(A::AbstractArray, slicedims::Int...)
        keep = trues(ndims(A))
        for i = 1:length(slicedims)
            keep[slicedims[i]] = false
        end
        s = strides(A)
        new(slicedims, ntuple(i->s[slicedims[i]], length(slicedims)), tuple((1:ndims(A))[keep]...))
    end
end

SliceData(img::AbstractImage, dimname::String, dimnames::String...) = SliceData(img, dimindexes(img, dimname, dimnames...)...)

function _slice(A::AbstractArray, sd::SliceData, I::Int...)
    if length(I) != length(sd.slicedims)
        throw(BoundsError())
    end
    indexes = RangeIndex[1:size(A, i) for i = 1:ndims(A)]
    for i = 1:length(I)
        indexes[sd.slicedims[i]] = I[i]
    end
    indexes
end

function slice(A::AbstractArray, sd::SliceData, I::Int...)
    indexes = _slice(A, sd, I...)
    slice(A, indexes...)
end

function sliceim(img::AbstractImage, sd::SliceData, I::Int...)
    indexes = _slice(img, sd, I...)
    sliceim(img, indexes...)
end

function first_index(A::SubArray, sd::SliceData)
    newfirst = 1
    for i = 1:length(sd.slicedims)
        newfirst += (A.indexes[sd.slicedims[i]]-1)*sd.slicestrides[i]
    end
    strds = strides(A)
    for i = 1:length(sd.rangedims)
        newfirst += (A.indexes[sd.rangedims[i]][1]-1)*strds[i]
    end
    newfirst
end

function reslice!(A::SubArray, sd::SliceData, I::Int...)
    indexes = RangeIndex[A.indexes...]
    for i = 1:length(I)
        indexes[sd.slicedims[i]] = I[i]
    end
    A.indexes = tuple(indexes...)
    A.first_index = first_index(A, sd)
    A
end

function reslice!(img::AbstractImage, sd::SliceData, I::Int...)
    reslice!(img.data, sd, I...)
    img
end

function rerange!(A::SubArray, sd::SliceData, I::(@compat Tuple{Vararg{RangeIndex}}))
    indexes = RangeIndex[A.indexes...]
    for i = 1:length(I)
        indexes[sd.rangedims[i]] = I[i]
    end
    A.indexes = tuple(indexes...)
    A.first_index = first_index(A, sd)
    A
end

function rerange!(img::AbstractImage, sd::SliceData, I::(@compat Tuple{Vararg{RangeIndex}}))
    rerange!(img.data, sd, I...)
    img
end


const emptyset = Set()
function showim(io::IO, img::AbstractImageDirect)
    IT = typeof(img)
    print(io, colorspace(img), " ", IT.name, " with:\n  data: ", summary(img.data), "\n  properties:")
    showdictlines(io, img.properties, get(img, "suppress", emptyset))
end
function showim(io::IO, img::AbstractImageIndexed)
    IT = typeof(img)
    print(io, colorspace(img), " ", IT.name, " with:\n  data: ", summary(img.data), "\n  cmap: ", summary(img.cmap), "\n  properties:")
    showdictlines(io, img.properties, get(img, "suppress", emptyset))
end
show(io::IO, img::AbstractImageDirect) = showim(io, img)
writemime(io::IO, ::MIME"text/plain", img::AbstractImageDirect) = showim(io, img)
show(io::IO, img::AbstractImageIndexed) = showim(io, img)
writemime(io::IO, ::MIME"text/plain", img::AbstractImageIndexed) = showim(io, img)

data(img::AbstractArray) = img
data(img::AbstractImage) = img.data

minimum(img::AbstractImageDirect) = minimum(img.data)
maximum(img::AbstractImageDirect) = maximum(img.data)
# min/max deliberately not defined for AbstractImageIndexed

function _squeeze(img::AbstractImage, dims)
    imgret = copyproperties(img, squeeze(data(img), dims))
    td = timedim(img)
    if td > 0
        imgret["timedim"] = squeezedims(td, dims)
    end
    cd = colordim(img)
    if cd > 0
        imgret["colordim"] = squeezedims(cd, dims)
    end
    c = coords_spatial(img)
    keep = setdiff(c, dims)
    if length(keep) < length(c)
        sp = spatialproperties(img)
        if !isempty(sp)
            for pname in sp
                p = img.properties[pname]
                if isa(p, Vector)
                    imgret.properties[pname] = p[keep]
                elseif isa(p, Matrix)
                    imgret.properties[pname] = p[keep, keep]
                else
                    error("Do not know how to handle property ", pname)
                end
            end
        end
    end
    imgret
end
squeeze(img::AbstractImage, dims::Integer) = _squeeze(img, dims)
squeeze(img::AbstractImage, dims::Dims) = _squeeze(img, dims)
squeeze(img::AbstractImage, dims) = _squeeze(img, dims)

function squeezedims(val, dims)
    if in(val, dims)
        val = 0
    else
        dec = 0
        for d in dims
            dec += val > d
        end
        val -= dec
    end
    val
end

#### Properties ####

# Generic programming with images uses properties to obtain information. The strategy is to define a particular property name, and then write an accessor function of the same name. The accessor function provides default behavior for plain arrays and when the property is not defined. Alternatively, use get(img, "propname", default) or haskey(img, "propname") to define your own default behavior.

# You can define whatever properties you want. Here is a list of properties used
# in some algorithms:
#   colorspace: "RGB", "ARGB", "Gray", "Binary", "RGB24", "Lab", "HSV", etc.
#   colordim: the array dimension used to store color information, or 0 if there
#     is no dimension corresponding to color
#   timedim: the array dimension used for time (i.e., sequence), or 0 for single images
#   limits: (minvalue,maxvalue) for this type of image (e.g., (0,255) for Uint8
#     images, even if pixels do not reach these values)
#   pixelspacing: the spacing between adjacent pixels along spatial dimensions
#   spacedirections: the direction of each array axis in physical space (a vector-of-vectors, one per dimension)
#   spatialorder: a string naming each spatial dimension, in the storage order of
#     the data array. Names can be arbitrary, but the choices "x" and "y" have special
#     meaning (horizontal and vertical, respectively, irrespective of storage order).
#     If supplied, you must have one entry per spatial dimension.

properties(A::AbstractArray) = @compat Dict(
    "colorspace" => colorspace(A),
    "colordim" => colordim(A),
    "timedim" => timedim(A),
    "pixelspacing" => pixelspacing(A),
    "spatialorder" => spatialorder(A))
properties{C<:Colorant}(A::AbstractArray{C}) = @compat Dict(
    "timedim" => timedim(A),
    "pixelspacing" => pixelspacing(A),
    "spatialorder" => spatialorder(A))
properties(img::AbstractImage) = img.properties

haskey(a::AbstractArray, k::String) = false
haskey(img::AbstractImage, k::String) = haskey(img.properties, k)

get(img::AbstractArray, k::String, default) = default
get(img::AbstractImage, k::String, default) = get(img.properties, k, default)

# So that defaults don't have to be evaluated unless they are needed, we also define a @get macro (thanks Toivo Hennington):
macro get(img, k, default)
    quote
        img, k = $(esc(img)), $(esc(k))
        local val
        if !isa(img, AbstractImage)
            val = $(esc(default))
        else
            index = Base.ht_keyindex(img.properties, k)
            val = (index > 0) ? img.properties.vals[index] : $(esc(default))
        end
        val
    end
end

# Using plain arrays, we have to make all sorts of guesses about colorspace and storage order. This can be a big problem for three-dimensional images, image sequences, cameras with more than 16-bits, etc. In such cases use an AbstractImage type.

# Here are the two most important assumptions (see also colorspace below):
defaultarraycolordim = 3
# defaults for plain arrays ("vertical-major")
const yx = ["y", "x"]
# order used in Cairo & most image file formats (with color as the very first dimension)
const xy = ["x", "y"]
spatialorder(::Type{Matrix}) = yx
spatialorder(img::AbstractArray) = (sdims(img) == 2) ? spatialorder(Matrix) : error("Wrong number of spatial dimensions for plain Array, use an AbstractImage type")

isdirect(img::AbstractArray) = true
isdirect(img::AbstractImageDirect) = true
isdirect(img::AbstractImageIndexed) = false

colorspace{C<:Colorant}(img::AbstractVector{C}) = ColorTypes.colorant_string(C)
colorspace{C<:Colorant}(img::AbstractMatrix{C}) = ColorTypes.colorant_string(C)
colorspace{C<:Colorant}(img::AbstractArray{C,3}) = ColorTypes.colorant_string(C)
colorspace{C<:Colorant}(img::AbstractImage{C}) = ColorTypes.colorant_string(C)
colorspace{C<:Colorant,T}(img::AbstractArray{TransparentColor{C,T},2}) = (S = ColorTypes.colorant_string(C); S == "Gray" ? "GrayAlpha" : string(S, "A"))
colorspace{C<:Colorant,T}(img::AbstractImage{TransparentColor{C,T}}) = (S = ColorTypes.colorant_string(C); S == "Gray" ? "GrayAlpha" : string(S, "A"))
colorspace(img::AbstractVector{Bool}) = "Binary"
colorspace(img::AbstractMatrix{Bool}) = "Binary"
colorspace(img::AbstractArray{Bool}) = "Binary"
colorspace(img::AbstractArray{Bool,3}) = "Binary"
colorspace(img::AbstractMatrix{Uint32}) = "RGB24"
colorspace(img::AbstractVector) = "Gray"
colorspace(img::AbstractMatrix) = "Gray"
colorspace{T}(img::AbstractArray{T,3}) = (size(img, defaultarraycolordim) == 3) ? "RGB" : error("Cannot infer colorspace of Array, use an AbstractImage type")
colorspace(img::AbstractImage{Bool}) = "Binary"
colorspace{T,N,A<:AbstractArray}(img::ImageCmap{T,N,A}) = string(T.name.name)
colorspace(img::AbstractImageIndexed) = @get img "colorspace" csinfer(eltype(img.cmap))
colorspace{T}(img::AbstractImageIndexed{T,2}) = @get img "colorspace" csinfer(eltype(img.cmap))
csinfer{C<:Colorant}(::Type{C}) = ColorTypes.colorant_string(C)
csinfer(C) = "Unknown"
colorspace(img::AbstractImage) = get(img.properties, "colorspace", "Unknown")

colorspacedict = Dict{ASCIIString,Any}()
for ACV in (Color, AbstractRGB)
    for CV in subtypes(ACV)
        (length(CV.parameters) == 1 && !(CV.abstract)) || continue
        str = string(CV.name.name)
        colorspacedict[str] = CV
    end
end
function getcolortype{T}(str::ASCIIString, ::Type{T})
    if haskey(colorspacedict, str)
        CV = colorspacedict[str]
        return CV{T}
    else
        if endswith(str, "A")
            CV = colorspacedict[str[1:end-1]]
            return coloralpha(CV){T}
        elseif startswith(str, "A")
            CV = colorspacedict[str[2:end]]
            return alphacolor(CV){T}
        else
            error("colorspace $str not recognized")
        end
    end
end

colordim{C<:Colorant}(img::AbstractVector{C}) = 0
colordim{C<:Colorant}(img::AbstractMatrix{C}) = 0
colordim{C<:Colorant}(img::AbstractArray{C,3}) = 0
colordim{C<:Colorant}(img::AbstractImage{C}) = 0
colordim(img::AbstractVector) = 0
colordim(img::AbstractMatrix) = 0
colordim{T}(img::AbstractImageDirect{T,3}) = get(img, "colordim", 0)::Int
colordim{T}(img::AbstractArray{T,3}) = (size(img, defaultarraycolordim) == 3) ? 3 : 0
colordim(img::AbstractImageDirect) = get(img, "colordim", 0)
colordim(img::AbstractImageIndexed) = 0

timedim(img) = get(img, "timedim", 0)::Int

limits(img::AbstractArray{Bool}) = 0,1
# limits{T<:Integer}(img::AbstractArray{T}) = typemin(T), typemax(T)  # best not to use Integers...
limits{T<:FloatingPoint}(img::AbstractArray{T}) = zero(T), one(T)
limits(img::AbstractImage{Bool}) = 0,1
limits{T}(img::AbstractImageDirect{T}) = get(img, "limits", (zero(T), one(T)))
limits(img::AbstractImageIndexed) = @get img "limits" (minimum(img.cmap), maximum(img.cmap))

pixelspacing{T}(img::AbstractArray{T,3}) = (size(img, defaultarraycolordim) == 3) ? [1.0,1.0] : error("Cannot infer pixelspacing of Array, use an AbstractImage type")
pixelspacing(img::AbstractMatrix) = [1.0,1.0]
pixelspacing{T}(img::AbstractImage{T}) = @get img "pixelspacing" _pixelspacing(img)
function _pixelspacing(img::AbstractImage)
    if haskey(img, "spacedirections")
        sd = img["spacedirections"]
        return [maximum(abs(sd[i])) for i = 1:length(sd)]
    end
    ones(sdims(img))
end

spacedirections(img::AbstractArray) = @get img "spacedirections" _spacedirections(img)
function _spacedirections(img::AbstractArray)
    ps = pixelspacing(img)
    T = eltype(ps)
    nd = length(ps)
    Vector{T}[(tmp = zeros(T, nd); tmp[i] = ps[i]; tmp) for i = 1:nd]
end

spatialorder(img::AbstractImage) = @get img "spatialorder" _spatialorder(img)
_spatialorder(img::AbstractImage) = (sdims(img) == 2) ? spatialorder(Matrix) : error("Cannot guess default spatial order for ", sdims(img), "-dimensional images")

# This is mostly for user information---in code it's generally better to use spatialorder, colordim, and timedim directly
function storageorder(img::AbstractArray)
    so = Array(ASCIIString, ndims(img))
    so[coords_spatial(img)] = spatialorder(img)
    cd = colordim(img)
    if cd != 0
        so[cd] = "color"
    end
    td = timedim(img)
    if td != 0
        so[td] = "t"
    end
    so
end

# number of spatial dimensions in the image
sdims(img) = ndims(img) - (colordim(img) != 0) - (timedim(img) != 0)

# number of time slices
function nimages(img)
    sd = timedim(img)
    if sd > 0
        return size(img, sd)
    else
        return 1
    end
end

# number of array elements used for each pixel/voxel
function ncolorelem(img)
    cd = colordim(img)
    return cd > 0 ? size(img, cd) : 1
end

# indices of spatial coordinates
function coords_spatial(img)
    nd = ndims(img)
    cd = colordim(img)
    td = timedim(img)
    if cd > nd || td > nd
        error("Properties are inconsistent with the array dimensionality")
    end
    ind = [1:nd;]
    if cd > td
        splice!(ind, cd)
        if td > 0
            splice!(ind, td)
        end
    elseif td > cd
        splice!(ind, td)
        if cd > 0
            splice!(ind, cd)
        end
    end
    ind
end

# size of the spatial grid
function size_spatial(img)
    sz = size(img)
    sz[coords_spatial(img)]
end

#### Utilities for writing "simple algorithms" safely ####
# If you don't feel like supporting multiple representations, call these

# Two-dimensional images
function assert2d(img::AbstractArray)
    if sdims(img) != 2
        error("Only two-dimensional images are supported")
    end
    if timedim(img) != 0
        error("Image sequences are not supported")
    end
end

# "Scalar color", either grayscale, RGB24, or an immutable type
function assert_scalar_color(img::AbstractArray)
    if colordim(img) != 0
        error("Only 'scalar color' is supported")
    end
end

# Check that the time dimension, if present, is last
function assert_timedim_last(img::AbstractArray)
    if 0 < timedim(img) < ndims(img)
        error("Time dimension is not last")
    end
end

# Spatial storage order
isyfirst(img::AbstractArray) = spatialorder(img)[1] == "y"
function assert_yfirst(img)
    if !isyfirst(img)
        error("Image must have y as its first dimension")
    end
end
isxfirst(img::AbstractArray) = spatialorder(img)[1] == "x"
function assert_xfirst(img::AbstractArray)
    if !isxfirst(img)
        error("Image must have x as its first dimension")
    end
end



#### Permutations over dimensions ####

# width and height, translating "x" and "y" spatialorder into horizontal and vertical, respectively
function widthheight(img::AbstractArray, p)
    c = coords_spatial(img)
    size(img, c[p[1]]), size(img, c[p[2]])
end
widthheight(img::AbstractArray) = widthheight(img, spatialpermutation(xy, img))

width(img::AbstractArray) = widthheight(img)[1]
height(img::AbstractArray) = widthheight(img)[2]

# Calculate the permutation needed to put the spatial dimensions into a specified order
spatialpermutation(to, img::AbstractArray) = default_permutation(to, spatialorder(img))
function spatialpermutation(to, img::AbstractImage)
    so = spatialorder(img)
    if so != nothing
        return default_permutation(to, so)
    else
        if sdims(img) != 2
            error("Cannot guess default spatialorder when there are more than 2 spatial dimensions")
        end
        return default_permutation(to, yx)
    end
end

# Permute the dimensions of an image, also permuting the relevant properties. If you have non-default properties that are vectors or matrices relative to spatial dimensions, include their names in the list of spatialprops.
permutedims(img::AbstractImage, p::(@compat Tuple{}), spatialprops::Vector = spatialproperties(img)) = img

function permutedims(img::AbstractImage, p::Union(Vector{Int}, (@compat Tuple{Vararg{Int}})), spatialprops::Vector = spatialproperties(img))
    if length(p) != ndims(img)
        error("The permutation must have length equal to the number of dimensions")
    end
    if issorted(p) && length(p) == ndims(img)
        return img   # should we return a copy?
    end
    ip = invperm(to_vector(p))
    cd = colordim(img)
    sd = timedim(img)
    ret = copyproperties(img, permutedims(img.data, p))
    if cd > 0
        ret.properties["colordim"] = ip[cd]
        p = setdiff(p, cd)
    end
    if sd > 0
        ret.properties["timedim"] = ip[sd]
        p = setdiff(p, sd)
    end
    if !isempty(spatialprops)
        ip = sortperm(p)
        for prop in spatialprops
            a = img.properties[prop]
            if isa(a, AbstractVector)
                ret.properties[prop] = a[ip]
            elseif isa(a, AbstractMatrix) && size(a,1) == size(a,2)
                ret.properties[prop] = a[ip,ip]
            else
                error("Do not know how to handle property ", prop)
            end
        end
    end
    ret
end

permutedims{S<:String}(img::AbstractImage, pstr::Union(Vector{S}, (@compat Tuple{Vararg{S}})), spatialprops::Vector = spatialproperties(img)) = permutedims(img, dimindexes(img, pstr...), spatialprops)

function permutation_canonical(img)
    assert2d(img)
    p = spatialpermutation(spatialorder(Matrix), img)
    p = coords_spatial(img)[p]
    cd = colordim(img)
    if cd > 0
        push!(p, cd)
    end
    p
end

# Define the transpose of a 2d image
function ctranspose(img::AbstractImage)
    assert2d(img)
    s = coords_spatial(img)
    p = [1:ndims(img)]
    p[s] = s[2:-1:1]
    permutedims(img, p)
end

# Default list of spatial properties possessed by an image
function spatialproperties(img::AbstractImage)
    if haskey(img, "spatialproperties")
        return img.properties["spatialproperties"]
    end
    spatialprops = ASCIIString[]
    if haskey(img, "spatialorder")
        push!(spatialprops, "spatialorder")
    end
    if haskey(img, "pixelspacing")
        push!(spatialprops, "pixelspacing")
    end
    if haskey(img, "spacedirections")
        push!(spatialprops, "spacedirections")
    end
    spatialprops
end
spatialproperties(img::AbstractVector) = ASCIIString[]  # these are not mutable


#### Low-level utilities ####
function permutation(to, from)
    n = length(to)
    nf = length(from)
    d = Dict([(from[i], i) for i = 1:length(from)])
    ind = Array(Int, max(n, nf))
    for i = 1:n
        ind[i] = get(d, to[i], 0)
    end
    ind[n+1:nf] = n+1:nf
    ind
end

function default_permutation(to, from)
    p = permutation(to, from)
    pzero = p .== 0
    if any(pzero)
        p[pzero] = setdiff(1:length(to), p)
    end
    p
end

function showdictlines(io::IO, dict::Dict, suppress::Set)
    for (k, v) in dict
        if k == "suppress"
            continue
        end
        if !in(k, suppress)
            print(io, "\n    ", k, ": ")
            printdictval(io, v)
        else
            print(io, "\n    ", k, ": <suppressed>")
        end
    end
end

printdictval(io::IO, v) = print(io, v)
function printdictval(io::IO, v::Vector)
    for i = 1:length(v)
        print(io, " ", v[i])
    end
end

# Support indexing via
#    img["t", 32, "x", 100:400]
# where anything not mentioned by name is assumed to include the whole range
function coords(img::AbstractImage, dimname::ASCIIString, ind, nameind...)
    c = Any[1:d for d in size(img)]
    so = spatialorder(img)
    c[require_dimindex(img, dimname, so)] = ind
    for i = 1:2:length(nameind)
        c[require_dimindex(img, nameind[i], so)] = nameind[i+1]
    end
    tuple(c...)
end

# Use keyword arguments
# e.g. coord(x=1:100, y=1:50)
function coords(img::AbstractImage; kwargs...)
    c = Any[1:d for d in size(img)]
    so = spatialorder(img)
    for (k, v) in kwargs
        c[require_dimindex(img, string(k), so)] = v
    end
    tuple(c...)
end


function dimindex(img::AbstractImage, dimname::ASCIIString, so = spatialorder(img))
    n::Int = 0
    if dimname == "color"
        n = colordim(img)
    elseif dimname == "t"
        n = timedim(img)
    else
        cd = colordim(img)
        td = timedim(img)
        j = 1
        tn = 1
        while j <= length(so)
            while tn == cd || tn == td
                tn += 1
            end
            if dimname == so[j]
                n = tn
                break
            end
            tn += 1
            j += 1
        end
    end
    n
end


require_dimindex(img::AbstractImage, dimname, so) = (di = dimindex(img, dimname, so); di > 0 || error("No dimension called ", dimname); di)

dimindexes(img::AbstractImage, dimnames::String...) = Int[dimindex(img, nam, spatialorder(img)) for nam in dimnames]

to_vector(v::AbstractVector) = v
to_vector(v::(@compat Tuple)) = [v...]

# converts keyword argument to a dictionary
function kwargs2dict(kwargs)
    d = Dict{ASCIIString,Any}()
    for (k, v) in kwargs
        d[string(k)] = v
    end
    return d
end
