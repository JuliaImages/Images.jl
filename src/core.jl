#### Types and constructors ####

# Plain arrays can be treated as images. Other types will have
# metadata associated, make yours a child of one of the following:
abstract AbstractImage{T,N} <: AbstractArray{T,N}         # image with metadata
abstract AbstractImageDirect{T,N} <: AbstractImage{T,N}   # each pixel has own value/color
abstract AbstractImageIndexed{T,N} <: AbstractImage{T,N}  # indexed images (i.e., lookup table)

# converts keyword argument to a dictionary
function kwargs2dict(kwargs)
    d = (ASCIIString=>Any)[]
    for (k, v) in kwargs
        d[string(k)] = v
    end
    return d
end

# Direct image (e.g., grayscale, RGB)
type Image{T,N,A<:AbstractArray} <: AbstractImageDirect{T,N}
    data::A
    properties::Dict
end
Image(data::AbstractArray, props::Dict) = Image{eltype(data),ndims(data),typeof(data)}(data,props)
Image(data::AbstractArray; kwargs...) = Image(data, kwargs2dict(kwargs))

# Indexed image (colormap)
type ImageCmap{T,N,A<:AbstractArray,C<:AbstractArray} <: AbstractImageIndexed{T,N}
    data::A
    cmap::C
    properties::Dict
end
ImageCmap(data::AbstractArray, cmap::AbstractArray, props::Dict) = ImageCmap{eltype(data),ndims(data),typeof(data),typeof(cmap)}(data, cmap, props)
ImageCmap(data::AbstractArray, cmap::AbstractArray; kwargs...) = ImageCmap(data, cmap, kwargs2dict(kwargs))

# Convenience constructors
grayim{T}(A::AbstractArray{T,2}) = Image(A; colorspace="Gray", spatialorder=["x","y"])
grayim{T}(A::AbstractArray{T,3}) = Image(A; colorspace="Gray", spatialorder=["x","y","z"])

# Dispatch-based scaling/clipping/type conversion
abstract ScaleInfo{T}

# An array type for colorized overlays of grayscale images
type Overlay{AT<:(AbstractArray...),N,SIT<:(ScaleInfo...)} <: AbstractArray{RGB,N}
    channels::AT   # this holds the grayscale arrays
    colors::Vector{RGB}
    scalei::SIT
    visible::BitVector

    function Overlay(channels::(AbstractArray...), colors, scalei::(ScaleInfo...), visible::BitVector)
        nc = length(channels)
        for i = 1:nc
            if ndims(channels[i]) != N
                error("All arrays must have the same dimensionality")
            end
        end
        sz = size(channels[1])
        for i = 2:nc
            if size(channels[i]) != sz
                error("All arrays must have the same size")
            end
        end
        if length(colors) != nc || length(scalei) != nc || length(visible) != nc
            error("All input must have the same length")
        end
        new(channels, [convert(RGB, c) for c in colors], scalei, visible)
    end
end
Overlay(channels::(AbstractArray...), colors, scalei::(ScaleInfo...), visible::BitVector) =
    Overlay{typeof(channels),ndims(channels[1]),typeof(scalei)}(channels,colors,scalei,visible)
Overlay(channels::(AbstractArray...), colors, scalei::(ScaleInfo...)) =
    Overlay{typeof(channels),ndims(channels[1]),typeof(scalei)}(channels,colors,scalei,trues(length(channels)))

function Overlay(channels::(AbstractArray...), colors, clim)
    n = length(channels)
    for i = 1:n
        if length(clim[i]) != 2
            error("clim must be a 2-vector")
        end
    end
    scalei = ntuple(n, i->scaleminmax(Float64, channels[i], clim[i][1], clim[i][2]))
    Overlay{typeof(channels),ndims(channels[1]),typeof(scalei)}(channels, colors, scalei, trues(n))
end

# Returns the overlay as an image, if possible
function OverlayImage(channels::(AbstractArray...), colors::(ColorValue...), clim)
    ovr = Overlay(channels, colors, clim)
    for i = 1:length(channels)
        if isa(channels[i], AbstractImage)
            prop = copy(properties(channels[i]))
            delete!(prop, "colorspace")
            return Image(ovr, prop)
        end
    end
    ovr
end

#### Core operations ####

eltype{T}(img::AbstractImage{T}) = T
eltype{T}(::Type{AbstractImage{T}}) = T

size(img::AbstractImage) = size(img.data)
size(img::AbstractImage, i::Integer) = size(img.data, i)
size(img::AbstractImage, dimname::String) = size(img.data, named2dimension(img, dimname))

ndims(img::AbstractImage) = ndims(img.data)

strides(img::AbstractImage) = strides(img.data)

copy(img::AbstractImage) = deepcopy(img)

# copy, replacing the data
copy(img::AbstractArray, data::AbstractArray) = data

copy(img::AbstractImageDirect, data::AbstractArray) = Image(data, copy(img.properties))

copy(img::AbstractImageIndexed, data::AbstractArray) = ImageCmap(data, copy(img.cmap), copy(img.properties))

# Provide new data but reuse the properties & cmap
share(img::AbstractArray, data::AbstractArray) = data

share(img::AbstractImageDirect, data::AbstractArray) = Image(data, img.properties)

share(img::AbstractImageIndexed, data::AbstractArray) = ImageCmap(data, img.cmap, img.properties)

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

# convert
convert{I<:AbstractImageDirect}(::Type{I}, img::I) = img
convert{I<:AbstractImageIndexed}(::Type{I}, img::I) = img
# Convert an indexed image (cmap) to a direct image
function convert{ID<:AbstractImageDirect,II<:AbstractImageIndexed}(::Type{ID}, img::II)
    if size(img.cmap, 2) == 1
        data = reshape(img.cmap[img.data[:]], size(img.data))
        prop = img.properties
        return Image(data, prop)
    else
        newsz = tuple(size(img.data)...,size(img.cmap,2))
        data = reshape(img.cmap[img.data[:],:], newsz)
        prop = copy(img.properties)
        prop["colordim"] = length(newsz)
        return Image(data, prop)
    end
end
# Convert an Image to an array. We convert the image into the canonical storage order convention for arrays. We restrict this to 2d images because for plain arrays this convention exists only for 2d.
# In other cases---or if you don't want the storage order altered---just grab the .data field and perform whatever manipulations you need directly.
function convert{T,N}(::Type{Array{T,N}}, img::AbstractImage)
    assert2d(img)
    if N != ndims(img)
        error("Number of dimensions of the output do not agree")
    end
    # put in canonical storage order
    p = spatialpermutation(spatialorder(Matrix), img)
    p = coords_spatial(img)[p]
    cd = colordim(img)
    if cd > 0
        push!(p, cd)
    end
    if issorted(p)
        return copy(img.data)
    else
        return permutedims(img.data, p)
    end
end
convert(::Type{Array}, img::AbstractImage) = convert(Array{eltype(img), ndims(img)}, img)

# Convert an array to an image
convert(::Type{Image}, A::Array) = Image(A, ["colorspace" => colorspace(A), "colordim" => colordim(A), "spatialorder" => spatialorder(A), "limits" => limits(A)])

# Indexing. In addition to conventional array indexing, support syntax like
#    img["x", 100:400, "t", 32]
# where anything not mentioned by name is taken to include the whole range

# setindex!
setindex!(img::AbstractImage, X, i::Real) = setindex!(img.data, X, i)

setindex!{T<:Real}(img::AbstractImage, X, I::Union(Real,AbstractArray{T})...) = setindex!(img.data, X, I...)

setindex!{T<:Real}(img::AbstractImage, X, dimname::String, ind::Union(Real,AbstractArray{T}), nameind...) = setindex!(img.data, X, named2coords(img, dimname, ind, nameind...)...)

# Adding a new property via setindex!
setindex!(img::AbstractImage, X, propname::String) = setindex!(img.properties, X, propname)

# Delete a property!
delete!(img::AbstractImage, propname::String) = delete!(img.properties, propname)


# getindex, sub, and slice return a value or AbstractArray, not an Image
getindex(img::AbstractImage, i::Real) = getindex(img.data, i)

getindex(img::AbstractImage, I::Union(Real,AbstractVector)...) = getindex(img.data, I...)

# getindex{T<:Real}(img::AbstractImage, dimname::String, ind::Union(Real,AbstractArray{T}), nameind...) = getindex(img.data, named2coords(img, dimname, ind, nameind...)...)
getindex(img::AbstractImage, dimname::ASCIIString, ind, nameind...) = getindex(img.data, named2coords(img, dimname, ind, nameind...)...)

getindex(img::AbstractImage, propname::ASCIIString) = getindex(img.properties, propname)

sub(img::AbstractImage, I::RangeIndex...) = sub(img.data, I...)

sub(img::AbstractImage, dimname::String, ind::RangeIndex, nameind::RangeIndex...) = sub(img.data, named2coords(img, dimname, ind, nameind...)...)

slice(img::AbstractImage, I::RangeIndex...) = slice(img.data, I...)

slice(img::AbstractImage, dimname::String, ind::RangeIndex, nameind::RangeIndex...) = slice(img.data, named2coords(img, dimname, ind, nameind...)...)

# getindexim, subim, and sliceim return an Image. The first two share properties, the last requires a copy.
function getindexim{T<:Real}(img::AbstractImage, I::Union(Real,AbstractArray{T})...)
    ret = copy(img, data(img)[I...])
    cd = colordim(img)
    nd = ndims(ret)
    if cd > nd
        ret["colordim"] = 0
        ret["colorspace"] = "Gray"
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

getindexim(img::AbstractImage, dimname::String, ind::Union(Real,AbstractArray), nameind...) = getindexim(img, named2coords(img, dimname, ind, nameind...)...)

subim(img::AbstractImage, I::RangeIndex...) = share(img, sub(img.data, I...))

subim(img::AbstractImage, dimname::String, ind::RangeIndex, nameind...) = subim(img, named2coords(img, dimname, ind, nameind...)...)

function sliceim(img::AbstractImage, I::RangeIndex...)
    dimmap = Array(Int, ndims(img))
    n = 0
    for j = 1:ndims(img)
        if !isa(I[j], Int); n += 1; end;
        dimmap[j] = n
    end
    ret = copy(img, slice(img.data, I...))
    cd = colordim(img)
    if cd > 0
        if isa(I[cd], Int)
            ret.properties["colordim"] = 0
            ret.properties["colorspace"] = "Gray"
        else
            ret.properties["colordim"] = dimmap[cd]
            if I[cd] != 1:size(img, cd)
                ret.properties["colorspace"] = "channels"
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
        l = Int[map(length, I[c])...]
        keep = l .> 1
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

sliceim(img::AbstractImage, dimname::String, ind::RangeIndex, nameind...) = subim(img, named2coords(img, dimname, ind, nameind...)...)

sliceim(img::AbstractImage, dimname::String, ind::RangeIndex, nameind...) = sliceim(img, named2coords(img, dimname, ind, nameind...)...)

# Support colon indexes
getindexim(img::AbstractImage, I...) = getindexim(img, ntuple(length(I), i-> isa(I[i], Colon) ? (1:size(img,i)) : I[i])...)
subim(img::AbstractImage, I...) = subim(img, ntuple(length(I), i-> isa(I[i], Colon) ? (1:size(img,i)) : I[i])...)
sliceim(img::AbstractImage, I...) = sliceim(img, ntuple(length(I), i-> isa(I[i], Colon) ? (1:size(img,i)) : I[i])...)


# We'll frequently want to pull out different 2d slices from the same image, so here's a type and set of functions making that easier.
# We deliberately do not require the user to specify the full list of new slicing/ranging parameters, as often we'll want to change some aspects (e.g., z-slice) but not others (e.g., color coordinates)
type SliceData
    slicedims::(Int...,)
    slicestrides::(Int...,)
    rangedims::(Int...,)

    function SliceData(A::AbstractArray, slicedims::Int...)
        keep = trues(ndims(A))
        for i = 1:length(slicedims)
            keep[slicedims[i]] = false
        end
        s = strides(A)
        new(slicedims, ntuple(length(slicedims), i->s[slicedims[i]]), tuple((1:ndims(A))[keep]...))
    end
end

SliceData(img::AbstractImage, dimname::String, dimnames::String...) = SliceData(img, named2dimindexes(img, dimname, dimnames...)...)

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
    for i = 1:length(sd.rangedims)
        newfirst += (A.indexes[sd.rangedims[i]][1]-1)*A.strides[i]
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

function rerange!(A::SubArray, sd::SliceData, I::(RangeIndex...,))
    indexes = RangeIndex[A.indexes...]
    for i = 1:length(I)
        indexes[sd.rangedims[i]] = I[i]
    end
    A.indexes = tuple(indexes...)
    A.first_index = first_index(A, sd)
    A
end

function rerange!(img::AbstractImage, sd::SliceData, I::(RangeIndex...,))
    rerange!(img.data, sd, I...)
    img
end


const emptyset = Set()
function show(io::IO, img::AbstractImageDirect)
    IT = typeof(img)
    print(io, colorspace(img), " ", IT.name, " with:\n  data: ", summary(img.data), "\n  properties:")
    showdictlines(io, img.properties, get(img, "suppress", emptyset))
end
function show(io::IO, img::AbstractImageIndexed)
    IT = typeof(img)
    print(io, colorspace(img), " ", IT.name, " with:\n  data: ", summary(img.data), "\n  cmap: ", summary(img.cmap), "\n  properties:")
    showdictlines(io, img.properties, get(img, "suppress", emptyset))
end

data(img::AbstractArray) = img
data(img::AbstractImage) = img.data

min(img::AbstractImageDirect) = min(img.data)
max(img::AbstractImageDirect) = max(img.data)
# min/max deliberately not defined for AbstractImageIndexed

# Overlays
eltype{T}(o::Overlay{T}) = T
eltype{T}(::Type{Overlay{T}}) = T
ndims{T,N}(o::Overlay{T,N}) = N
ndims{T,N}(::Type{Overlay{T,N}}) = N

size(o::Overlay) = size(o.channels[1])
size(o::Overlay, i::Integer) = size(o.channels[1], i)

show(io::IO, o::Overlay) = print(io, summary(o), " with colors ", o.colors)

function squeeze(img::AbstractImage, dims)
    imgret = copy(img, squeeze(data(img), dims))
    td = timedim(img)
    if td > 0 && in(td, dims)
        imgret["timedim"] = 0
    end
    cd = colordim(img)
    if cd > 0 && in(cd, dims)
        imgret["colordim"] = 0
    end
    imgret
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
#   spatialorder: a string naming each spatial dimension, in the storage order of
#     the data array. Names can be arbitrary, but the choices "x" and "y" have special
#     meaning (horizontal and vertical, respectively, irrespective of storage order).
#     If supplied, you must have one entry per spatial dimension.

properties(img::AbstractArray) = [
    "colorspace" => colorspace(img),
    "colordim" => colordim(img),
    "timedim" => timedim(img),
    "limits" => limits(img),
    "pixelspacing" => pixelspacing(img),
    "spatialorder" => spatialorder(img)]
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

colorspace{C<:ColorValue}(img::AbstractMatrix{C}) = string(C)
colorspace{C<:ColorValue}(img::AbstractArray{C,3}) = string(C)
colorspace{C<:ColorValue}(img::AbstractImage{C}) = string(C)
colorspace(img::AbstractMatrix{Bool}) = "Binary"
colorspace(img::AbstractArray{Bool}) = "Binary"
colorspace(img::AbstractArray{Bool,3}) = "Binary"
colorspace{T<:Union(Int32,Uint32)}(img::AbstractMatrix{T}) = "RGB24"
colorspace(img::AbstractMatrix) = "Gray"
# colorspace{T}(img::AbstractArray{T,3}) = (size(img, defaultarraycolordim) == 3) ? "RGB" : error("Cannot infer colorspace of Array, use an AbstractImage type")
colorspace{T}(img::AbstractArray{T,3}) = (size(img, defaultarraycolordim) == 3) ? "RGB" : 0
colorspace(img::AbstractImage{Bool}) = "Binary"
colorspace{T,N,A<:AbstractArray,C<:ColorValue}(img::ImageCmap{T,N,A,Array{C,1}}) = string(C)
colorspace(img::AbstractImageIndexed) = @get img "colorspace" csinfer(eltype(img.cmap))
colorspace{T}(img::AbstractImageIndexed{T,2}) = @get img "colorspace" csinfer(eltype(img.cmap))
csinfer{C<:ColorValue}(::Type{C}) = string(C)
csinfer(C) = "Unknown"
colorspace(img::AbstractImage) = get(img.properties, "colorspace", "Unknown")

colordim{C<:ColorValue}(img::AbstractVector{C}) = 0
colordim{C<:ColorValue}(img::AbstractMatrix{C}) = 0
colordim{C<:ColorValue}(img::AbstractArray{C,3}) = 0
colordim{C<:ColorValue}(img::AbstractImage{C}) = 0
colordim(img::AbstractMatrix) = 0
# colordim{T}(img::AbstractArray{T,3}) = (size(img, defaultarraycolordim) == 3) ? 3 : error("Cannot infer colordim of Array, use an AbstractImage type")
colordim{T}(img::AbstractArray{T,3}) = (size(img, defaultarraycolordim) == 3) ? 3 : 0
colordim(img::AbstractImageDirect) = get(img, "colordim", 0)
colordim(img::AbstractImageIndexed) = 0

timedim(img) = get(img, "timedim", 0)

limits(img::AbstractArray{Bool}) = 0,1
limits{T<:Integer}(img::AbstractArray{T}) = typemin(T), typemax(T)
limits{T<:FloatingPoint}(img::AbstractArray{T}) = zero(T), one(T)
limits(img::AbstractImage{Bool}) = 0,1
limits{T}(img::AbstractImageDirect{T}) = get(img, "limits", (typemin(T), typemax(T)))
limits(img::AbstractImageIndexed) = @get img "limits" (min(img.cmap), max(img.cmap))

pixelspacing{T}(img::AbstractArray{T,3}) = (size(img, defaultarraycolordim) == 3) ? [1.0,1.0] : error("Cannot infer pixelspacing of Array, use an AbstractImage type")
pixelspacing(img::AbstractMatrix) = [1.0,1.0]
pixelspacing{T}(img::AbstractImage{T}) = @get img "pixelspacing" _pixelspacing(img)
_pixelspacing(img::AbstractImage) = ones(sdims(img))

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

# indices of spatial coordinates
function coords_spatial(img)
    nd = ndims(img)
    cd = colordim(img)
    td = timedim(img)
    if cd > nd || td > nd
        error("Properties are inconsistent with the array dimensionality")
    end
    ind = [1:nd]
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
permutedims(img::AbstractImage, p::(), spatialprops::Vector = spatialproperties(img)) = img

function permutedims(img::AbstractImage, p::Union(Vector{Int}, (Int...)), spatialprops::Vector = spatialproperties(img))
    if length(p) != ndims(img)
        error("The permutation must have length equal to the number of dimensions")
    end
    ip = invperm(p)
    cd = colordim(img)
    sd = timedim(img)
    ret = copy(img, permutedims(img.data, p))
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

permutedims{S<:String}(img::AbstractImage, pstr::Union(Vector{S}, (S...)), spatialprops::Vector = spatialproperties(img)) = permutedims(img, named2dimindexes(img, pstr...), spatialprops)

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
    spatialprops
end
spatialproperties(img::AbstractVector) = ASCIIString[]  # these are not mutable


#### Low-level utilities ####
function permutation(to, from)
    n = length(to)
    d = Dict(tuple(from...), tuple([1:length(from)]...))
    ind = Array(Int, n)
    for i = 1:n
        ind[i] = get(d, to[i], 0)
    end
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
function named2coords(img::AbstractImage, nameind...)
    c = Any[map(i->1:i, size(img))...]
    so = spatialorder(img)
    for i = 1:2:length(nameind)
        dimname = nameind[i]
        n = named2dimension(img, dimname, so)
        if n == 0
            error("There is no dimension called ", dimname)
        end
        c[n] = nameind[i+1]
    end
    tuple(c...)
end

function named2dimension(img::AbstractImage, dimname, so = spatialorder(img))
    local n
    if dimname == "color"
        n = colordim(img)
    elseif dimname == "t"
        n = timedim(img)
    else
        cd = colordim(img)
        td = timedim(img)
        matched = false
        j = 1
        n = 1
        while j <= length(so)
            while n == cd || n == td
                n += 1
            end
            if dimname == so[j]
                matched = true
                break
            end
            n += 1
            j += 1
        end
        if !matched; n = 0; end
    end
    n
end

function named2dimindexes(img::AbstractImage, dimnames::String...)
    so = spatialorder(img)
    dimindexes = Array(Int, length(dimnames))
    for i = 1:length(dimnames)
        dimname = dimnames[i]
        n = named2dimension(img, dimname, so)
        if n == 0
            error("There is no dimension called ", dimname)
        end
        dimindexes[i] = n
    end
    dimindexes
end

