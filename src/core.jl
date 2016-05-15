#### Types and constructors ####

# Plain arrays can be treated as images. Other types will have
# metadata associated, make yours a child of one of the following:
abstract AbstractImage{T,N} <: AbstractArray{T,N}         # image with metadata
"""
`AbstractImageDirect` is the supertype of all images where pixel
values are stored directly in an `AbstractArray`.  See also
`AbstractImageIndexed`.
"""
abstract AbstractImageDirect{T,N} <: AbstractImage{T,N}
"""
`AbstractImageIndexed` is the supertype of all "colormap" images,
where pixel values are accessed from a lookup table.  See also
`AbstractImageDirect`.
"""
abstract AbstractImageIndexed{T,N} <: AbstractImage{T,N}

"""
```
Image(data, [properties])
Image(data, prop1=val1, prop2=val2, ...)
```
creates a new "direct" image, one in which the values in `data`
correspond to the pixel values. In contrast with `convert`, `grayim`
and `colorim`, this does not permute the data array or attempt to
guess any of the `properties`. If `data` encodes color information
along one of the dimensions of the array (as opposed to using a
`Color` array, from the `Colors.jl` package), be sure to specify the
`"colordim"` and `"colorspace"` in `properties`.
"""
type Image{T,N,A<:AbstractArray} <: AbstractImageDirect{T,N}
    data::A
    properties::Dict{String,Any}
end
Image(data::AbstractArray, props::Dict) = Image{eltype(data),ndims(data),typeof(data)}(data,props)
Image{T,N}(data::AbstractArray{T,N}, props::Dict) = Image{T,N,typeof(data)}(data,props)
Image(data::AbstractArray; kwargs...) = Image(data, kwargs2dict(kwargs))

"""
```
ImageCmap(data, cmap, [properties])
```
creates an indexed (colormap) image.
"""
type ImageCmap{T<:Colorant,N,A<:AbstractArray} <: AbstractImageIndexed{T,N}
    data::A
    cmap::Vector{T}
    properties::Dict{String,Any}
end
ImageCmap{_,N}(data::AbstractArray{_,N}, cmap::AbstractVector, props::Dict) = ImageCmap{eltype(cmap),N,typeof(data)}(data, cmap, props)
ImageCmap(data::AbstractArray, cmap::AbstractVector; kwargs...) = ImageCmap(data, cmap, kwargs2dict(kwargs))

# Convenience constructors
"""
```
img = grayim(A)
```
creates a 2d or 3d _spatial_ grayscale Image from an AbstractArray
`A`, assumed to be in "horizontal-major" order (and without permuting
any dimensions). If you are working with 3d grayscale images, usage of
this function is strongly recommended. This can fix errors like any of
the following:

```
ERROR: Wrong number of spatial dimensions for plain Array, use an AbstractImage type
ERROR: Cannot infer colorspace of Array, use an AbstractImage type
ERROR: Cannot infer pixelspacing of Array, use an AbstractImage type
```

The main reason for such errors---and the reason that `grayim` is
recommended---is the Matlab-derived convention that a `m x n x 3` array is to be
interpreted as RGB.  One might then say that an `m x n x k` array, for `k`
different from 3, could be interpreted as grayscale. However, this would lead to
difficult-to-track-down surprises on the day where `k` happened to be 3 for your
grayscale image.

See also: `colorim`, `Image`, `convert(Image, A)`.
"""
grayim(A::AbstractImage) = A
grayim(A::AbstractArray{UInt8,2})  = grayim(reinterpret(UFixed8, A))
grayim(A::AbstractArray{UInt16,2}) = grayim(reinterpret(UFixed16, A))
grayim(A::AbstractArray{UInt8,3})  = grayim(reinterpret(UFixed8, A))
grayim(A::AbstractArray{UInt16,3}) = grayim(reinterpret(UFixed16, A))
grayim{T}(A::AbstractArray{T,2}) = Image(A; colorspace="Gray", spatialorder=["x","y"])
grayim{T}(A::AbstractArray{T,3}) = Image(A; colorspace="Gray", spatialorder=["x","y","z"])

"""
```
img = colorim(A, [colorspace])
```
Creates a 2d color image from an AbstractArray `A`, auto-detecting which of the
first or last dimension encodes the color and choosing between "horizontal-" and
"vertical-major" accordingly. `colorspace` defaults to `"RGB"` but could also be
e.g. `"Lab"` or `"HSV"`.  If the array represents a 4-channel image, the
`colorspace` option is mandatory since there is no way to automatically
distinguish between `"ARGB"` and `"RGBA"`.  If both the first and last
dimensions happen to be of size 3 or 4, it is impossible to guess which one
represents color and thus an error is generated.  Thus, if your code needs to be
robust to arbitrary-sized images, you should use the `Image` constructor
directly.

See also: `grayim`, `Image`, `convert(Image{RGB}, A)`.
"""
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

colorim(A::AbstractArray{UInt8,3})  = colorim(reinterpret(UFixed8, A))
colorim(A::AbstractArray{UInt16,3}) = colorim(reinterpret(UFixed16, A))
colorim(A::AbstractArray{UInt8,3},  colorspace) = colorim(reinterpret(UFixed8, A), colorspace)
colorim(A::AbstractArray{UInt16,3}, colorspace) = colorim(reinterpret(UFixed16, A), colorspace)


#### Core operations ####

eltype{T}(img::AbstractImage{T}) = T

size(img::AbstractImage) = size(img.data)
size(img::AbstractImage, i::Integer) = size(img.data, i)
size(img::AbstractImage, dimname::AbstractString) = size(img.data, dimindex(img, dimname))

resize!(a::AbstractImage, nl::Integer) = resize!(a.data, nl)

ndims(img::AbstractImage) = ndims(img.data)

linearindexing(img::Image) = linearindexing(img.data)

strides(img::AbstractImage) = strides(img.data)

copy(img::Image) = Image(copy(img.data), dictcopy(img.properties))
copy(img::ImageCmap) = ImageCmap(copy(img.data), copy(img.cmap), dictcopy(img.properties))

if VERSION < v"0.5.0-dev"
    function dictcopy(dct)
        newkeys = [copy(key) for key in keys(dct)]
        newvals = [copy(val) for val in values(dct)]
        Dict{String,Any}(zip(newkeys,newvals))
    end
else
    dictcopy(dct) = deepcopy(dct)
end

"""
```
imgnew = copyproperties(img, data)
```
Creates a new image from the data array `data`, copying the properties from
Image `img`.
"""
copyproperties(img::AbstractArray, data::AbstractArray) = data

copyproperties(img::AbstractImageDirect, data::AbstractArray) = Image(data, deepcopy(img.properties))

copyproperties(img::AbstractImageIndexed, data::AbstractArray) = ImageCmap(data, copy(img.cmap), deepcopy(img.properties))

copyproperties(img::AbstractImageDirect, _data::AbstractImageDirect) = copyproperties(img, data(_data))

"""
```
imgnew = shareproperties(img, data)
```
Creates a new image from the data array `data`, *sharing* the properties of
Image `img`. Any modifications made to the properties of one will affect the
other.
"""
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
function copy!(imgdest::AbstractImage, imgsrc::AbstractImage, prop1::String, props::String...)
    imgdest[prop1] = imgsrc[prop1]
    for p in props
        imgdest[p] = imgsrc[p]
    end
    imgdest
end

function reshape{N}(img::AbstractImage, dims::NTuple{N,Int})
    ret = copyproperties(img, reshape(data(img), dims))
    for prop in spatialproperties(img)
        delete!(ret, prop)
    end
    if colordim(img) != 0
        delete!(ret, "colordim")
        delete!(ret, "colorspace")
    end
    delete!(ret, "timedim")
    ret
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
    elseif sizeof(T)*n_elts(CV) == sizeof(CV)
        return reinterpret(T, A, tuple(n_elts(CV), size(A)...))
    end
    error("result shape not specified")
end
reinterpret{CV<:Colorant}(A::StridedArray{CV}) = reinterpret(eltype(CV), A)

# Images
reinterpret{CV1<:Colorant,CV2<:Colorant}(::Type{CV1}, img::AbstractImageDirect{CV2}) =
    shareproperties(img, reinterpret(CV1, data(img)))
function reinterpret{CV<:Colorant}(::Type{UInt32}, img::AbstractImageDirect{CV})
    CV <: Union{RGB24, ARGB32} || (CV <: AbstractRGB && sizeof(CV) == 4) || error("Can't convert $CV to UInt32")
    A = reinterpret(UInt32, data(img))
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
#   form 2: reinterpret(RGB{UFixed8}, img)
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

"""
```
imgraw = raw(img)
```
returns a reference to the array data in raw (machine-native) storage
format. This is particularly useful when Images.jl wraps image data in
a `FixedPointNumbers` type, and raw data access is desired. For
example

```
img = load("someimage.tif")
typeof( data(img) )  # return Array{UFixed{UInt8,8},2}
typeof( raw(img) )   # returns Array{UInt8,2}
```
"""
raw(img::AbstractArray) = _raw(data(img), eltype(eltype(img)))
_raw{T<:UFixed}(A::Array, ::Type{T}) = reinterpret(FixedPointNumbers.rawtype(T), A)
_raw{T}(A::Array, ::Type{T}) = A
_raw{T}(A::AbstractArray, ::Type{T}) = _raw(convert(Array, A), T)

## convert
# Implementations safe for under-specified color types
# ambiguity resolution:
convert{T<:Colorant,n}(::Type{Array{T}}, x::Array{T,n}) = x
convert{T<:Colorant,n}(::Type{Array{T,n}}, x::Array{T,n}) = x
convert{T<:Colorant,n}(::Type{Array{T}}, x::BitArray{n}) = convert(Array{ccolor(T,Gray{Bool}),n}, x)
if VERSION >= v"0.5.0-dev"
    # See julia #15801. May be unfixable on 0.4.
    convert{T<:Colorant,n}(::Type{Array{T,n}}, x::BitArray{n}) = Base._convert(Array{ccolor(T,Gray{Bool}),n}, x)
end

if VERSION < v"0.5.0-dev"
    convert{T<:Colorant,n,S}(::Type{Array{T}}, x::Array{S,n}) = convert(Array{ccolor(T,S),n}, x)
    convert{T<:Colorant,n,S}(::Type{Array{T,n}}, x::Array{S,n}) = copy!(Array(ccolor(T,S), size(x)), x)
end

"""
```
img = convert(Image, A)
img = convert(Image{HSV}, img)
```
Create a 2d Image from an array, setting up default properties. The
data array is assumed to be in "vertical-major" order, and an m-by-n-by-3 array
will be assumed to encode color along its third dimension.

Optionally, you can specify the desired colorspace of the returned `img`.

See also: `Image`, `grayim`, `colorim`.
"""
convert{T}(::Type{Image{T}}, img::Image{T}) = img
convert(::Type{Image}, img::Image) = img
convert(::Type{Image}, A::AbstractArray) = Image(A, properties(A))
# Convert an indexed image (cmap) to a direct image
function convert(::Type{Image}, img::ImageCmap)
    data = reshape(img.cmap[vec(img.data)], size(img.data))
    Image(data, copy(properties(img)))
end

convert{C<:Colorant}(::Type{Image{C}}, img::Image{C}) = img
convert{Cdest<:Colorant,Csrc<:Colorant}(::Type{Image{Cdest}}, img::Image{Csrc}) = copyproperties(img, convert(Array{ccolor(Cdest,Csrc)}, data(img)))
convert{Cdest<:Colorant,Csrc<:Colorant}(::Type{Image{Cdest}}, img::AbstractArray{Csrc}) = Image(convert(Array{Cdest}, data(img)), properties(img))

# Convert an Image to an array. We convert the image into the canonical storage order convention for arrays.
# We restrict this to 2d images because for plain arrays this convention exists only for 2d.
# In other cases---or if you don't want the storage order altered---just use data(img)
"""
`A = convert(Array, img)` converts an Image `img` to an Array,
permuting dimensions (if needed) to put it in vertical-major (Matlab)
storage order.

See also `data`.
"""
convert{T<:Real,N}(::Type{Array{T}}, img::AbstractImageDirect{T,N}) = convert(Array{T,N}, img)
convert{T<:Colorant,N}(::Type{Array{T}}, img::AbstractImageDirect{T,N}) = convert(Array{T,N}, img)
convert{T}(::Type{Vector{T}}, img::AbstractImageDirect{T,1}) = convert(Vector{T}, data(img))
convert{T<:Colorant,N,S}(::Type{Array{T,N}}, img::AbstractImageDirect{S,N}) = _convert(Array{ccolor(T,S),N}, img)
convert{T,N,S}(::Type{Array{T,N}}, img::AbstractImageDirect{S,N}) = _convert(Array{T,N}, img)
function _convert{T,N,S}(::Type{Array{T,N}}, img::AbstractImageDirect{S,N})
    assert2d(img)  # only well-defined in 2d
    p = permutation_canonical(img)
    dat = convert(Array{T}, data(img))
    if issorted(p)
        return dat
    else
        return permutedims(dat, p)
    end
end
convert{T<:Colorant,n,S}(::Type{Array{T}}, x::AbstractArray{S,n}) = convert(Array{ccolor(T,S),n}, x)
if VERSION < v"0.5.0-dev"
    convert{T,N}(::Type{Array{T,N}}, img::AbstractArray) = copy!(Array{T}(size(img)), img)
else
    convert{T<:Colorant,n,S}(::Type{Array{T,n}}, x::AbstractArray{S,n}) = copy!(Array(ccolor(T,S), size(x)), x)
end

convert{Cdest<:Colorant,Csrc<:Colorant}(::Type{Image{Cdest}}, img::AbstractImageDirect{Csrc}) =
    copyproperties(img, convert(Array{ccolor(Cdest,Csrc)}, data(img)))  # FIXME when Julia issue ?? is fixed
convert{Cdest<:Colorant,Csrc<:Colorant,N}(::Type{Array{Cdest}}, img::AbstractImageDirect{Csrc,N}) =
    convert(Array{ccolor(Cdest,Csrc)}, convert(Array{Csrc,N}, img))

#convert{T<:Colorant}(::Type{Array{T}}, x) = copy!(similar(x,ccolor(T,eltype(x))), x)
#convert{T<:Colorant,n,S}(::Type{Array{T,n}}, x::AbstractArray{S,n}) = copy!(Array(ccolor(T,S), size(x)), x)

"""
`imgs = separate(img)` separates the color channels of `img`, for
example returning an `m-by-n-by-3` array from an `m-by-n` array of
`RGB`.
"""
function separate{CV<:Colorant}(img::AbstractImage{CV})
    p = permutation_canonical(img)
    A = _separate(data(img), p)
    so = spatialorder(img)[p]
    props = copy(properties(img))
    props["colorspace"] = colorspace(img)
    props["colordim"] = ndims(A)
    props["spatialorder"] = so
    Image(A, props)
end
function _separate{CV}(A::Array{CV}, p)
    T = eltype(CV)
    if n_elts(CV) > 1
        permutedims(reinterpret(T, A, tuple(n_elts(CV), size(A)...)), [p+1;1])
    else
        permutedims(reinterpret(T, A, size(A)), p)
    end
end
_separate{CV}(A::AbstractArray{CV}, p) = _separate(convert(Array, A), p)
function separate{CV<:Colorant}(A::Array{CV})
    T = eltype(CV)
    if n_elts(CV) > 1
        permutedims(reinterpret(T, A, tuple(n_elts(CV), size(A)...)), [2:ndims(A)+1;1])
    else
        reinterpret(T, A, size(A))
    end
end
separate{CV<:Colorant}(A::AbstractArray{CV}) = separate(convert(Array, A))
separate(A::AbstractArray) = A

# Image{Numbers} -> Image{Colorant} (the opposite of separate)
convert{C<:Colorant,T<:Fractional}(::Type{Image{C}}, img::Union{AbstractArray{T},AbstractImageDirect{T}}) =
    _convert(Image{C}, eltype(C), img)
_convert{C<:Colorant,T<:Fractional}(::Type{Image{C}}, ::Type{Any}, img::Union{AbstractArray{T},AbstractImageDirect{T}}) =
    _convert(Image{C{T}}, img)
_convert{C<:Colorant,T<:Fractional}(::Type{Image{C}}, ::DataType, img::Union{AbstractArray{T},AbstractImageDirect{T}}) =
    _convert(Image{C}, img)
function _convert{C<:Colorant,T<:Fractional}(::Type{Image{C}}, img::Union{AbstractArray{T},AbstractImageDirect{T}})
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

typealias RealIndex{T<:Real} Union{T, AbstractVector{T}, Colon}

# setindex!
setindex!(img::AbstractImage, X, i::Real) = setindex!(img.data, X, i)
setindex!(img::AbstractImage, X, I::RealIndex) = setindex!(img.data, X, I)
setindex!(img::AbstractImage, X, I::RealIndex, J::RealIndex) = setindex!(img.data, X, I, J)
setindex!(img::AbstractImage, X, I::RealIndex, J::RealIndex, K::RealIndex) = setindex!(img.data, X, I, J, K)
setindex!(img::AbstractImage, X, I::RealIndex, J::RealIndex,
                   K::RealIndex, L::RealIndex) = setindex!(img.data, X, I, J, K, L)
setindex!(img::AbstractImage, X, I::RealIndex...) = setindex!(img.data, X, I...)
setindex!(img::AbstractImage, X, dimname::AbstractString, ind::RealIndex, nameind...) = setindex!(img.data, X, coords(img, dimname, ind, nameind...)...)

# Adding a new property via setindex!
setindex!(img::AbstractImage, X, propname::AbstractString) = setindex!(img.properties, X, propname)

# Delete a property!
delete!(img::AbstractImage, propname::AbstractString) = delete!(img.properties, propname)


# getindex, sub, and slice return a value or AbstractArray, not an Image
getindex(img::AbstractImage, i::Real) = getindex(img.data, i)
getindex(img::AbstractImage, I::RealIndex) = getindex(img.data, I)
getindex(img::AbstractImage, I::RealIndex, J::RealIndex) = getindex(img.data, I, J)
getindex(img::AbstractImage, I::RealIndex, J::RealIndex, K::RealIndex) = getindex(img.data, I, J, K)
getindex(img::AbstractImage, I::RealIndex, J::RealIndex,
            K::RealIndex, L::RealIndex) = getindex(img.data, I, J, K, L)
getindex(img::AbstractImage, I::RealIndex...) = getindex(img.data, I...)

# getindex(img::AbstractImage, dimname::AbstractString, ind::RealIndex, nameind...) = getindex(img.data, coords(img, dimname, ind, nameind...)...)
getindex(img::AbstractImage, dimname::String, ind, nameind...) = getindex(img.data, coords(img, dimname, ind, nameind...)...)

getindex(img::AbstractImage, propname::String) = getindex(img.properties, propname)

typealias Indexable{T<:Real} Union{Int, AbstractVector{T}, Colon}  # for ambiguity resolution
sub(img::AbstractImage, I::Indexable...) = sub(img.data, I...)
sub(img::AbstractImage, I::RealIndex...) = sub(img.data, I...)

sub(img::AbstractImage, dimname::String, ind::RealIndex, nameind...) = sub(img.data, coords(img, dimname, ind, nameind...)...)

slice(img::AbstractImage, I::Indexable...) = slice(img.data, I...)
slice(img::AbstractImage, I::RealIndex...) = slice(img.data, I...)

slice(img::AbstractImage, dimname::String, ind::RealIndex, nameind...) = slice(img.data, coords(img, dimname, ind, nameind...)...)

"""
```
imgnew = getindexim(img, i, j, k,...)
imgnew = getindexim(img, "x", 100:200, "y", 400:600)
```
return a new Image `imgnew`, copying (and where necessary modifying)
the properties of `img`.  This is in contrast with `img[i, j, k...]`,
which returns an `Array`.
"""
function getindexim(img::AbstractImage, I::RealIndex...)
    ret = copyproperties(img, data(img)[I...])
    cd = colordim(img)
    nd = ndims(ret)
    if cd > nd || (cd > 0 && _length(I[cd], img, cd) < size(img, cd))
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
_length(indx, A, d) = length(indx)
_length(indx::Colon, A, d) = size(A,d)


getindexim(img::AbstractImage, dimname::String, ind::RealIndex, nameind...) = getindexim(img, coords(img, dimname, ind, nameind...)...)

"""
```
imgs = subim(img, i, j, k, ...)
imgs = subim(img, "x", 100:200, "y", 400:600)
```
returns an `Image` with `SubArray` data, with indexing semantics similar to `sub`.
"""
subim(img::AbstractImage, I::RealIndex...) = _subim(img, I)
_subim{TT}(img, I::TT) = shareproperties(img, sub(img.data, I...))  # work around #8504

subim(img::AbstractImage, dimname::String, ind::RealIndex, nameind...) = subim(img, coords(img, dimname, ind, nameind...)...)

"""
```
imgs = sliceim(img, i, j, k, ...)
imgs = sliceim(img, "x", 100:200, "y", 400:600)
```
returns an `Image` with `SubArray` data, with indexing semantics similar to `slice`.
"""
sliceim(img::AbstractImage, I::RealIndex...) = _sliceim(img, I)
function _sliceim{IT}(img::AbstractImage, I::IT)
    dimmap = Array(Int, ndims(img))
    n = 0
    for j = 1:ndims(img)
        if !isa(I[j], Real); n += 1; end;
        dimmap[j] = n
    end
    S = slice(img.data, I...)
    ret = copyproperties(img, S)
    cd = colordim(img)
    if cd > 0
        if isa(I[cd], Real)
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
        ret.properties["timedim"] = isa(I[td], Real) ? 0 : dimmap[td]
    end
    sp = spatialproperties(img)
    if !isempty(sp)
        c = coords_spatial(img)
        keep = Bool[map(x -> !isa(x, Real), I[c])...]
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

sliceim(img::AbstractImage, dimname::AbstractString, ind::RealIndex, nameind...) = sliceim(img, coords(img, dimname, ind, nameind...)...)


# Iteration
# Defer to the array object in case it has special iteration defined
next{T,N}(img::AbstractImage{T,N}, s::Tuple{Bool,Base.IteratorsMD.CartesianIndex{N}}) = next(data(img), s)
done{T,N}(img::AbstractImage{T,N}, s::Tuple{Bool,Base.IteratorsMD.CartesianIndex{N}}) = done(data(img), s)
start(img::AbstractImage) = start(data(img))
next(img::AbstractImage, s) = next(data(img), s)
done(img::AbstractImage, s) = done(data(img), s)


# We'll frequently want to pull out different 2d slices from the same image, so here's a type and set of functions making that easier.
# We deliberately do not require the user to specify the full list of new slicing/ranging parameters, as often we'll want to change some aspects (e.g., z-slice) but not others (e.g., color coordinates)
type SliceData
    slicedims::Tuple{Vararg{Int}}
    slicestrides::Tuple{Vararg{Int}}
    rangedims::Tuple{Vararg{Int}}

    function SliceData(A::AbstractArray, slicedims::Int...)
        keep = trues(ndims(A))
        for i = 1:length(slicedims)
            keep[slicedims[i]] = false
        end
        s = strides(A)
        new(slicedims, ntuple(i->s[slicedims[i]], length(slicedims)), tuple((1:ndims(A))[keep]...))
    end
end

SliceData(img::AbstractImage, dimname::AbstractString, dimnames::AbstractString...) = SliceData(img, dimindexes(img, dimname, dimnames...)...)

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

function rerange!(A::SubArray, sd::SliceData, I::Tuple{Vararg{RangeIndex}})
    indexes = RangeIndex[A.indexes...]
    for i = 1:length(I)
        indexes[sd.rangedims[i]] = I[i]
    end
    A.indexes = tuple(indexes...)
    A.first_index = first_index(A, sd)
    A
end

function rerange!(img::AbstractImage, sd::SliceData, I::Tuple{Vararg{RangeIndex}})
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

"""
```
A = data(img)
```
returns a reference `A` to the array data in `img`. It allows you to
use algorithms specialized for particular `AbstractArray` types on
`Image` types. This works for both `AbstractImage`s and
`AbstractArray`s (for the latter it just returns the input), so is a
"safe" component of any algorithm.

For algorithms written to accept arbitrary `AbstractArrays`, this
function is not needed.
"""
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
#   pixelspacing: the spacing between adjacent pixels along spatial dimensions
#   spacedirections: the direction of each array axis in physical space (a vector-of-vectors, one per dimension)
#   spatialorder: a string naming each spatial dimension, in the storage order of
#     the data array. Names can be arbitrary, but the choices "x" and "y" have special
#     meaning (horizontal and vertical, respectively, irrespective of storage order).
#     If supplied, you must have one entry per spatial dimension.

"""
`prop = properties(img)` returns the properties-dictionary for an
`AbstractImage`, or creates one if `img` is an `AbstractArray`.
"""
properties(A::AbstractArray) = Dict(
    "colorspace" => colorspace(A),
    "colordim" => colordim(A),
    "timedim" => timedim(A),
    "pixelspacing" => pixelspacing(A),
    "spatialorder" => spatialorder(A))
properties{C<:Colorant}(A::AbstractArray{C}) = Dict(
    "timedim" => timedim(A),
    "pixelspacing" => pixelspacing(A),
    "spatialorder" => spatialorder(A))
properties(img::AbstractImage) = img.properties

haskey(a::AbstractArray, k::AbstractString) = false
haskey(img::AbstractImage, k::AbstractString) = haskey(img.properties, k)

get(img::AbstractArray, k::AbstractString, default) = default
get(img::AbstractImage, k::AbstractString, default) = get(img.properties, k, default)

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

"""
```
order = spatialorder(img)
order = spatialorder(ImageType)
```

Returns the storage order of the _spatial_ coordinates of the image, e.g.,
`["y", "x"]`. The second version works on a type, e.g., `Matrix`. See
`storageorder`, `timedim`, and `colordim` for related properties.
"""
spatialorder(::Type{Matrix}) = yx
spatialorder(img::AbstractArray) = (sdims(img) == 2) ? spatialorder(Matrix) : error("Wrong number of spatial dimensions for plain Array, use an AbstractImage type")

"""
`isdirect(img)` returns true if `img` encodes its values directly,
rather than via an indexed colormap.
"""
isdirect(img::AbstractArray) = true
isdirect(img::AbstractImageDirect) = true
isdirect(img::AbstractImageIndexed) = false

"""
`cs = colorspace(img)` returns a string specifying the colorspace
representation of the image.
"""
colorspace{C<:Colorant}(img::AbstractVector{C}) = ColorTypes.colorant_string(C)
colorspace{C<:Colorant}(img::AbstractMatrix{C}) = ColorTypes.colorant_string(C)
colorspace{C<:Colorant}(img::AbstractArray{C,3}) = ColorTypes.colorant_string(C)
colorspace{C<:Colorant}(img::AbstractImage{C}) = ColorTypes.colorant_string(C)
colorspace(img::AbstractVector{Bool}) = "Binary"
colorspace(img::AbstractMatrix{Bool}) = "Binary"
colorspace(img::AbstractArray{Bool}) = "Binary"
colorspace(img::AbstractArray{Bool,3}) = "Binary"
colorspace(img::AbstractMatrix{UInt32}) = "RGB24"
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

colorspacedict = Dict{String,Any}()
for ACV in (Color, AbstractRGB)
    for CV in subtypes(ACV)
        (length(CV.parameters) == 1 && !(CV.abstract)) || continue
        str = string(CV.name.name)
        colorspacedict[str] = CV
    end
end
function getcolortype{T}(str::String, ::Type{T})
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

"""
`dim = colordim(img)` returns the dimension used to encode color, or 0
if no dimension of the array is used for color. For example, an
`Array` of size `(m, n, 3)` would result in 3, whereas an `Array` of
`RGB` colorvalues would yield 0.

See also: `ncolorelem`, `timedim`.
"""
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

"""
`dim = timedim(img)` returns the dimension used to represent time, or
0 if this is a single image.

See also: `nimages`, `colordim`.
"""
timedim(img) = get(img, "timedim", 0)::Int

oldlimits(img::AbstractArray{Bool}) = 0,1
# limits{T<:Integer}(img::AbstractArray{T}) = typemin(T), typemax(T)  # best not to use Integers...
oldlimits{T<:AbstractFloat}(img::AbstractArray{T}) = zero(T), one(T)
oldlimits(img::AbstractImage{Bool}) = 0,1
oldlimits{T}(img::AbstractImageDirect{T}) = get(img, "limits", (zero(T), one(T)))
oldlimits(img::AbstractImageIndexed) = @get img "limits" (minimum(img.cmap), maximum(img.cmap))

"""
```
ps = pixelspacing(img)
```

Returns a vector `ps` containing the spacing between adjacent pixels along each
dimension. If this property is not available, it will be computed from
`"spacedirections"` if present; otherwise it defaults to `ones(sdims(img))`. If
desired, you can set this property in terms of physical
[units](https://github.com/Keno/SIUnits.jl).

See also: `spacedirections`.
"""
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


"""
```
sd = spacedirections(img)
```

Returns a vector-of-vectors `sd`, each `sd[i]`indicating the displacement between adjacent
pixels along spatial axis `i` of the image array, relative to some external
coordinate system ("physical coordinates").  For example, you could indicate
that a photograph was taken with the camera tilted 30-degree relative to
vertical using

```
img["spacedirections"] = [[0.866025,-0.5],[0.5,0.866025]]
```

If not specified, it will be computed from `pixelspacing(img)`, placing the
spacing along the "diagonal".  If desired, you can set this property in terms of
physical [units](https://github.com/loladiro/SIUnits.jl).

See also: `pixelspacing`.
"""
spacedirections(img::AbstractArray) = @get img "spacedirections" _spacedirections(img)
function _spacedirections(img::AbstractArray)
    ps = pixelspacing(img)
    T = eltype(ps)
    nd = length(ps)
    Vector{T}[(tmp = zeros(T, nd); tmp[i] = ps[i]; tmp) for i = 1:nd]
end

"""
```
so = spatialorder(img)
so = spatialorder(ImageType)
```

Returns the storage order of the *spatial* coordinates of the image, e.g.,
`["y", "x"]`. The second version works on a type, e.g., `Matrix`.

See also: `storageorder`, `coords_spatial`, `timedim`, and `colordim`.
"""
spatialorder(img::AbstractImage) = @get img "spatialorder" _spatialorder(img)
_spatialorder(img::AbstractImage) = (sdims(img) == 2) ? spatialorder(Matrix) : error("Cannot guess default spatial order for ", sdims(img), "-dimensional images")

"""
```
so = storageorder(img)
```

Returns the complete storage order of the image array, including `"t"` for time
and `"color"` for color.

See also: `spatialorder`, `colordim`, `timedim`.
"""
function storageorder(img::AbstractArray)
    so = Array(String, ndims(img))
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
"""
`n = sdims(img)` is similar to `ndims`, but it returns just the number of *spatial* dimensions in
the image array (excluding color and time).
"""
sdims(img) = ndims(img) - (colordim(img) != 0) - (timedim(img) != 0)

# number of time slices
"""
`n = nimages(img)` returns the number of time-points in the image
array. This is safer than `size(img, "t")` because it also works for
plain `AbstractArray` types.
"""
function nimages(img)
    sd = timedim(img)
    if sd > 0
        return size(img, sd)
    else
        return 1
    end
end

"""
`n = ncolorelem(img)` returns the number of color elements/voxel, or 1 if color is not a separate dimension of the array.
"""
function ncolorelem(img)
    cd = colordim(img)
    return cd > 0 ? size(img, cd) : 1
end

"""
`c = coords_spatial(img)` returns a vector listing the spatial
dimensions of the image. For example, an `Array` of size `(m,n,3)`
would return `[1,2]`.

See also: `spatialorder`.
"""
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
"""
```
ssz = size_spatial(img)
```

Returns a tuple listing the sizes of the spatial dimensions of the image. For
example, an `Array` of size `(m,n,3)` would return `(m,n)`.

See also: `nimages`, `width`, `height`, `widthheight`.
"""
function size_spatial(img)
    sz = size(img)
    sz[coords_spatial(img)]
end

#### Utilities for writing "simple algorithms" safely ####
# If you don't feel like supporting multiple representations, call these

"""
`assert2d(img)` triggers an error if the image has more than two spatial
dimensions or has a time dimension.
"""
function assert2d(img::AbstractArray)
    if sdims(img) != 2
        error("Only two-dimensional images are supported")
    end
    if timedim(img) != 0
        error("Image sequences are not supported")
    end
end

"""
`assert_scalar_color(img)` triggers an error if the image uses an
array dimension to encode color.
"""
function assert_scalar_color(img::AbstractArray)
    if colordim(img) != 0
        error("Only 'scalar color' is supported")
    end
end


"""
`assert_timedim_last(img)` triggers an error if the image has a time
dimension that is not the last dimension.
"""
function assert_timedim_last(img::AbstractArray)
    if 0 < timedim(img) < ndims(img)
        error("Time dimension is not last")
    end
end

"""
`tf = isyfirst(img)` tests whether the first spatial dimension is `"y"`.

See also: `isxfirst`, `assert_yfirst`.
"""
isyfirst(img::AbstractArray) = spatialorder(img)[1] == "y"
"""
`assert_yfirst(img)` triggers an error if the first spatial dimension
is not `"y"`.
"""
function assert_yfirst(img)
    if !isyfirst(img)
        error("Image must have y as its first dimension")
    end
end

"""
`tf = isxfirst(img)` tests whether the first spatial dimension is `"x"`.

See also: `isyfirst`, `assert_xfirst`.
"""
isxfirst(img::AbstractArray) = spatialorder(img)[1] == "x"
"""
`assert_xfirst(img)` triggers an error if the first spatial dimension
is not `"x"`.
"""
function assert_xfirst(img::AbstractArray)
    if !isxfirst(img)
        error("Image must have x as its first dimension")
    end
end



#### Permutations over dimensions ####

# width and height, translating "x" and "y" spatialorder into horizontal and vertical, respectively

"""
`w, h = widthheight(img)` returns the width and height of an image, regardless of storage order.

See also: `width`, `height`.
"""
function widthheight(img::AbstractArray, p)
    c = coords_spatial(img)
    size(img, c[p[1]]), size(img, c[p[2]])
end
widthheight(img::AbstractArray) = widthheight(img, spatialpermutation(xy, img))


"""
`w = width(img)` returns the horizontal size of the image, regardless
of storage order. By default horizontal corresponds to dimension
`"x"`, but see `spatialpermutation` for other options.
"""
width(img::AbstractArray) = widthheight(img)[1]
"""
`h = height(img)` returns the vertical size of the image, regardless
of storage order. By default horizontal corresponds to dimension
`"y"`, but see `spatialpermutation` for other options.
"""
height(img::AbstractArray) = widthheight(img)[2]

"""
```
p = spatialpermutation(to, img)
```

Calculates the *spatial* permutation needed to convert the spatial dimensions to
a given order. This is probably easiest to understand by examples: for an
`Array` `A` of size `(m,n,3)`, `spatialorder(A)` would yield `["y", "x"]`, so
`spatialpermutation(["y", "x"], A) = [1,2]` and `spatialpermutation(["x", "y"],
A) = [2,1]`.  For an image type, here's a demonstration:

```
julia> Aimg = convert(Image, A)
RGB Image with:
  data: 4x5x3 Array{Float64,3}
  properties:
    colordim: 3
    spatialorder:  y x
    colorspace: RGB

julia> Ap = permutedims(Aimg, [3, 1, 2])
RGB Image with:
  data: 3x4x5 Array{Float64,3}
  properties:
    colordim: 1
    spatialorder:  y x
    colorspace: RGB

julia> spatialpermutation(["x","y"], Ap)
2-element Array{Int64,1}:
 2
 1
```
"""
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
permutedims(img::AbstractImage, p::Tuple{}, spatialprops::Vector = spatialproperties(img)) = img

function permutedims(img::AbstractImage, p::Union{Vector{Int}, Tuple{Vararg{Int}}}, spatialprops::Vector = spatialproperties(img))
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

permutedims{S<:AbstractString}(img::AbstractImage, pstr::Union{Vector{S}, Tuple{Vararg{S}}}, spatialprops::Vector = spatialproperties(img)) = permutedims(img, dimindexes(img, pstr...), spatialprops)

if VERSION < v"0.5.0-dev"
    permutedims(A::AbstractArray, p) = permutedims(convert(Array, A), p)
end

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
    p = collect(1:ndims(img))
    p[s] = s[2:-1:1]
    permutedims(img, p)
end

# Default list of spatial properties possessed by an image
"""
```
sp = spatialproperties(img)
```

Returns all properties whose values are of the form of an array or tuple, with
one entry per spatial dimension. If you have a custom type with additional
spatial properties, you can set `img["spatialproperties"] = ["property1",
"property2", ...]`. An advantage is that functions that change spatial
dimensions, like `permutedims` and `slice`, will also adjust the properties. The
default is `["spatialorder", "pixelspacing"]`; however, if you override the
setting then these are not included automatically (you'll want to do so
manually, if applicable).
"""
function spatialproperties(img::AbstractImage)
    if haskey(img, "spatialproperties")
        return img.properties["spatialproperties"]
    end
    spatialprops = String[]
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
spatialproperties(img::AbstractVector) = String[]  # these are not mutable


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
function coords(img::AbstractImage, dimname::String, ind, nameind...)
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


function dimindex(img::AbstractImage, dimname::String, so = spatialorder(img))
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

dimindexes(img::AbstractImage, dimnames::AbstractString...) = Int[dimindex(img, nam, spatialorder(img)) for nam in dimnames]

to_vector(v::AbstractVector) = v
to_vector(v::Tuple) = [v...]

# converts keyword argument to a dictionary
function kwargs2dict(kwargs)
    d = Dict{String,Any}()
    for (k, v) in kwargs
        d[string(k)] = v
    end
    return d
end

n_elts{C<:Colorant}(::Type{C}) = div(sizeof(C), sizeof(eltype(C)))
