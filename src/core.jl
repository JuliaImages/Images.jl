#### Types and constructors ####

# Plain arrays can be treated as images. Other types will have
# metadata associated, make yours a child of one of the following:
abstract AbstractImage{T} <: AbstractArray{T}         # image with metadata
abstract AbstractImageDirect{T} <: AbstractImage{T}   # each pixel has own value/color
abstract AbstractImageIndexed{T} <: AbstractImage{T}  # indexed images (i.e., lookup table)

# Direct image (e.g., grayscale, RGB)
type Image{T,A<:AbstractArray} <: AbstractImageDirect{T}
    data::A
    properties::Dict
end
Image{A<:AbstractArray}(data::A, props::Dict) = Image{eltype(data),A}(data,props)
Image{A<:AbstractArray}(data::A) = Image(data,Dict{String,Any}())

# Indexed image (colormap)
type ImageCmap{T,A<:AbstractArray,C<:AbstractArray} <: AbstractImageIndexed{T}
    data::A
    cmap::C
    properties::Dict
end
ImageCmap{A<:AbstractArray,C<:AbstractArray}(data::A, cmap::C, props::Dict) = ImageCmap{eltype(data),A,C}(data, cmap, props)
ImageCmap{A<:AbstractArray,C<:AbstractArray}(data::A, cmap::C) = ImageCmap(data, cmap, Dict{String,Any}())

#### Core operations ####

size(img::AbstractImage) = size(img.data)
size(img::AbstractImage, i::Integer) = size(img.data, i)

ndims(img::AbstractImage) = ndims(img.data)

copy(img::AbstractImage) = deepcopy(img)
# copy, replacing the data
copy(img::Image, data::AbstractArray) = Image(data, copy(img.properties))
copy(img::ImageCmap, data::AbstractArray) = ImageCmap(data, copy(img.cmap), copy(img.properties))

similar{T}(img::Image, ::Type{T}, dims::Dims) = Image(similar(img.data, T, dims), copy(img.properties))
similar{T}(img::Image, ::Type{T}) = Image(similar(img.data, T), copy(img.properties))
similar(img::Image) = Image(similar(img.data), copy(img.properties))
similar{T}(img::ImageCmap, ::Type{T}, dims::Dims) = ImageCmap(similar(img.data, T, dims), copy(img.cmap), copy(img.properties))
similar{T}(img::ImageCmap, ::Type{T}) = ImageCmap(similar(img.data, T), copy(img.cmap), copy(img.properties))
similar(img::ImageCmap) = ImageCmap(similar(img.data), copy(img.cmap), copy(img.properties))

convert(::Type{Image}, img::Image) = img
function convert(::Type{Image}, img::ImageCmap)
    local data
    local prop
    if size(img.cmap, 2) == 1
        data = reshape(img.cmap[img.data[:]], size(img.data))
        prop = img.properties
    else
        newsz = tuple(size(img.data)...,size(img.cmap,2))
        data = reshape(img.cmap[img.data[:],:], newsz)
        prop = copy(img.properties)
        prop["colordim"] = length(newsz)
        indx = Base.ht_keyindex(prop, "storageorder")
        if indx > 0
            prop.vals[indx] = [prop.vals[indx], "color"]
        end
    end
    Image(data, prop)
end

assign(img::AbstractImage, X, i::Real) = assign(img.data, X, i)
assign{T<:Real}(img::AbstractImage, X, I::Union(Real,AbstractArray{T})...) = assign(img.data, X, I...)

# ref and sub return a value or AbstractArray, not an Image
ref(img::AbstractImage, i::Real) = ref(img.data, i)
ref{T<:Real}(img::AbstractImage, I::Union(Real,AbstractArray{T})...) = ref(img.data, I...)
sub{T<:Real}(img::AbstractImage, I::RangeIndex...) = sub(img.data, I...) # needed because subarray not in sync with array
sub{T<:Real}(img::AbstractImage, I::Union(Real,AbstractArray{T})...) = sub(img.data, I...)

# refim and subim return an Image
refim{T<:Real}(img::AbstractImage, I::Union(Real,AbstractArray{T})...) = copy(img, ref(img.data, I...))
subim{T<:Real}(img::AbstractImage, I::Union(Real,AbstractArray{T})...) = copy(img, sub(img.data, I...))

function show(io::IO, img::AbstractImageDirect)
    IT = typeof(img)
    print(io, colorspace(img), " ", IT.name, " with:\n  data: ", summary(img.data), "\n  properties: ", img.properties)
end
function show(io::IO, img::AbstractImageIndexed)
    IT = typeof(img)
    print(io, colorspace(img), " ", IT.name, " with:\n  data: ", summary(img.data), "\n  cmap: ", summary(img.cmap), "\n  properties: ", img.properties)
end

data(img::AbstractArray) = img
data(img::AbstractImage) = img.data

min(img::AbstractImageDirect) = min(img.data)
max(img::AbstractImageDirect) = max(img.data)
# min/max deliberately not defined for AbstractImageIndexed

#### Properties ####

# Generic programming with images uses properties to obtain information. The strategy is to define a particular property name, and then write an accessor function of the same name. The accessor function provides default behavior for plain arrays and when the property is not defined. Alternatively, use get(img, "propname", default) or has(img, "propname") to define your own default behavior.

# You can define whatever properties you want. Here is a list of properties that are used in some algorithms:
#   colorspace: "RGB", "RGBA", "Gray", "Binary", "Lab", "HSV", etc.
#   colordim: the array dimension used to store color information, or 0 if not present
#   seqdim: the array dimension used for time (i.e., sequence), or 0 for single images
#   limits: (minvalue,maxvalue) for this type of image (e.g., (0,255) for Uint8 images, even if pixels do not reach these values)
#   pixelspacing: the spacing between adjacent pixels along spatial dimensions
#   storageorder: a string naming each dimension (names can be arbitrary and hence are not used in generic algorithms)

has(a::AbstractArray, k::String) = false
has(img::AbstractImage, k::String) = has(img.properties, k)

get(img::AbstractArray, k::String, default) = default
get(img::AbstractImage, k::String, default) = get(img.properties, k, default)

# So that defaults don't have to be evaluated unless they are needed, we also define a @get macro (thanks Toivo Hennington):
macro get(img, k, default)
    quote
        img, k = $(esc(img)), $(esc(k))
        local val
        if isa(img, StridedArray)
            val = $(esc(default))
        else
            index = Base.ht_keyindex(img.properties, k)
            val = (index > 0) ? img.properties.vals[index] : $(esc(default))
        end
        val
    end
end


colorspace(img::AbstractArray{Bool}) = "Binary"
colorspace(img::AbstractArray{Bool,3}) = "Binary"
colorspace(img::AbstractArray) = "Gray"
colorspace{T}(img::AbstractArray{T,3}) = (size(img, 3) == 3) ? "RGB" : error("Cannot infer colorspace of Array, use an AbstractImage type")
colorspace(img::AbstractImage{Bool}) = "Binary"
colorspace(img::AbstractImage) = get(img.properties, "colorspace", "Gray")

colordim{T}(img::AbstractArray{T,3}) = (size(img, 3) == 3) ? 3 : error("Cannot infer colordim of Array, use an AbstractImage type")
colordim(img::AbstractImageDirect) = get(img, "colordim", 0)
colordim(img::AbstractImageIndexed) = 0

seqdim(img) = get(img, "seqdim", 0)

limits(img::AbstractArray{Bool}) = 0,1
limits{T<:Integer}(img::AbstractArray{T}) = typemin(T), typemax(T)
limits{T<:FloatingPoint}(img::AbstractArray{T}) = zero(T), one(T)
limits(img::AbstractImage{Bool}) = 0,1
limits{T}(img::AbstractImageDirect{T}) = get(img, "limits", (typemin(T), typemax(T)))
limits(img::AbstractImageIndexed) = @get img "limits" (min(img.cmap), max(img.cmap))

pixelspacing{T}(img::AbstractArray{T,3}) = (size(img, 3) == 3) ? [1.0,1.0] : error("Cannot infer pixelspacing of Array, use an AbstractImage type")
pixelspacing(img::AbstractMatrix) = [1.0,1.0]
pixelspacing(img::AbstractImage) = @get img "pixelspacing" _pixelspacing(img)
_pixelspacing(img::AbstractImage) = ones(sdims(img))

# number of spatial dimensions in the image
sdims(img) = ndims(img) - (colordim(img) != 0) - (seqdim(img) != 0)

storageorder(img::AbstractMatrix) = ["y", "x"]
storageorder{T}(img::AbstractArray{T,3}) = (size(img, 3) == 3) ? ["y", "x", "c"] : error("Cannot infer storageorder of Array, use an AbstractImage type")
storageorder(img::AbstractImage) = img.properties["storageorder"]

function coords_spatial(img)
    ind = [1:ndims(img)]
    cdim = colordim(img)
    if cdim > 0
        ind = setdiff(ind, cdim)
    end
    tdim = seqdim(img)
    if tdim > 0
        ind = setdiff(ind, tdim)
    end
    ind
end

function size_spatial(img)
    sz = size(img)
    sz[coords_spatial(img)]
end
