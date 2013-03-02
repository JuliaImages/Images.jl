# Plain arrays can be treated as images. Other types will have
# metadata associated, make yours a child of one of the following:
abstract AbstractImage{T} <: AbstractArray{T}      # image with metadata
abstract AbstractImageDirect{T} <: AbstractImage{T}        # each pixel has own value/color
abstract AbstractImageIndexed{T} <: AbstractImage{T}       # indexed images (i.e., lookup table)

# Types to tag the colorspace
abstract ColorSpace
type RGB <: ColorSpace end
type RGBA <: ColorSpace end
type Gray <: ColorSpace end
type LAB <: ColorSpace end
type HSV <: ColorSpace end

# Generic programming with images---whether represented as plain arrays or as types with metadata---uses the following accessor functions:
#  


##### Treating plain arrays as images #####

# The following functions allow plain arrays to be treated as images
# for all image-processing functions:
# storageorder(img::Matrix) = ["y","x"]     # vertical-major storage order
# function storageorder{T}(img::Array{T,3})
#     if size(img, 3) == 3
#         return ["y","x","c"]              # an rgb image
#     else
#         error("Cannot infer storage order of Array, use a type with metadata")
#     end
# end
# # You could write the following generically in terms of the above, but
# # for efficiency it's better to have a direct implementation:
# colorspace(img::Matrix) = CSgray
# function colorspace{T}(img::Array{T,3})
#     if size(img, 3) == 3
#         return CSsRGB
#     else
#         error("Cannot infer colorspace of Array, use a type with metadata")
#     end
# end
# ncoords(img::Matrix) = 2
# function ncoords{T}(img::Array{T,3})
#     if size(img, 3) == 3
#         return 2
#     else
#         error("Cannot infer the number of spatial dimensions of Array, use a type with metadata")
#     end
# end

data(img::Array) = img

##### Basic image types with metadata #####

# Image with a defined color space and storage order
type ImageCS{T,A<:AbstractArray,CS<:ColorSpace} <: AbstractImageDirect{T}
    data::A
    order::Vector{ASCIIString}  # storage order, e.g., ["y","x","c"]
    
    function ImageCS(data::A, order::Vector{ASCIIString}, ::Type{CS})
        check_colordim(CS, order)
        new(data, order)
    end
end
ImageCS{T,A<:AbstractArray{T},CS<:ColorSpace}(data::A, order::Vector{ASCIIString}, ::Type{CS}) = ImageCS{T,A,CS}(data,order)

# Image with color space and explicit pixel min/max values (useful for contrast scaling)
type ImageCSMinMax{T,A<:AbstractArray{T},CS<:ColorSpace} <: AbstractImageDirect{T}
    data::A
    order::Vector{ASCIIString}
    min::T
    max::T

    function ImageCSMinMax(data::A, order::Vector{ASCIIString}, mn, mx)
        check_colordim(CS, order)
        new(data, order, convert(T, mn), convert(T, mx))
    end
end
ImageCSMinMax{T,A<:AbstractArray{T},CS<:ColorSpace}(data::A, order::Vector{ASCIIString}, ::Type{CS}, mn, mx) = ImageCSMinMax{T,A,CS}(data,order,mn,mx)

# Indexed image (colormap)
type ImageCmap{T,A<:AbstractArray,C<:AbstractArray,CS<:ColorSpace} <: AbstractImageIndexed{T}
    data::A
    order::Vector{ASCIIString}
    cmap::C
end
ImageCmap{T,A<:AbstractArray{T},C<:AbstractArray,CS<:ColorSpace}(data::A, order::Vector{ASCIIString}, ::Type{CS}, cmap::C) = ImageCmap{T,A,C,CS}(data, order, cmap)

data(img::AbstractImage) = img.data

assign{IM<:AbstractImage}(img::IM, X, ind...) = assign(img.data, X, ind...)
# ref and sub return a value or AbstractArray, not an Image
ref{IM<:AbstractImage}(img::IM, ind...) = ref(img.data, ind...)
sub{IM<:AbstractImage}(img::IM, ind...) = sub(img.data, ind...)
# refim and subim return an Image
refim{T,A<:AbstractArray,CS<:ColorSpace}(img::ImageCS{T,A,CS}, ind...) = ImageCS(ref(img.data, ind...), img.order, CS)
refim{T,A<:AbstractArray,CS<:ColorSpace}(img::ImageCSMinMax{T,A,CS}, ind...) = ImageCS(ref(img.data, ind...), img.order, CS, img.min, img.max)
refim{T,A<:AbstractArray,C<:AbstractArray,CS<:ColorSpace}(img::ImageCmap{T,A,C,CS}, ind...) = ImageCmap(ref(img.data, ind...), img.order, CS, cmap)


copy(img::AbstractImage) = deepcopy(img)

# ncoords{T,N}(img::Union(ImageCS{T,N}, ImageCSMinMax{T,N}, ImageColormap{T,N})) = N
# 
# storageorder(img::Union(ImageCS, ImageCSMinMax, ImageColormap)) = img.order
# 
# colorspace{T, N, CS<:ColorSpace}(img::Union(ImageCS{T,N,CS}, ImageCSMinMax{T,N,CS}, ImageColormap{T,N,CS})) = CS
# 
# function size_spatial(img::Union(Array,AbstractImageDirect))
#     order = storageorder(img)
#     data = pixeldata(img)
#     m = match(r"c", order)
#     if m == nothing
#         return size(data)
#     else
#         return ntuple(ncoords(img), i->size(data, i+(i >= m.offset)))
#     end
# end
# size_spatial(img::ImageColormap) = size(img.data)
# 
# # The number of color channels
# function ncolors(img::Union(Array,AbstractImageDirect))
#     order = storageorder(img)
#     data = pixeldata(img)
#     m = match(r"c", order)
#     if m == nothing
#         return 1
#     else
#         return size(data, m.offset)
#     end
# end

clim_min{T<:Float}(img::Union(Array{T}, ImageCS{T}, ImageColormap{T})) = zero(T)
clim_max{T<:Float}(img::Union(Array{T}, ImageCS{T}, ImageColormap{T})) = one(T)
clim_min{T<:Integer}(img::Union(Array{T}, ImageCS{T}, ImageColormap{T})) = typemin(T)
clim_max{T<:Integer}(img::Union(Array{T}, ImageCS{T}, ImageColormap{T})) = typemax(T)
clim_min(img::ImageCSMinMax) = img.min
clim_max(img::ImageCSMinMax) = img.max


_show(io::IO, img::AbstractImage) = print(io, typeof(img), "\n  data: ", summary(img.data), "\n  order: ", show(io, img.order))
show(io::IO, img::AbstractImage) = _show(io, img)
function show(io::IO, img::ImageCSMinMax)
    _show(io, img)
    print(io, "\n  minmax: ", img.min, ", ", img.max)
end
function show(io::IO, img::ImageCmap)
    _show(io, img)
    print(io, "\n  map: ", summary(img.map))
end


check_colordim(::Type{Gray}, order::Vector{ASCIIString}) = nothing
function check_colordim{CS<:ColorSpace}(::Type{CS}, order::Vector{ASCIIString})
    havec = false
    for s in order
        if s == "c"
            havec = true
            break
        end
    end
    if !havec
        error("order array does not indicate the color dimension")
    end
end
