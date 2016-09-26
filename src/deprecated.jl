export
    AbstractImage,
    AbstractImageDirect,
    AbstractImageIndexed,
    Image

const yx = ["y", "x"]
const xy = ["x", "y"]

SliceData(args...) = error("SliceData has been removed, please use julia's regular indexing operations")
reslice!(args...) = error("reslice! has been removed, along with SliceData; please use julia's regular indexing operations")
rerange!(args...) = error("reslice! has been removed, along with SliceData; please use julia's regular indexing operations")

# These should have been deprecated long ago
@deprecate uint32color(img) immap(mapinfo(UInt32, img), img)
@deprecate uint32color!(buf, img::AbstractArray) map!(mapinfo(UInt32, img), buf, img)
@deprecate uint32color!(buf, img::AbstractArray, mi::MapInfo) map!(mi, buf, img)
@deprecate uint32color!{T,N}(buf::Array{UInt32,N}, img::AbstractArray{T,N}) map!(mapinfo(UInt32, img), buf, img)
@deprecate uint32color!{T,N,N1}(buf::Array{UInt32,N}, img::ChannelView{T,N1}) map!(mapinfo(UInt32, img), buf, img, Val{1})
@deprecate uint32color!{T,N}(buf::Array{UInt32,N}, img::AbstractArray{T,N}, mi::MapInfo) map!(mi, buf, img)
@deprecate uint32color!{T,N,N1}(buf::Array{UInt32,N}, img::ChannelView{T,N1}, mi::MapInfo) map!(mi, buf, img, Val{1})

@deprecate flipx(img) flipdim(img, 2)
@deprecate flipy(img) flipdim(img, 1)
@deprecate flipz(img) flipdim(img, 3)

@deprecate ando3 KernelFactors.ando3
@deprecate ando4 KernelFactors.ando3
@deprecate ando5 KernelFactors.ando3
@deprecate gaussian2d() Kernel.gaussian(0.5)
@deprecate gaussian2d(σ::Number) Kernel.gaussian(σ)
@deprecate gaussian2d(σ::Number, filter_size) Kernel.gaussian((σ,σ), (filter_size...,))
@deprecate imaverage KernelFactors.boxcar
@deprecate imdog Kernel.DoG
@deprecate imlog Kernel.LoG
@deprecate imlaplacian Kernel.Laplacian

@deprecate extremefilt!(A::AbstractArray, ::Base.Order.ForwardOrdering, region=coords_spatial(A)) extremefilt!(A, max, region)
@deprecate extremefilt!(A::AbstractArray, ::Base.Order.ReverseOrdering, region=coords_spatial(A)) extremefilt!(A, min, region)
@deprecate extremefilt!{C<:AbstractRGB}(A::AbstractArray{C}, ::Base.Order.ForwardOrdering, region=coords_spatial(A)) extremefilt!(A, (x,y)->mapc(max,x,y), region)
@deprecate extremefilt!{C<:AbstractRGB}(A::AbstractArray{C}, ::Base.Order.ReverseOrdering, region=coords_spatial(A)) extremefilt!(A, (x,y)->mapc(min,x,y), region)

function restrict{S<:String}(img::AbstractArray, region::Union{Tuple{String,Vararg{String}}, Vector{S}})
    depwarn("restrict(img, strings) is deprecated, please use restrict(img, axes) with an AxisArray", :restrict)
    so = spatialorder(img)
    regioni = Int[]
    for i = 1:length(region)
        push!(regioni, require_dimindex(img, region[i], so))
    end
    restrict(img, regioni)
end

function magnitude_phase(img::AbstractArray, method::AbstractString, border::AbstractString="replicate")
    f = ImageFiltering.kernelfunc_lookup(method)
    depwarn("magnitude_phase(img, method::AbstractString, [border]) is deprecated, use magnitude_phase(img, $f, [border]) instead", :magnitude_phase)
    magnitude_phase(img, f, border)
end
