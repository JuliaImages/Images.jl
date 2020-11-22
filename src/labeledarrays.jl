struct ColorizedArray{C<:Colorant,N,A<:AbstractArray,L<:AbstractArray} <: AbstractArray{C,N}
    intensity::A
    label::L
end

"""
    ColorizedArray(intensity, label::IndirectArray) -> A

Create an array, combining a `label` array (where each pixel is
assigned one of a list of discrete colors) and an `intensity` array
(where each pixel has a scalar value). `A` satisfies

    A[i,j,...] = intensity[i,j,...] * label[i,j,...]

The label array "tinges" the grayscale intensity with the color
associated with that point's label.

This computation is performed lazily, as to be suitable even for large arrays.
"""
function ColorizedArray(intensity, label::IndirectArray{C,N}) where {C<:Colorant,N}
    depwarn("`ColorizedArray(intensity, label)` is deprecated, use `mappedarray(*, intensity, label)` from `MappedArrays` instead", :ColorizedArray)

    axes(intensity) == axes(label) || throw(DimensionMismatch("intensity and label must have the same axes, got $(axes(intensity)) and $(axes(label))"))
    CI = typeof(zero(C)*zero(eltype(intensity)))
    ColorizedArray{CI,N,typeof(intensity),typeof(label)}(intensity, label)
end

# TODO: an implementation involving AxisArray that matches the shared
# axes, and therefore allows `label` to be of lower dimensionality
# than `intensity`.

intensitytype(::Type{ColorizedArray{C,N,A,L}}) where {C<:Colorant,N,A<:AbstractArray,L<:AbstractArray} = A
labeltype(::Type{ColorizedArray{C,N,A,L}}) where {C<:Colorant,N,A<:AbstractArray,L<:AbstractArray} = L

Base.size(A::ColorizedArray) = size(A.intensity)
Base.axes(A::ColorizedArray) = axes(A.intensity)
Base.IndexStyle(::Type{CA}) where {CA<:ColorizedArray} = IndexStyle(IndexStyle(intensitytype(CA)), IndexStyle(labeltype(CA)))

@inline function Base.getindex(A::ColorizedArray, i::Integer)
    @boundscheck checkbounds(A, i)
    @inbounds ret = A.intensity[i]*A.label[i]
    ret
end
@inline function Base.getindex(A::ColorizedArray{C,N}, I::Vararg{Int,N}) where {C,N}
    @boundscheck checkbounds(A, I...)
    @inbounds ret = A.intensity[I...]*A.label[I...]
    ret
end
