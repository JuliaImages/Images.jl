@deprecate otsu_threshold(img::AbstractArray{T}, nbins::Int = 256) where {T<:Union{Gray,Real}} T(find_threshold(img, Otsu(); nbins))
@deprecate yen_threshold(img::AbstractArray{T}, nbins::Int = 256) where {T<:Union{Gray,Real}} T(find_threshold(img, Yen(); nbins))
