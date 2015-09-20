type LabeledArray{T,N,A<:AbstractArray,L<:AbstractArray} <: AbstractArray{T,N}
    data::A
    label::L
    colors::Vector{RGB}

    # Enforce that the label has Integer type
    LabeledArray{Ti<:Integer}(data::AbstractArray{T,N}, label::AbstractArray{Ti}, colors::Vector{RGB}) =
        new(data, label, colors)
end
LabeledArray{T,N}(data::AbstractArray{T,N}, label::AbstractArray, colors::Vector{RGB}) =
    LabeledArray{T,N,typeof(data),typeof(label)}(data, label, colors)

size(A::LabeledArray) = size(A.data)
size(A::LabeledArray, i::Integer) = size(A.data, i)
eltype{T}(A::LabeledArray{T}) = T
ndims{T,N}(A::LabeledArray{T,N}) = N

for N = 1:4
    @eval begin
        # All but the last of these are inside the @eval loop simply to avoid ambiguity warnings.
        # These first two are additional ones needed to avoid ambiguity warnings.
        _uint32color_gray!{T}(buf::Array{UInt32}, A::LabeledArray{T,$N}, mapi::ScaleSigned) = error("Cannot use ScaleSigned with a labeled array")
        _uint32color_gray!{T,L<:LabeledArray}(buf::Array{UInt32}, A::SubArray{T,$N,L}, mapi::ScaleSigned) = error("Cannot use ScaleSigned with a labeled array")

        function _uint32color_gray!{T}(buf::Array{UInt32}, A::LabeledArray{T,$N}, mapi::MapInfo = mapinfo(UInt8, A))
            if size(buf) != size(A)
                error("Size mismatch")
            end
            dat = A.data
            label = A.label
            col = A.colors
            for i = 1:length(dat)
                gr = map(mapi, dat[i])
                lbl = label[i]
                if lbl == 0
                    buf[i] = rgb24(gr,gr,gr)
                else
                    buf[i] = convert(RGB24, gr*col[lbl])
                end
            end
            buf
        end

        # For SubArrays, we can't efficiently use linear indexing, and in any event
        # we want to broadcast label where necessary
        function _uint32color_gray!{T,A<:LabeledArray}(buf::Array{UInt32}, S::SubArray{T,$N,A}, mapi::MapInfo = mapinfo(UInt8, A))
            if size(buf) != size(S)
                error("Size mismatch")
            end
            indexes = S.indexes
            dat = slice(S.parent.data, indexes)
            plabel = S.parent.label
            newindexes = RangeIndex[size(plabel,i)==1 ? (isa(indexes[i], Int) ? 1 : (1:1)) : indexes[i]  for i = 1:ndims(plabel)]
            label = slice(plabel, newindexes...)
            col = S.parent.colors
            _uint32color_labeled(buf, dat, label, col, mapi) # type of label can't be inferred, use function boundary
        end

        function _uint32color_labeled{T}(buf, dat::AbstractArray{T,$N}, label, col, mapi)
            k = 0
            @inbounds @nloops $N i buf begin
                val = @nref $N dat i
                gr = map(mapi, val)
                lbl = @nref $N label i
                if lbl == 0
                    buf[k+=1] = rgb24(gr,gr,gr)
                else
                    buf[k+=1] = convert(RGB24, gr*col[lbl])
                end
            end
            buf
        end
    end
end
