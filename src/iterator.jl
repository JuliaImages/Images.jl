module Iter

import Base.assign, Base.done, Base.next, Base.ref, Base.start

parent(A::Array) = A
parent(A::SubArray) = A.parent
first_index(A::Array) = 1
first_index(A::SubArray) = A.first_index

function Iterator(A::StridedArray, dims, slices)
    notdims = setdiff(1:ndims(A), [dims...])
    if length(slices) != length(notdims)
        error("The dimensionalities do not add up")
    end
    sz = size(A)
    s = strides(A)
    N = length(dims)
    f = first_index(A)
    for i = 1:length(slices)
        if ~isa(slices[i], Integer)
            error("slices must be integers")
        end
        if !(1 <= slices[i] <= sz[i])
            throw(BoundsError())
        end
        f += (slices[i]-1)*s[notdims[i]]
    end
    if N == 2
        return Iterator2D(sz[dims[1]], sz[dims[2]], s[dims[1]], s[dims[2]], f)
    else
        error(N, " dimensions not yet supported")
    end
end
Iterator(A::StridedArray) = Iterator(A, 1:ndims(A), ())

#### Iterator definitions ####
# Iterator stores size information, IteratorState holds the current state

abstract IteratorState

# 2D
immutable Iterator2D
    ni::Int
    nj::Int
    stridei::Int
    stridej::Int
    first_index::Int
end

immutable Iterator2DState <: IteratorState
    i::Int
    j::Int
    index::Int
end

start(iter::Iterator2D) = Iterator2DState(1, 1, iter.first_index)

next(iter::Iterator2D, state::Iterator2DState) = (state.i+1 <= iter.ni) ? (state, Iterator2DState(state.i+1, state.j, state.index + iter.stridei)) : (state, Iterator2DState(1, state.j+1, iter.first_index+state.j*iter.stridej))

done(iter::Iterator2D, state::Iterator2DState) = state.j > iter.nj


#### ref/assign ####

ref(A::StridedArray, state::IteratorState) = parent(A)[state.index]

assign{T}(A::StridedArray{T}, x::T, state::IteratorState) = assign(parent(A), x, state.index)

end
