parent(A::Array) = A
parent(A::SubArray) = A.parent
parent(img::AbstractImage) = parent(data(img))
first_index(A::Array) = 1
first_index(A::SubArray) = A.first_index

function iterate_spatial(img::AbstractArray)
    sz = size(img)
    s = strides(img)
    colorsz = 1
    colorstride = 0
    timesz = 1
    timestride = 0
    cd = colordim(img)
    if cd != 0
        colorsz = sz[cd]
        colorstride = s[cd]
    end
    td = timedim(img)
    if td != 0
        timesz = sz[td]
        timestride = s[td]
    end
    cspatial = setdiff(1:ndims(img), [cd, td])
    first_index(data(img)), sz[cspatial], s[cspatial], colorsz, colorstride, timesz, timestride
end

# module Iter
# 
# import Base.assign, Base.done, Base.next, Base.ref, Base.start
# 
# export Iterator
# 
# 
# function parsedims(A::StridedArray, dims, slices)
#     notdims = setdiff(1:ndims(A), [dims...])
#     if length(slices) != length(notdims)
#         error("The dimensionalities do not add up")
#     end
#     sz = size(A)
#     s = strides(A)
#     f = first_index(A)
#     for i = 1:length(slices)
#         if ~isa(slices[i], Integer)
#             error("slices must be integers")
#         end
#         if !(1 <= slices[i] <= sz[i])
#             throw(BoundsError())
#         end
#         f += (slices[i]-1)*s[notdims[i]]
#     end
#     sz, s, f
# end    
# 
# function Iterator(A::StridedArray, dims::NTuple{1,Int}, slices)
#     sz, s, f = parsedims(A, dims, slices)
#     Iterator1D(sz[dims[1]], s[dims[1]], f)
# end
# 
# function Iterator(A::StridedArray, dims::NTuple{2,Int}, slices)
#     sz, s, f = parsedims(A, dims, slices)
#     Iterator2D(sz[dims[1]], sz[dims[2]], s[dims[1]], s[dims[2]], f)
# end
# Iterator(A::StridedArray) = Iterator(A, ntuple(ndims(A), i->i), ())
# 
# #### Iterator definitions ####
# # Iterator stores size information, IteratorState holds the current state
# 
# abstract IteratorState
# 
# # 1D
# immutable Iterator1D
#     ni::Int
#     stridei::Int
#     first_index::Int
# end
# 
# immutable Iterator1DState <: IteratorState
#     i::Int
#     index::Int
# end
# 
# start(iter::Iterator1D) = Iterator1DState(1, iter.first_index)
# 
# # next(iter::Iterator1D, state::Iterator1DState) = state, Iterator1DState(state.i+1, state.index + iter.stridei)
# next(iter::Iterator1D, state::Iterator1DState) = state, Iterator1DState(state.i+1, state.index+1)
# 
# done(iter::Iterator1D, state::Iterator1DState) = state.i > iter.ni
# 
# 
# 
# # 2D
# immutable Iterator2D
#     ni::Int
#     nj::Int
#     stridei::Int
#     stridej::Int
#     first_index::Int
# end
# 
# immutable Iterator2DState <: IteratorState
#     i::Int
#     j::Int
#     index::Int
# end
# 
# start(iter::Iterator2D) = Iterator2DState(1, 1, iter.first_index)
# 
# next(iter::Iterator2D, state::Iterator2DState) = (state.i+1 <= iter.ni) ? (state, Iterator2DState(state.i+1, state.j, state.index + iter.stridei)) : (state, Iterator2DState(1, state.j+1, iter.first_index+state.j*iter.stridej))
# 
# done(iter::Iterator2D, state::Iterator2DState) = state.j > iter.nj
# 
# 
# #### ref/assign ####
# 
# ref(A::StridedArray, state::IteratorState) = parent(A)[state.index]
# 
# assign{T}(A::StridedArray{T}, x::T, state::IteratorState) = assign(parent(A), x, state.index)
# 
# end
