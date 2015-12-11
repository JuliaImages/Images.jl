#### Similarity and dissimilarity measures between images ####

"""
`MHD(imgA, imgB)` is the modified Hausdorff distance between binary
images (or point sets).

References
----------
Dubuisson, M-P; Jain, A. K., 1994. A Modified Hausdorff Distance for
Object-Matching.
"""
function MHD(imgA::AbstractArray, imgB::AbstractArray)
  @assert size(imgA) == size(imgB) "Images must have the same dimensions"

  # trivial case
  isequal(imgA, imgB) && return 0.

  A, B = find(imgA), find(imgB)

  # return if there is no object to match
  (isempty(A) || isempty(B)) && return Inf

  # grid coordinates (ndims by npoints)
  A = hcat([Float64[ind2sub(size(imgA), a)...] for a in A]...)
  B = hcat([Float64[ind2sub(size(imgB), b)...] for b in B]...)

  m = size(A, 2); n = size(B, 2)

  D = zeros(m, n)
  for j=1:n, i=1:m
    @inbounds D[i,j] = norm(A[:,i] - B[:,j])
  end

  dAB = mean(minimum(D, 2))
  dBA = mean(minimum(D, 1))

  max(dAB, dBA)
end
