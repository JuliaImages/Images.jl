#### Similarity and dissimilarity measures between images ####

"""
`hausdorff_distance(imgA, imgB)` is the modified Hausdorff distance
between binary images (or point sets).

References
----------
Dubuisson, M-P; Jain, A. K., 1994. A Modified Hausdorff Distance for
Object-Matching.
"""
function hausdorff_distance(imgA::AbstractArray, imgB::AbstractArray)
  Base.depwarn("Images.hausdorff_distance(A, B) is deprecated and will be removed in the future. Image distances were moved and expanded upon in its own dedicated package ImageDistances. Please use ImageDistances.modified_hausdorf(A, B) instead." , :hausdorff_distance)
  @assert size(imgA) == size(imgB) "Images must have the same dimensions"

  # trivial case
  isequal(imgA, imgB) && return 0.

  A, B = find(imgA), find(imgB)

  # return if there is no object to match
  (isempty(A) || isempty(B)) && return Inf

  m = length(A); n = length(B)

  D = zeros(m, n)
  for j=1:n
    b = [ind2sub(size(imgB), B[j])...]
    for i=1:m
      a = [ind2sub(size(imgA), A[i])...]
      @inbounds D[i,j] = norm(a - b)
    end
  end

  dAB = mean(minimum(D, 2))
  dBA = mean(minimum(D, 1))

  max(dAB, dBA)
end
