"""
```
chull = convexhull(img)
```
Computes the convex hull of a binary image and returns the vertices of convex hull as a CartesianIndex array.
"""
function convexhull(img::AbstractArray{T, 2}) where T<:Union{Bool,Gray{Bool}}

    function getboundarypoints(img)
        points = CartesianIndex{2}[]
        for j = axes(img, 2)
            v = Base.view(img, :, j)
            i1 = findfirst(v)
            if i1 != nothing
                i2 = findlast(v)
                push!(points, CartesianIndex(i1, j))
                if i1 != i2
                    push!(points, CartesianIndex(i2, j))
                end
            end
        end
        points
    end

    function right_oriented(ref, a, b)
        (a[2]-ref[2])*(b[1]-ref[1])-(b[2]-ref[2])*(a[1]-ref[1])<0
    end

    function collinear(a, b, c)
        (a[2]-c[2])*(b[1]-c[1])-(b[2]-c[2])*(a[1]-c[1])==0
    end

    dist2(a, b) = sum(abs2, (a-b).I)

    function angularsort!(points, ref, img)
        last_point = ref

        for i in eachindex(points)
            if points[i]==last_point
                deleteat!(points, i)
                break
            end
        end

        sort!(points, lt=(a, b) -> right_oriented(last_point, a, b))

        i=0
        while i<=length(points)
            i=i+1
            if i+1>length(points)
                break
            end

            if collinear(last_point, points[i], points[i+1])
                if dist2(last_point, points[i]) < dist2(last_point, points[i+1])
                    deleteat!(points, i)
                else
                    deleteat!(points, i+1)
                end
                i=i-1
            end
        end
    end

    # Used Graham scan algorithm

    points=getboundarypoints(img)
    last_point=CartesianIndex(map(r->first(r)-1, axes(img)))
    for point in points
        if point[2]>last_point[2] || (point[2]==last_point[2] && point[1]>last_point[1])
            last_point=point
        end
    end
    angularsort!(points, last_point, img)

    n_points=length(points)

    if n_points<3
        error("Not enough points to compute convex hull.")
    end

    convex_hull=CartesianIndex{2}[]
    push!(convex_hull, last_point)
    push!(convex_hull, points[1])
    push!(convex_hull, points[2])

    for i in 3:n_points
        while (right_oriented(convex_hull[end],convex_hull[end-1], points[i]) || collinear(convex_hull[end],convex_hull[end-1], points[i]))
            pop!(convex_hull)
        end
        push!(convex_hull, points[i])
    end

    return convex_hull
end
