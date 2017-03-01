"""
```
chull = convexhull(img)
chull = convexhull(img, "points/boundary/filled")
```

Computes the convex hull of a binary image.
The function can return either of these-
- vector of vertices of the convex hull
- image with boundary pixels marked
- image with convex hull filled

In case the image isn't a binary image, it considers pixel intensity<255 as black.
"""

function drawline{T<:ColorTypes.Gray}(img::AbstractArray{T, 2}, y0::Int, x0::Int, y1::Int, x1::Int, color)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
 
    sx = x0 < x1 ? 1 : -1
    sy = y0 < y1 ? 1 : -1;
 
    err = (dx > dy ? dx : -dy) / 2
    
    while true
        img[y0, x0] = color
        (x0 != x1 || y0 != y1) || break
        e2 = err
        if e2 > -dx
            err -= dy
            x0 += sx
        end
        if e2 < dy
            err += dx
            y0 += sy
        end
    end

    img
end


function convexhull{T<:Images.Gray}(img::AbstractArray{T, 2}, returntype="points")

    if returntype!="points" && returntype!="boundary" && returntype!="filled"
        error("Invalid argument returntype")
    end

    for i in CartesianRange(size(img))
        if img[i]!=0 && img[i]!=1
            warn("Input image isn't binary!")
            warn("Function considers pixel intensity<255 as black")
            break
        end
    end

    function getboundarypoints{T<:Images.Gray}(img::AbstractArray{T, 2})

        points = CartesianIndex{2}[]

        for i in indices(img, 1)
            ones = find(img[i,:] .==1)

            if length(ones) == 1
                push!(points, CartesianIndex{2}(i, minimum(ones)))
            elseif length(ones)!=0
                push!(points, CartesianIndex{2}(i, minimum(ones)))
                push!(points, CartesianIndex{2}(i, maximum(ones)))
            end
        end
        return points
    end

    function right_oriented(ref, a, b)
        ((a[2]-ref[2])*(b[1]-ref[1])-(b[2]-ref[2])*(a[1]-ref[1])<0) ? true : false
    end

    function collinear(a, b, c)
        ((a[2]-c[2])*(b[1]-c[1])-(b[2]-c[2])*(a[1]-c[1])==0) ? true : false
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

    function drawboundary(convex_hull, img)
        chull = zeros(T, size(img))
        for point in convex_hull
            chull[point]=1
        end

        for i in 1:length(convex_hull)-1
            chull=drawline(chull, convex_hull[i][1], convex_hull[i][2], convex_hull[i+1][1], convex_hull[i+1][2], 1)
        end
        chull=drawline(chull, convex_hull[1][1], convex_hull[1][2], convex_hull[end][1], convex_hull[end][2], 1)

        return chull
    end

    function fillhull(convex_hull, img)
        chull = zeros(T, size(img))
        for point in convex_hull
            chull[point]=1
        end

        for i in 1:length(convex_hull)-1
            chull=drawline(chull, convex_hull[i][1], convex_hull[i][2], convex_hull[i+1][1], convex_hull[i+1][2], 1)
        end
        chull=drawline(chull, convex_hull[1][1], convex_hull[1][2], convex_hull[end][1], convex_hull[end][2], 1)

        for i in indices(img, 1)
            ones = find(chull[i,:] .==1)
            if length(ones) >= 2
                first = CartesianIndex{2}(i, ones[1])
                last = CartesianIndex{2}(i, ones[end])
                chull=drawline(chull, first[1], first[2], last[1], last[2], 1)
            end
        end

        return chull
    end


    # Used Graham scan algorithm

    points = getboundarypoints(img)
    last_point = CartesianIndex{2}(1, 1)
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
        while (right_oriented(convex_hull[end],convex_hull[end-1], points[i]))
            pop!(convex_hull)
        end
        push!(convex_hull, points[i])
    end

    if returntype=="points"
        return convex_hull
    elseif returntype=="boundary"
        return drawboundary(convex_hull, img)
    elseif returntype=="filled"
        return fillhull(convex_hull, img)
    end

end
