"""
```
chull = convexhull(img)
chull = convexhull(img, "hull boundary/filled hull")
```

Computes the convex hull of a binary image.
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


function convexhull{T<:Images.Gray}(img::AbstractArray{T, 2}, returntype="hull boundary")

    if returntype!="hull boundary" && returntype!="filled hull"
        error("Invalid argument returntype")
    end

    points = []
    h, w = size(img)

    #just store first and last point in each row for computing convex hull
    for i in 1:h
        first_point_found=false
        last_point_found=false
        first_point=[]
        last_point=[]
        for j in 1:w
            if !first_point_found && img[i, j]==1
                first_point=[i, j]
                first_point_found=true

            elseif first_point_found && img[i, j]==1
                last_point=[i, j]
                last_point_found=true
            end
        end
        if first_point_found
            push!(points, first_point)
        end
        if last_point_found
            push!(points, last_point)
        end
    end

    # Used Graham scan algorithm

    last_point=[-1, -1]
    for i in 1:h
        for j in 1:w
            if img[i,j]==1 && j>last_point[2]
                last_point=[i, j]
            elseif img[i,j]==1 && j==last_point[2] && i>last_point[1]
                last_point=[i, j]
            end
        end
    end

    function right_oriented(a, b)
        ref=last_point
        ((a[2]-ref[2])*(b[1]-ref[1])-(b[2]-ref[2])*(a[1]-ref[1])<0) ? true : false
    end

    function right_oriented(ref, a, b)
        ((a[2]-ref[2])*(b[1]-ref[1])-(b[2]-ref[2])*(a[1]-ref[1])<0) ? true : false
    end

    function collinear(a, b, c)
        ((a[2]-c[2])*(b[1]-c[1])-(b[2]-c[2])*(a[1]-c[1])==0) ? true : false
    end

    for i in 1:size(points)[1]
        if(points[i]==last_point)
            deleteat!(points, i)
            break
        end
    end

    #angular sort wrt last_point
    sort!(points, lt=right_oriented)

    #if polar angle of 2 points wrt last_point is same, just keep farther point
    i=0
    while i<=size(points)[1]
        i=i+1
        if i+1>size(points)[1]
            break
        end

        if(collinear(last_point,points[i], points[i+1]))
            if (points[i][1]-last_point[1])^2 + (points[i][2]-last_point[2])^2< (points[i+1][1]-last_point[1])^2 + (points[i+1][2]-last_point[2])^2
                deleteat!(points, i)
            else 
                deleteat!(points, i+1)
            end
            i=i-1
        end
    end

    n_points=size(points)[1]

    if n_points<3
        error("Not enough points to compute convex hull.")
    end

    convex_hull=[]
    push!(convex_hull, last_point)
    push!(convex_hull, points[1])
    push!(convex_hull, points[2])

    for i in 3:n_points
        while (right_oriented(convex_hull[end],convex_hull[end-1], points[i]))
            pop!(convex_hull)
        end
        push!(convex_hull, points[i])
    end


    chull = zeros(T, h, w)
    for i = 1:size(convex_hull)[1]
        chull[convex_hull[i][1], convex_hull[i][2]]=1
    end

    for i=1:size(convex_hull)[1]-1
        chull=drawline(chull, convex_hull[i][1], convex_hull[i][2], convex_hull[i+1][1], convex_hull[i+1][2], 1)
    end
    chull=drawline(chull, convex_hull[1][1], convex_hull[1][2], convex_hull[end][1], convex_hull[end][2], 1)

    if returntype=="hull boundary"
        return chull
    end

    # fill convex hull
    for i in 1:h
        first_point_found=false
        last_point_found=false
        first_point=[]
        last_point=[]
        for j in 1:w
            if !first_point_found && chull[i, j]==1
                first_point=[i, j]
                first_point_found=true

            elseif first_point_found && chull[i, j]==1
                last_point=[i, j]
                last_point_found=true
            end
        end
        if first_point_found && last_point_found
            chull=drawline(chull, first_point[1], first_point[2], last_point[1], last_point[2], 1)
        end
    end

    if returntype=="filled hull"
        return chull
    end

end
