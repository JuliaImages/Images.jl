using Images, ImageProjectiveGeometry, ImageFiltering, ImageView, PaddedViews, Interpolations

function warpfast(img::AbstractArray, M::AbstractArray, output_shape::Int64= 0, order::Int64 =1, mode="constant", cval::Int64 =0)
    mode_c = Int64(uppercase(mode[1]))

    println("axes  ", length(axes(img,1))," type  ", typeof(length(axes(img,1))))
    if output_shape == 0
        out_r = length(axes(img,1))
        out_c = length(axes(img,2))
    else
        out_r = output_shape
        out_c = output_shape
    end

    out = zeros(Int64,(out_r, out_c))

    
    newimg = imgtrans(img, M)
    
    if order == 3
        
        itp = interpolate(newimg, (BSpline(Constant()), Constant()), OnGrid())
    elseif order == 1
        
        itp = interpolate(newimg, (BSpline(Linear()), Linear()), OnGrid())
    elseif order == 2
        itp = interpolate(newimg, (BSpline(Quadratic()), Quadratic()), OnGrid())
    elseif order == 3
        itp = interpolate(newimg, (BSpline(Cubic()), Cubic()), OnGrid())
    end
    for tfr in range(out_r)
        for tfc in range(out_c)
            out[tfr, tfc] = itp[tfr, tfc]
        end
    end
    return out
end

ogrid(ys, xs) = [[[xs[i]] for i in 1:size(xs,1)], [ys[j] for j in 1:size(ys)]]

function radon(image, theta= "nothing", circle= "nothing")
    if theta == "nothing"
        theta = collect(1:1:180)
    end
    if circle == "nothing"
        diagonal = sqrt(2) * max(size(image,1),size(image,2))
        pad = [Int64(ceil(diagonal - s)) for s in size(image)]
        new_center = [Int64((s + p) / 2) for (s, p) in zip(size(image), pad)]
        old_center = [Int64(s / 2) for s in size(image)]
        pad_before = [nc - oc for (oc, nc) in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for (pb, p) in zip(pad_before, pad)]
        new_tup= (pad[1], pad[2])
        println(-pad_width[1][1]+1,"   ", size(image,1)+pad_width[1][2],"  and  ", -pad_width[2][1]+1,"   ", size(image,2)+pad_width[2][2])
        padded_image = PaddedView(Gray(0),image, (-pad_width[1][1]+1:size(image,1)+pad_width[1][2],-pad_width[2][1]+1:size(image,2)+pad_width[2][2]))
    end
    # padded_image is always square
    padrow = size(image,1)+pad_width[1][2]+pad_width[1][1]
    padcol = size(image,2)+pad_width[2][2]+pad_width[2][1]
    radon_image = zeros(padrow, size(theta,1))
    center = Int64(padrow / 2)

    shift0 = Array([[1, 0, -center],
                       [0, 1, -center],
                       [0, 0, 1]])
    shift1 = Array([[1, 0, center],
                       [0, 1, center],
                       [0, 0, 1]])

    function build_rotation(theta)
        T = theta * (pi/180)
        R = Array([[cos(T), sin(T), 0],
                      [-sin(T), cos(T), 0],
                      [0, 0, 1]])
        println(typeof(dot(shift1,R)))
        return ((shift1'R)'shift0)
    end

    for i in collect(1:1:size(theta,1))
        rotated = warpfast(padded_image, build_rotation(theta[i]))
        cumsum!(radon_image[:, i], rotated)
    end 

    return radon_image
end
