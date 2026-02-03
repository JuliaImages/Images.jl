
function radon(B, angles)
    h, w = size(B)                # height and width of the image
    A = convert(Array{Gray}, B)   # converting the image to gray scale if is RGB
    
    d = floor(Int, sqrt(w*w+h*h)/2)  # half-length of the diagonal
    Nr = 2d+1                        # length of the diagonal rounded to nearest integers
    R = linspace(-Nr/2, Nr/2, Nr)    # equally spaced intervals
    
    sinogram = zeros(Nr, length(angles)) # creating array of expected size of radon image

    for j = 1:length(angles)          # for all the values of theta
        a = angles[j]                 
        θ = a + pi/2              
        sinθ, cosθ = sin(θ), cos(θ)
        for k = 1:length(R)           # for each possible pixel distance
            rk = R[k]
            x0, y0 = rk*cosθ + w/2, rk*sinθ + h/2
            # for a line perpendicular to this displacement from the center,
            # determine the length in either direction within the boundary of the image
            lfirst, llast = interior(x0, y0, sinθ, cosθ, w, h, R) # function to calculate 
            tmp = 0.0                 # initialize tmp = 0
            @inbounds for l in lfirst:llast  # for all the legal values 
                rl = R[l]
                x, y = x0 - rl*sinθ, y0 + rl*cosθ  # use nearest neighbour approximation
                ix, iy = trunc(Int, x), trunc(Int, y) # calculate lower index
                fx, fy = x-ix, y-iy                   # calculate the weight factor
                tmp += (1-fx)*((1-fy)*A[ix,iy] + fy*A[ix,iy+1]) + # similar to calculating g'(round(alpha*n + beta), n)
                       fx*((1-fy)*A[ix+1,iy] + fy*A[ix+1,iy+1])  # if we g(x,y) = image, g'(x,y) = new weighted g(x,y)
            end # increment tmp
            sinogram[k,j] = tmp    # update the matrix elements
        end
    end
    
    return sinogram;
end
# function to calculate range of signed distances from center (r) that gives pixels lying within the image
function interior(x0, y0, sinθ, cosθ, w, h, R::Range)
    rx1, rxw = (x0-1)/sinθ, (x0-w)/sinθ  # to calculate the extreme values of r w.r.t. x coordinate
    ry1, ryh = (1-y0)/cosθ, (h-y0)/cosθ  # to calculate the extreme values of r w.r.t. y coordinate
    rxlo, rxhi = minmax(rx1, rxw)    # assigning minimum and maximum r w.r.t. x coordinate    
    rylo, ryhi = minmax(ry1, ryh)    # assigning minimum and maximum r w.r.t. y coordinate
    rfirst = max(minimum(R), ceil(Int,  max(rxlo, rylo))) # over all max r
    rlast  = min(maximum(R), floor(Int, min(rxhi, ryhi))) # over all min r
    Rstart, Rstep = first(R), step(R)
    lfirst, llast = ceil(Int, (rfirst-Rstart)/Rstep), floor(Int, (rlast-Rstart)/Rstep) # actual range of indices over R
    if lfirst == 0 # if r is corresponding to diagonal end
        lfirst = length(R)+1
    end
    # Because of roundoff error, we still can't trust that the forward
    # calculation will be correct. Make adjustments as needed.
    # to ensure lfirst > llast
    while lfirst <= llast   
        rl = R[lfirst]
        x, y = x0 - rl*sinθ, y0 + rl*cosθ
        if (1 <= x < w) && (1 <= y < h)
            break
        end
        lfirst += 1
    end
    while lfirst <= llast
        rl = R[llast]
        x, y = x0 - rl*sinθ, y0 + rl*cosθ
        if (1 <= x < w) && (1 <= y < h)
            break
        end
        llast -= 1
    end
    lfirst, llast
end

