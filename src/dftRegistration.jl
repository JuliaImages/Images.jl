### DFT registration
## Algorithm from (and code very inspired by the matlab code provided by the authors):
## Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
##  "Efficient subpixel image registration algorithms," Opt. Lett. 33, 156-158 (2008).

### Apply dft registration to an image/array (or each frame of such an image if the input is 3D) compared to a reference image (by default the first image of the image stack inputed). If align==false, returns an array with one row per frame in the image, and columns corresponding to the error, the phase difference, the estimated row shift and the estimated column shift. Otherwise returns the registered image.
function dftReg(imgser;ref::AbstractArray=imgser[:,:,1],ufac::Int64=10,align::Bool=false)
    ref = fft(ref)
    imgF = fft(imgser,(1,2))
    if ndims(imgF) == 2
        results = dftRegfft(data(ref),imgF,ufac).'
    else
        results = zeros(size(imgF)[3],4)
        for i=1:size(imgF)[3]
           results[i,:] = dftRegfft(data(ref),imgF[:,:,i],ufac)
        end
    end
    if align
        return(alignFromDft(imgser,results))
    else
        return(results)
    end
end
    

### Align an image using the results of the dftReg function.
function alignFromDft(img2reg::AbstractArray,dftRegRes)
    imRes = similar(img2reg,Float64)
    img2regF = fft(img2reg,(1,2))
    if ndims(img2reg)==2
        return(subpixelshift(img2regF,dftRegRes[3],dftRegRes[4],dftRegRes[2]))
    end
    for i=1:size(dftRegRes)[1]
        imRes[:,:,i] = subpixelshift(img2regF[:,:,i],dftRegRes[i,3],dftRegRes[i,4],dftRegRes[i,2])
    end
    imRes
end


### The DFT algorithm : computes misalignment error, phase difference and row and column shifts between two images. The input is the fft transform of the two images, and the oversampling factor usfac.
function dftRegfft(reffft,imgfft,usfac)
    if usfac==0
        ## Compute error for no pixel shift
        CCmax = sum(reffft.*conj(imgfft))
        rfzero = sumabs2(reffft)
        rgzero = sumabs2(imgfft)
        error = sqrt(abs(1 - CCmax*conj(CCmax)/(rgzero*rfzero)))
        diffphase = atan2(imag(CCmax),real(CCmax))
        output = ["error" => error, "diffphase" => diffphase]
    elseif usfac==1
        ## Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the peak
        L = length(reffft)
        CC = fftshift(reffft).*conj(fftshift(imgfft))
        CC = ifft(ifftshift(CC))
        loc = indmax(abs(CC))
        CCmax=CC[loc]
        rfzero = sumabs2(reffft)/L
        rgzero = sumabs2(imgfft)/L
        error = sqrt(abs(1 - CCmax*conj(CCmax)/(rgzero*rfzero)))
        diffphase = atan2(imag(CCmax),real(CCmax))

        (m,n) = size(reffft)
        md2 = div(m,2)
        nd2 = div(n,2)
        locI = ind2sub((m,n),loc)
        
        if locI[1]>md2
            rowShift = locI[1]-m-1
        else rowShift = locI[1]-1
        end
        if locI[2]>nd2
            colShift = locI[2]-n-1
            else colShift = locI[2]-1
        end
        output = [error,diffphase,rowShift,colShift]
    else
        ## Partial pixel shift
        
        ##First upsample by a factor of 2 to obtain initial estimate
        ##Embed Fourier data in a 2x larger array
        dimR = size(reffft)
        dimRL = map((x) -> x*2, dimR)
        CC = zeros(Complex,dimRL)
        CC[(dimR[1]+1-div(dimR[1],2)):(dimR[1]+1+div((dimR[1]-1),2)),(dimR[2]+1-div(dimR[2],2)):(dimR[2]+1+div((dimR[2]-1),2))] = fftshift(reffft).*conj(fftshift(imgfft))

        ##  Compute crosscorrelation and locate the peak 
        CC = ifft(ifftshift(CC))
        loc = indmax(abs(CC))
        (m,n) = size(CC)
        locI = ind2sub((m,n),loc)
        CCmax = CC[loc]

        ## Obtain shift in original pixel grid from the position of the crosscorrelation peak
        md2 = div(m,2)
        nd2 = div(n,2)
          
        if locI[1] > md2
            rowShift = locI[1]-m-1
        else rowShift = locI[1]-1
        end
        
        if locI[2] > nd2
            colShift = locI[2]-n-1
        else colShift = locI[2]-1
        end
        rowShift = rowShift/2
        colShift = colShift/2

        ## If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2
            ### DFT Computation ###
            # Initial shift estimate in upsampled grid
            rowShift = iround(rowShift*usfac)/usfac
            colShift = iround(colShift*usfac)/usfac
            dftShift = div(ceil(usfac*1.5),2) ## center of output array at dftshift+1
            ## Matrix multiply DFT around the current shift estimate
            CC = conj(dftups(imgfft.*conj(reffft),ceil(Int64,usfac*1.5),ceil(Int64,usfac*1.5),usfac,dftShift-rowShift*usfac,dftShift-colShift*usfac))/(md2*nd2*usfac^2)
            ## Locate maximum and map back to original pixel grid
            loc = indmax(abs(CC))
            locI = ind2sub(size(CC),loc)
            CCmax = CC[loc]
            rgzero = dftups(reffft.*conj(reffft),1,1,usfac)[1]/(md2*nd2*usfac^2)
            rfzero = dftups(imgfft.*conj(imgfft),1,1,usfac)[1]/(md2*nd2*usfac^2)
            locI = map((x) -> x - dftShift - 1,locI)
            rowShift = rowShift + locI[1]/usfac
            colShift = colShift + locI[2]/usfac
        else  
            rgzero = sum(reffft.*conj(reffft))/m/n
            rfzero = sum(imgfft.*conj(imgfft))/m/n  
        end
        error = sqrt(abs(1 - CCmax*conj(CCmax)/(rgzero*rfzero)))
        diffphase = atan2(imag(CCmax),real(CCmax))
        ## If its only one row or column the shift along that dimension has no effect. We set to zero.
        if md2 == 1
            rowShift = 0
        end
        if  nd2==1
            colShift = 0
        end
        output = [error,diffphase,rowShift,colShift]
    end
    output
end

### Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just a small region
function dftups(inp,nor,noc,usfac=1,roff=0,coff=0)
    (nr,nc) = size(inp)  
    kernc = exp((-1im*2*pi/(nc*usfac))*((ifftshift(0:(nc-1)))-floor(nc/2))*((0:(noc-1))-coff).')
    kernr = exp((-1im*2*pi/(nr*usfac))*((0:(nor-1))-roff)*(ifftshift(0:(nr-1))-floor(nr/2)).')
    kernr*data(inp)*kernc
end

### Translate a 2D image/array at subpixel resolution. Outputs the original images translated. If the input is of complex type, it is assumed to be the Fourier transform of the image to shift.
function subpixelshift(img::AbstractArray,rowShift,colShift,diffphase)
    img = fft(img)
    (nr,nc) = size(img)
    Nr = ifftshift((-div(nr,2)):(ceil(Int64,nr/2)-1))
    Nc = ifftshift((-div(nc,2)):(ceil(Int64,nc/2)-1))
    Greg = data(img).* exp(2im*pi*((-rowShift*Nr/nr).-(colShift*Nc/nc).'))
    Greg = Greg * exp(1im*diffphase)
    copyproperties(img,real(ifft(Greg)))
end         

function subpixelshift(img::AbstractArray{Complex{Float64},2},rowShift,colShift,diffphase)
    (nr,nc) = size(img)
    Nr = ifftshift((-div(nr,2)):(ceil(Int64,nr/2)-1))
    Nc = ifftshift((-div(nc,2)):(ceil(Int64,nc/2)-1))
    Greg = data(img) .* exp(2im*pi*((-rowShift*Nr/nr).-(colShift*Nc/nc).'))
    Greg = Greg * exp(1im*diffphase)
    copyproperties(img,real(ifft(Greg)))
end
           





