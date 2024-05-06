using HDF5, Plots, DSP, DSP.Windows, Interpolations

function save_data(x,y,z,data; pltfile,title,colormap,clim=nothing,h5file,h5openflag,h5varname)
    @info "save_data: $h5file"
    n = (length(x),length(y),length(z))
    o = (x[1],y[1],z[1])
    d = (x[2]-x[1],y[2]-y[1],z[2]-z[1])
    isnothing(clim) && (clim = (minimum(data),maximum(data)))

    xy_slice = 250
    xz_slice = 800
    yz_slice = 800

    plt = Plots.heatmap(x, y, data[:,:,xy_slice], c=colormap, 
        xlims=(x[1],x[end]), 
        ylims=(y[1],y[end]), yflip=true,
        title=title * " $(xy_slice) slice",
        clim=clim,
        xlabel="Lateral position X [km]",
        ylabel="Lateral position Y [km]",
        dpi=600)
    Plots.savefig(plt, pltfile * "_xy.png")

    plt = Plots.heatmap(x, z, data[:,xz_slice,:], c=colormap, 
        xlims=(x[1],x[end]), 
        ylims=(z[1],z[end]), yflip=true,
        title=title * " $(xz_slice) slice",
        clim=clim,
        xlabel="Lateral position X [km]",
        ylabel="Depth [km]",
        dpi=600)
    Plots.savefig(plt, pltfile * "_xz.png")

    plt = Plots.heatmap(y, z, data[yz_slice,:,:], c=colormap, 
        xlims=(y[1],y[end]), 
        ylims=(z[1],z[end]), yflip=true,
        title=title * " $(yz_slice) slice",
        clim=clim,
        xlabel="Lateral position Y [km]",
        ylabel="Depth [km]",
        dpi=600)
    Plots.savefig(plt, pltfile * "_xz.png")

  fid = h5open(h5file, h5openflag)
  (haskey(fid, h5varname)) && (delete_object(fid, h5varname))
  (haskey(fid, "o")) && (delete_object(fid, "o"))
  (haskey(fid, "n")) && (delete_object(fid, "n"))
  (haskey(fid, "d")) && (delete_object(fid, "d"))
  write(fid, 
      h5varname, Matrix(adjoint(data)), # convert adjoint(Matrix) type to Matrix
      "o", collect(o.*1000f0), 
      "n", collect(n), 
      "d", collect(d.*1000f0))
  close(fid)
end

function save_data_as_segy(x,z,data)

end

function save_fhistory(fhistory; h5file,h5openflag,h5varname)
  @info "save_fhistory: $h5file"
  fid = h5open(h5file, h5openflag)
  (haskey(fid, h5varname)) && (delete_object(fid, h5varname))
  write(fid, h5varname, fhistory)
  close(fid)
end

"""
Recalculate slowness^2 to density using Gardner formulae
"""
rho_from_slowness(m) = 0.23.*(sqrt.(1f0 ./ m).*1000f0).^0.25

function get_seabed_ind(model_file, var)
  fid = h5open(model_file, "r")
  data = Float32.(read(fid, var))
  close(fid)

  seabed_ind = zeros(Int, size(data)[1])
  for i = 1:size(data)[1]
    for j = 1:size(data)[2]-1
      if data[i,j] == data[i,1]
        seabed_ind[i] += 1
      else
        break
      end
    end
  end
  return seabed_ind
end

# """
#   klauder
#     f1::Real
#     f2::Real
#     dt::Real
#     t::Real
#     t_shift::Real

#   Function to generate Klauder wavelet.
#     `f1`: start frequency (Hz)
#     `f2`: stop frequency (Hz)
#     `dt`: sampling interval (second)
#     `t`: duration of the signal (second)
#     `t_shift`: time shift (second)

#   Return:
#     `ft`: vector of amplitudes
# """
# function klauder(;f1,f2,dt,t,t_shift)
#   k = (f2-f1)./t; # rate of change of frequenc y with time
#   f0 = (f2+f1)./2.0; # midfrequency of bandwidth
#   tt = 0:dt:t; # vector for the total recording time
#   amp = 1.0;
  
#   # Klauder wavelet equation 
#   ft = real((sin.(pi*k*(tt.-t_shift).*(t.-(tt.-t_shift)))./(pi*k*(tt.-t_shift))).*exp.(2.0*pi*1im*f0.*(tt.-t_shift)));
#   ft = ft./maximum(ft);
#   ft = amp.*ft;

#   return ft
# end

function ormsby(; dt=0.002, t=1.0, f1=2.0, f2=10.0, f3=40.0, f4=60.0)

  fc = (f2+f3)/2.0
  nw = 2.2/(fc*dt)
  nc = floor(Int, nw/2)
  tt = dt*collect(-nc:1:nc)
  nw = 2*nc + 1
  a4 = (pi*f4)^2/(pi*(f4-f3))
  a3 = (pi*f3)^2/(pi*(f4-f3))
  a2 = (pi*f2)^2/(pi*(f2-f1))
  a1 = (pi*f1)^2/(pi*(f2-f1))

  u = a4*(sinc.(f4*tt)).^2 - a3*(sinc.(f3*tt)).^2
  v = a2*(sinc.(f2*tt)).^2 - a1*(sinc.(f1*tt)).^2

  w = u - v
  w = w.*hamming(nw)/maximum(w)

  ttexp = 0:dt:t
  nt = length(ttexp)
  resize!(w, nt)  # doesn't initialize new vals with zero
  if nw < nt
      w[nw+1:end] .= 0.0
  end
  return w
end


function sweep(fmin=10.0,fmax=100.0,dt=1e-3,tmax=10.0,staper=0.2,etaper=staper,ltaper=0.0,
                        amplitude=1.0,theta0=-pi/2)
# SWEEP: generate a linear Vibroseis sweep()
#
# [s,t]= sweep(fmin,fmax,dt,tmax,staper,etaper,ltaper,amplitude,theta0)
# [s,t]= sweep(); #use all defaults
# [s,t]= sweep([],[],2e-3)#use defaults, but override sample rate
# NOTE that ltaper=2.5 better matches the high frequency end of a sweep() 
#      with the same parameters generated by a Pelton Vibe controller
#
# SWEEP generates a linear synthetic Vibroseis sweep for the 
# specified passband. Reference: Aldrige, D.F., 1992, Mathematics of linear sweeps, vol 28[1]
# 62-68; http://csegjournal.com/assets/pdfs/archives/1992_06/1992_06_math_linear_sweeps.pdf
#
# fmin= minimum swept frequency in Hz
#       default = 10.
# fmax= maximum swept frequency in Hz
#       default = 100.
# dt= time sample rate in seconds
#       default = 1e-3 [1 ms]
# tmax= sweep length in seconds
#       default = 10.
# staper= length of start taper [cos] in seconds
#       default = 0.2
# etaper= length of end taper [cos] in seconds
#       default = staper
# ltaper= percentage reduction in amplitude over length of sweep()
#       default = 0.
# amplitude= maximum amplitude of the sweep()
#       default = 1.
# theta0= start phase of the sweep()
#       default = -pi/2.
#
# s= output linear sweep()
# t= output time coordinate vector 
# check inputs
#function [s,t] = ksweep(fmin,fmax,dt,tmax,staper,etaper,ltaper,amplitude,theta0)
  
  if isnothing(staper)
    staper=0.2;
  end

  ## create time vector
  t = (0:tmax/dt)*dt
  
  ## create amplitude function
  a = amplitude*ones(size(t))
  
  #start taper
  if staper>0
      a[t .<= staper] = (amplitude/2) * (1 .- cos.(pi*t[t .<= staper]/staper))
  end
  #end taper
  if etaper>0
      a[t .>= tmax-etaper] = (amplitude/2) * (1 .+ cos.(pi*(t[t .>= tmax-etaper] .- tmax .+ etaper)/etaper))
  end
  
  #linear taper
  a = a.*range(1,1-ltaper/100,length=length(t))
  
  ## create sweep()
  s = a.*cos.(theta0 .+ 2*pi*fmin*t .+ pi*((fmax-fmin)/tmax)*t.^2)
  return s,t
end


function auto(v,n=length(v),flag=1.0)
# AUTO: single-sided autocorrelation
#
# a=auto(v,n,flag)
# a=auto(v,n)
# a=auto(v)
#
# auto computes n lags of the one sided autocorrelation of 
# the vector "v'. The first lag, a[1], is termed the 'zeroth lag"
# 
# v= input vector
# n= number of lags desired [can be no larger than length(v)].
#    ********* default =length(v) *********
# flag= 1.0 ... normalize the "zero lag" (first returned value)
#               to 1.0.
#        anything else ... don't normalize
#       ******* default =1.0 ******
# a= one sided autocorrelation returned as a row vvector. a[1] is zero lag.
#
# NOTE: A two sided autocorrelation | a cross correlation can be
#       computed with XCORR in the signal toolbox.
  
# 
# master loop
# 
  
  nrow = size(v,1)
  ncol = size(v,2)
  done=0
  a=zeros(1,n)
  # for row vectors
  if nrow==1 
    u=v'
    for k=1:n
      a[k]=v*u
      v=[0.0; v[1:length(v)-1]]'
    end
    done=1
  end
  # for column vectors
  if ncol==1
    u=v'
    for k=1:n
      a[k]=u*v
      u=[0.0; u[1:length(u)-1]]'
    end
    done=1
  end
  if done==0
    error(" input not a vector")
  end
  # normalize
  if flag==1.0
    a=a/maximum(a)
  end
  return vec(a)
end


function han(n)
  xint=2*pi/(n+1)
  x=xint*(1:n).-pi
  
  w=.5*(1 .+ cos.(x))'
  return w
end


function mwindow(n,percent=10.0)
  # MWINDOW: creates an mwindow (boxcar with raised-cosine tapers)
  #
  # w = mwindow(n,percent)
  # w = mwindow(n)
  # 
  # MWINDOW returns the N-point Margrave window in a 
  # column vector. This window is a boxcar over the central samples
  # round((100-2*percent)*n/100) in number, while it has a raised cosine
  # (hanning style) taper on each end. If n is a vector, it is()
  # the same as mwindow(length(n))
  #
  # n= input length of the mwindow. If a vector, length(n) is()
  #    used
  # percent= percent taper on the ends of the window
  #   ************* default=10 ************
    
  if length(n)>1
    n=length(n)
  end
  # compute the Hanning function 
  if percent>50 || percent<0
    error(" invalid percent for mwindow")
  end
  m=2.0*percent*n/100.
  m=Int(2*floor(m/2))
  h=han(m)
  w = [h[1:Int(m/2)]; ones(n-m); h[Int(m/2):-1:1]]
  return w
end


function mwhalf(n,percent=10.0)
  # MWHALF: half an mwindow [boxcar with raised-cosine taper on one end]
  #
  # mwhalf(n,percent)
  # mwhalf(n)
  # 
  # MWHALF returns the N-point half-Margrave window in a 
  # row vector. This window is a boxcar over the first samples
  # (100-percent)*n/100 in number, while it has a raised cosine
  # (hanning style) taper on the end. If n is a vector, it is()
  # the same as mwindow[length(n)
  #
  # n= input length of the mwindow. If a vector, length(n) is()
  #    used
  # percent= percent taper on the ends of the window
  #   ************* default=10 ************
    
   if length(n)>1
     n=length(n)
   end
  # compute the Hanning function 
   if percent>100 || percent<0
     error(" invalid percent for mwhalf")
   end
   m=Int(floor(percent*n/100))
   h=han(2*m)
   h=h[:]
   w = [ones(n-m); h[m:-1:1]]
   return w
end


function convm(r,w,pct=10.0)
  # CONVM: convolution followed by truncation for min phase filters
  #
  # s= convm(r,w,pct)
  #
  # CONVM is a modification of the "conv" routine from the MATLAB
  # toolbox. The changes make it more convenient for seismic purposes
  # in that the output vector; s; has a length equal to the first
  # input vector;  r. Thus; 'r' might correspond to a reflectivity
  # estimate to be convolved with a wavelet contained in 'w' to
  # produce a synthetic seismic response 's'. It is assumed that
  # the wavelet in w is causal & that the first sample occurs at time zero.
  # For non-causal wavelets; use "convz". An warning will occur if
  # w is longer than r. If the first argument is a matrix; then convm outputs
  # a matrix of the same size where the second argument has been convolved
  # with each column. By default convm does a raised cosine taper at the end
  # of the trace to reduce truncation artefacts.
  #
  # r ... reflectivity
  # w ... wavelet
  # pct ... percent taper at the end of the trace to reduce truncation
  #       effects. See mwhalf.
  #  ********** default = 10 ********
  
  # 
  #convert to column vectors
  a=size(r,1)
  b=size(r,2)
  if a==1; r=transpose(r); end
  w=w[:]
  nsamps=size(r,1)
  ntr=size(r,2)
  # if(length(w)>nsamps) 
  #     warning("second argument longer than the first, output truncated to length of first argument."); 
  # end
  s=zeros(size(r))
  if pct>0
      mw=mwhalf(nsamps,pct)
  else
      mw=ones(nsamps,1)
  end
  
  for k=1:ntr
      temp=DSP.conv(r[:,k],w)
      s[:,k]=temp[1:nsamps].*mw[:]
  end
  
  if a==1
    s=transpose(s)
  end
  return s
end


function xcoord(xstart,delx,nx)
  # XCOORD: create a coordinate vector given start; increment; & number
  #
  # x=xcoord(xstart,delx,nx)
  # |
  # x=xcoord(xstart,delx,v)
  #
  # XCOORD computes a coordinate vector of length nx; starting
  #   at x=xstart; & incrementing by delx. If the third argument
  #   is a vector, v, it is the same as xcoord(xstart,delx,length(v)).
   
  if length(nx)>1
     nx=length(nx)
  end
  xmax=xstart+(nx-1)*delx
  x=xstart:delx:xmax
  return vec(x)
end


function wavevib(fmin,fmax,dt,slength,wlength,taper=nothing)
  # WAVEVIB: creates a vibroseis [Klauder] wavelet
  #
  # [w,tw]=wavevib(fmin,fmax,dt,slength,wlength,taper)
  #
  # WAVEVIB generates a vibroseis waveform [Clauder wavelet] by
  # first calling SWEEP to generate a linear sweep & then calling
  # AUTO to autocorrelate that sweep. Theoretically; the autocorrelation
  # length is ~ 2*slength but only only the central "wlength" long
  # part is generated. This is like windowing the auto with a boxcar.
  #
  #
  # fmin= minimum swept frequency in Hz
  # fmax= maximum swept frequency in Hz
  # dt= time sample rate in seconds
  # slength= sweep length in seconds
  # wlength= desired wavelet length()
  #  taper= length of cosine sweep taper in seconds
  #  ********** default =.5 if tmax>4.0
  #             otherwise =.25 *************
  #
  # w= output wavelet
  # tw= time coordinate vector for w
  
  wlength=.5*wlength
  s,t=sweep(fmin,fmax,dt,slength,taper)
  halfw=auto(s,Int(round(wlength/dt,RoundToZero)+1))
  w=[halfw[length(halfw):-1:2]; halfw[:]]
  tw=xcoord(-(length(halfw)-1)*dt,dt,w)

  #taper
  w=w[:].*mwindow(length(w))

  # normalize
  # generate a refenence sinusoid at the dominant frequency
  refwave=sin.(2*pi*(fmin+fmax)*.5*t)
  reftest=convm(refwave,w)
  fact=maximum(refwave)/maximum(reftest)
  w=w*fact
  return w,tw
end


"""
    SeisKolmogoroff(in)

Transform a wavelet into its minimum phase equivalent.

# Arguments
- `in::Array{Real,1}`: input wavelet.

# Example
```julia
julia> using PyPlot
julia> w = Ricker()
julia> wmin = SeisKolmogoroff(w)
julia> plot(w); plot(wmin)
```

# Reference
* Claerbout, Jon F., 1976, Fundamentals of geophysical data processing.
McGraw-Hill Inc.
"""
function SeisKolmogoroff(w::Array{T,1}) where T<:Real

    nw = length(w)
    nf = 8*nextpow(2,nw)
    W = fft(cat(w,zeros(nf-nw),dims=1))
    A = log.(abs.(W) .+ 1.e-8)
    a = 2*ifft(A)
    n2 = floor(Int, nf/2)
    a[n2+2:nf] .= 0.0
    a[1] = a[1]/2
    A = exp.(fft(a))
    a = real(ifft(A))
    wmin = real(a[1:nw])

end


function klauder(;fmin,fmax,dt,slength,wlength,taper=nothing, tshift=nothing, minphase=false)
# KLAUDER ... an alias for WAVEVIB
# 
# [w,tw]=klauder(fmin,fmax,dt,slength,wlength,taper)
#
# KLAUDER generates a vibroseis waveform [Klauder wavelet] by
# first calling SWEEP to generate a linear sweep & then calling
# AUTO to autocorrelate that sweep. Theoretically; the autocorrelation
# length is ~ 2*slength but only only the central "wlength" long
# part is generated. This is like windowing the auto with a boxcar.
# This function is a wrapper around wavevib.m .
#
#
# fmin= minimum swept frequency in Hz
# fmax= maximum swept frequency in Hz
# dt= time sample rate in seconds
# slength= sweep length in seconds
# wlength= desired wavelet length()
#  taper= length of cosine sweep taper in seconds
#  ********** default =.5 if tmax>4.0
#             otherwise =.25 *************
#
# w= output wavelet
# tw= time coordinate vector for w
  
  
  if minphase
    w,tw=wavevib(fmin,fmax,dt,slength,wlength,taper)
    return SeisKolmogoroff(w),tw
  elseif isnothing(tshift)
    w,tw=wavevib(fmin,fmax,dt,slength,wlength,taper)
    return w,tw
  end

  w,tw=wavevib(fmin,fmax,dt,slength,2*wlength,taper)
  itp = LinearInterpolation(tw, w, extrapolation_bc=0f0)
  tw = (0:dt:wlength) .+ tshift
  w = itp(tw)
  w = w/maximum(w) # normalize
  return w,tw
end
