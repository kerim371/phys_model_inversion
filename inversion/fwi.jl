using Statistics, Random, LinearAlgebra, Interpolations, DelimitedFiles, Distributed
using JUDI, SlimOptim, NLopt, HDF5, SegyIO, Plots, ImageFiltering
using SetIntersectionProjection
@everywhere using JUDI.FFTW, Zygote, Flux

include("../utils.jl")

############# INITIAL DATA #############
modeling_type = "bulk"    # slowness, bulk
frq = 0.005 # kHz

prestk_dir = "$(@__DIR__)/../data/trim_segy/"
prestk_file = "shot"
dir_out = "$(@__DIR__)/../data/fwi_lbfgs_$(modeling_type)/$(frq)Hz/"

model_file = "$(@__DIR__)/../data/init_model.h5"
model_file_out = "model"

# use original wavelet file 
wavelet_file = "$(@__DIR__)/../data/source_signal.txt" # dt=1, skip=25, inverse_phase=false
wavelet_skip_start = 0    # 0 [lines] for raw source and 0 for deghosted source
wavelet_dt = 2.5          # 1 [ms] for raw source and 4 [ms] for deghosted source
inverse_phase = false

segy_depth_key_src = "SourceSurfaceElevation"
segy_depth_key_rec = "RecGroupElevation"

seabed = 2000  # [m]

############# INITIAL PARAMS #############
# water velocity, km/s
global vwater = 3.75
global vair = vwater   # 0.33 or must be equal to vwater

# water density, g/cm^3
global rhowater = 1.02
global rhoair = rhowater    # 0.001

# JUDI options
buffer_size = 0f0    # limit model (meters) even if 0 buffer makes reflections from borders that does't hurt much the FWI result

# prepare folder for output data
mkpath(dir_out)

# Load data and create data vector
container = segy_scan(prestk_dir, prestk_file, ["SourceX", "SourceY", "GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container; segy_depth_key = segy_depth_key_rec)

srcx = Float32.(get_header(container, "SourceX")[:,1])
grpx = Float32.(get_header(container, "GroupX")[:,1])
min_src_x = minimum(srcx)./1000f0
max_src_x = maximum(srcx)./1000f0
min_grp_x = minimum(grpx)./1000f0
max_grp_x = maximum(grpx)./1000f0
min_x = minimum([min_src_x, min_grp_x])
max_x = maximum([max_src_x, max_grp_x])

# Load starting model (mlog - slowness built with Vs from logs; mvsp - built from VSP)
fid = h5open(model_file, "r")
n, d, o = read(fid, "n", "d", "o")
m0 = Float32.(read(fid, "m"))
close(fid)

n = Tuple(Int64(i) for i in n)
d = Tuple(Float32(i) for i in d)
o = Tuple(Float32(i) for i in o)

# # ============ MAKE STARTING MODEL DENSER ============
# # dense_factor = 1.5625                # make model n-times denser to achieve better stability
# dense_factor = 1f0

# i_dense = 1:1/Float32(dense_factor):size(m0)[1]
# j_dense = 1:1/Float32(dense_factor):size(m0)[2]

# m0_itp = interpolate(m0, BSpline(Linear()))
# m0 = m0_itp(i_dense, j_dense)
# n = size(m0)
# d = Tuple(Float32(i/dense_factor) for i in d)

# ============ MAKE STARTING MODEL DENSER ============
nb = 20
if modeling_type == "slowness"
    model0 = Model(n, d, o, m0, nb=nb)
elseif modeling_type == "bulk"
    rho0 = rho_from_slowness(m0)
    model0 = Model(n, d, o, m0, rho=rho0, nb=nb)
end

# ============ SMOOTH STARTING MODEL ============
model0.m.data = imfilter(Float32, model0.m.data, Kernel.gaussian((10,10,10)))
# ============ SMOOTH STARTING MODEL ============

@info "modeling_type: $modeling_type"
@info "n: $n"
@info "d: $d"
@info "o: $o"

x = (o[1]:d[1]:o[1]+(n[1]-1)*d[1])./1000f0
y = (o[2]:d[2]:o[2]+(n[2]-1)*d[2])./1000f0
z = (o[3]:d[3]:o[3]+(n[3]-1)*d[3])./1000f0

global air_ind = Int.(round.(abs(o[end])./d[end]))+1
global seabed_ind = air_ind + Int.(round.(seabed./d[end]))+1
if modeling_type == "slowness"
    model0.m[:,:,1:air_ind] .= (1/vair)^2
    model0.m[:,:,air_ind:seabed_ind] .= (1/vwater)^2
elseif modeling_type == "bulk"
    model0.m[:,:,1:air_ind] .= (1/vair)^2
    model0.m[:,:,air_ind:seabed_ind] .= (1/vwater)^2
    model0.rho[:,:,1:air_ind] .= rhoair
    model0.rho[:,:,air_ind:seabed_ind] .= rhowater
end

# Set up wavelet and source vector
src_geometry = Geometry(container; key = "source", segy_depth_key = segy_depth_key_src)

# setup wavelet
wavelet_raw = readdlm(wavelet_file, skipstart=wavelet_skip_start)
itp = LinearInterpolation(0:wavelet_dt:wavelet_dt*(length(wavelet_raw)-1), wavelet_raw[:,1], extrapolation_bc=0f0)
wavelet = Matrix{Float32}(undef,src_geometry.nt[1],1)
wavelet[:,1] = itp(0:src_geometry.dt[1]:src_geometry.t[1])
if inverse_phase
    wavelet[:,1] = wavelet[:,1] * (-1f0)
end

q = judiVector(src_geometry, wavelet)

# Data muters (t0 somehow can't be 0)
Ml_ref = judiDataMute(q.geometry, d_obs.geometry, vp=2000f0, t0=0.50f0, mode=:reflection, taperwidth=8) # keep reflections
Ml_tur = judiDataMute(q.geometry, d_obs.geometry, vp=1500f0, t0=0.30f0, mode=:turning, taperwidth=8)    # keep turning waves

# Bandpass filter
Ml_freq = judiFilter(d_obs.geometry, 0.001, frq*1000f0)
Mr_freq = judiFilter(q.geometry, 0.001, frq*1000f0)

# Bound constraints
vmin = 3.2  # 1.2
vmax = 8.2 # 5.2
vBoundCoef = 0.8
kr = 50

# Slowness squared [s^2/km^2]
mmin = (1f0 ./ vmax).^2
mmax = (1f0 ./ vmin).^2
mminArr = ones(Float32, size(model0)) .* mmin
mmaxArr = ones(Float32, size(model0)) .* mmax
# mminArr = imfilter(model0.m.data * (1f0-vBoundCoef), Kernel.gaussian(kr))
# mmaxArr = imfilter(model0.m.data * (1f0+vBoundCoef), Kernel.gaussian(kr))
# ind = model0.m .< mminArr
# mminArr[ind] .= model0.m[ind]
# ind = model0.m .> mmaxArr
# mmaxArr[ind] .= model0.m[ind]

############# FWI #############
# JUDI options
global jopt = JUDI.Options(
    IC = "fwi",
    limit_m = true,
    buffer_size = buffer_size,
    optimal_checkpointing=false,
    subsampling_factor=50,
    free_surface=false,  # free_surface is ON to model multiples as well
    space_order=16)     # increase space order for > 12 Hz source wavelet

# optimization parameters
niterations = 10
shot_from = 1
shot_to = length(d_obs)
shot_step = 100   # we may want to calculate only each Nth shot to economy time
count = 0
fhistory = Vector{Float32}(undef, 0)
mute_reflections = false
mute_turning = false

@everywhere function H(x)
    n = size(x, 1)
    σ = ifftshift(sign.(-n/2+1:n/2))
    y = imag(ifft(σ.*fft(x, 1), 1))
    return y
end

@everywhere envelope(x, y) = sum(abs2.((x - y) .+ 1im .* H(x - y)))
@everywhere envelope(x::JUDI.judiVector{Float32, Matrix{Float32}}, y::JUDI.judiVector{Float32, Matrix{Float32}}) = sum([abs2.((xd - yd) .+ 1im * H(xd - yd)) for (xd,yd) in zip(x.data,y.data)])
@everywhere denvelope(x, y) = Zygote.gradient(xs->envelope(xs, y), x)[1]
@everywhere denvelope(x::JUDI.judiVector{Float32, Matrix{Float32}}, y::JUDI.judiVector{Float32, Matrix{Float32}}) = [Zygote.gradient(xs->envelope(xs, yd), xd)[1] for (xd,yd) in zip(x.data,y.data)]
@everywhere myloss(x, y) = (envelope(x, y), denvelope(x, y))

@everywhere myloss(randn(Float32, 10, 10), randn(Float32, 10, 10))

F = judiModeling(model0, q.geometry, d_obs.geometry; options=jopt)
# J = judiJacobian(F(model0), q)
J = judiJacobian(F(model0), Mr_freq*q)

# m_update = model0.m
# grad = ones(Float32, size(model0))

# NLopt objective function
function nlopt_obj_fun!(m_update, grad)

    global x, y, z, count, jopt, air_ind, seabed_ind;
    count += 1

    @info "mute_reflections: $mute_reflections"
    @info "mute_turning: $mute_turning"

    # Update model
    model0.m .= Float32.(reshape(m_update, size(model0)))
    if modeling_type == "bulk"
        model0.rho .= Float32.(reshape(rho_from_slowness(model0.m), size(model0)))
        model0.rho[:,:,1:air_ind] .= rhoair
        model0.rho[:,:,air_ind:seabed_ind] .= rhowater
    end

    # Select batch and calculate gradient
    # Subsampling the number of sources should in practice never be used for second order methods such as L-BFGS.
    # get_data(d_obs) is a temporal solution as Ml_freq doesn't work yet with SeisCon
    indsrc = shot_from:shot_step:shot_to
    if mute_reflections
        d0 = F(model0)[indsrc]*(Mr_freq[indsrc]*q[indsrc])
        # r = Ml_tur[indsrc] * (d0 - Ml_freq[indsrc]*d_obs[indsrc])
        r = judiVector(d0.geometry, myloss(Ml_tur[indsrc] * d0 , Ml_tur[indsrc] * Ml_freq[indsrc]*d_obs[indsrc])[2])
        gradient = J'[indsrc] * r
        fval = .5*norm(r)^2
    elseif mute_turning
        d0 = F(model0)[indsrc]*(Mr_freq[indsrc]*q[indsrc])
        r = judiVector(d0.geometry, myloss(Ml_ref[indsrc] * d0 , Ml_ref[indsrc] * Ml_freq[indsrc]*d_obs[indsrc])[2])
        gradient = J'[indsrc] * r
        fval = .5*norm(r)^2
    else
        fval, gradient = fwi_objective(model0, Mr_freq[indsrc]*q[indsrc], Ml_freq[indsrc]*d_obs[indsrc], options=jopt, misfit=myloss)
    end
    gradient = reshape(gradient, size(model0))
    # normalize gradient with depth
    # gradient = gradient .* ((z.-seabed/1000f0)./0.3 .+ 1f0)'    # /3f0
    # gradient = gradient .* (z.^1.0)'
    # grad_amplifier = range(1f0, 5f0, length(z))'
    # gradient = gradient .* grad_amplifier

    gradient[:,:,1:seabed_ind] .= 0f0

    grad[1:end] .= gradient[1:end]

    push!(fhistory, fval)

    println("iteration: ", count, "\tfval: ", fval, "\tnorm: ", norm(gradient))
    save_data(x,y,z,reshape(model0.m.data,size(model0)); 
            pltfile=dir_out * "FWI slowness $count",
            title="FWI slowness^2 with L-BFGS $modeling_type: $(frq*1000)Hz, iter $count",
            colormap=cgrad(:Spectral, rev=true),
            h5file=dir_out * model_file_out * " " * string(count) * ".h5",
            h5openflag="w",
            h5varname="m")
    save_data(x,y,z,sqrt.(1f0 ./ reshape(model0.m.data,size(model0))); 
            pltfile=dir_out * "FWI $count",
            title="FWI velocity with L-BFGS $modeling_type: $(frq*1000)Hz, iter $count",
            colormap=cgrad(:Spectral, rev=true),
            h5file=dir_out * model_file_out * " " * string(count) * ".h5",
            h5openflag="r+",
            h5varname="v")
    save_data(x,y,z,reshape(gradient.data,size(model0)); 
            pltfile=dir_out * "Gradient $count",
            title="FWI gradient with L-BFGS $modeling_type: $(frq*1000)Hz, iter $count",
            clim=(-maximum(gradient.data)/5f0, maximum(gradient.data)/5f0),
            colormap=:bluesreds,
            h5file=dir_out * model_file_out * " " * string(count) * ".h5",
            h5openflag="r+",
            h5varname="grad")
    save_fhistory(fhistory; 
            h5file=dir_out * model_file_out * " " * string(count) * ".h5",
            h5openflag="r+",
            h5varname="fhistory")

    return convert(Float64, fval)
end

println("No.  ", "fval         ", "norm(gradient)")

opt = Opt(:LD_LBFGS, prod(size(model0)))
min_objective!(opt, nlopt_obj_fun!)
lower_bounds!(opt, vec(mminArr)); upper_bounds!(opt, vec(mmaxArr))
maxeval!(opt, niterations)
(minf, minx, ret) = optimize(opt, copy(model0.m.data[:]))
