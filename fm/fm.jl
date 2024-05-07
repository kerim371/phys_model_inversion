using Statistics, Random, LinearAlgebra, Interpolations, DelimitedFiles, DSP
using JUDI, NLopt, HDF5, SegyIO, PyPlot

modeling_type = "bulk"  # bulk or slowness
use_wavelet_file = true
frq = 0.008 # kHz
filter_type = "lowpass"       # lowpass, bandpass, none
frq_low_pass = frq   # kHz
frq1_band_pass = 0.001   # kHz
frq2_band_pass = frq   # kHz
trc_len = 4000  # ms

prestk_dir = "/home/jovyan/work/phys_model_inversion/data/trim_segy/"
prestk_file = "shot"

model_file = "/home/jovyan/work/phys_model_inversion/data/init_model.h5"
wavelet_file = "/home/jovyan/work/phys_model_inversion/data/source_signal.txt"
wavelet_skip_start = 0
wavelet_dt = 2.5  # ms
segy_depth_key_src = "SourceSurfaceElevation"
segy_depth_key_rec = "RecGroupElevation" 

buffer_size = 0f0

# helpful functions
rho_from_slowness(m) = 0.23.*(sqrt.(1f0 ./ m).*1000f0).^0.25

# Load starting model
fid = h5open(model_file, "r")
n, d, o, m = read(fid, "n", "d", "o", "m")
close(fid)

n = Tuple(Int64(i) for i in n)
d = Tuple(Float32(i) for i in d)
o = Tuple(Float32(i) for i in o)
nb = 20
if modeling_type == "slowness"
  model = Model(n, d, o, m, nb=nb)
elseif modeling_type == "bulk"
  rho = rho_from_slowness(m)
  model = Model(n, d, o, m, rho=rho, nb=nb)
end

# Load data and create data vector
container = segy_scan(prestk_dir, prestk_file, ["SourceX", "SourceY", "GroupX", "GroupY", segy_depth_key_rec, segy_depth_key_src, "dt"])
d_obs = judiVector(container; segy_depth_key = segy_depth_key_rec)

# Set up receiver structure
rec_geometry = Geometry(container; key = "receiver", segy_depth_key = segy_depth_key_rec)

# Set up source structure
src_geometry = Geometry(container; key = "source", segy_depth_key = segy_depth_key_src)

# sampling frequency
fs = 1000f0/src_geometry.dt[1]
# Nyquist frequency
f_nyq = fs/2f0

# setup wavelet
if use_wavelet_file
  wavelet_raw = readdlm(wavelet_file, skipstart=wavelet_skip_start)
  itp = LinearInterpolation(0:wavelet_dt:wavelet_dt*(length(wavelet_raw)-1), wavelet_raw[:,1], extrapolation_bc=0f0)
  wavelet = Matrix{Float32}(undef,src_geometry.nt[1],1)
  if filter_type == "lowpass"
    responsetype = Lowpass(frq_low_pass*1000f0; fs=fs)
    designmethod = Butterworth(8)
    wavelet[:,1] = filt(digitalfilter(responsetype, designmethod), itp(0:src_geometry.dt[1]:src_geometry.t[1]))
  elseif filter_type == "bandpass"
    responsetype = Bandpass(frq1_band_pass*1000f0, frq2_band_pass*1000f0; fs=fs)
    designmethod = Butterworth(8)
    wavelet[:,1] = filt(digitalfilter(responsetype, designmethod), itp(0:src_geometry.dt[1]:src_geometry.t[1]))
  elseif filter_type == "none"
    wavelet[:,1] = itp(0:src_geometry.dt[1]:src_geometry.t[1])
  end
else
  f0 = frq     # kHz
  wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], f0)
end
wavelet_spec = abs.(fft(wavelet[:,1]))[1:Int(round(length(wavelet)/2))]
q = judiVector(src_geometry, wavelet)

fig, (ax1, ax2) = PyPlot.subplots(1, 2, sharex=false, sharey=false)
ax1.plot(range(0, stop=src_geometry.t[1]/1000f0, length=length(wavelet[:,1])), wavelet[:,1])
ax2.plot(range(0, stop=f_nyq, length=length(wavelet_spec)), wavelet_spec)
ax1.set_title("Wavelet")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude")
ax1.set_xlim(0,0.5)
ax1.grid(visible=true)
ax2.set_title("Spectrum")
ax2.set_xlabel("Frequency [Hz]")
# ax2.set_xlim(0,20)
ax2.grid(visible=true)
display(fig)

# Setup options
opt = Options(space_order=16, limit_m=true, buffer_size=buffer_size, free_surface=false)

src_num = 100

# Setup operators
Pr = judiProjection(rec_geometry[src_num])
F = judiModeling(model; options=opt)
Ps = judiProjection(src_geometry[src_num])

# Model and image data

# We first model synthetic data using our defined source and true model 
# Nonlinear modeling
dobs = Pr*F*adjoint(Ps)*q[src_num]

xrec = grpx = Float32.(get_header(container[src_num], "GroupX")[:,1])./1000f0
yrec = grpy = Float32.(get_header(container[src_num], "GroupY")[:,1])./1000f0

dt = rec_geometry[src_num].dt[1]
nt = rec_geometry[src_num].nt[1]
timeD = (nt-1)*dt/1000f0

data_syn = dobs.data[1]
data_field = Float32.(container[src_num].data)
if filter_type != "none"
    for i in 1:size(data_field)[2]
        data_field[:,i] = filt(digitalfilter(responsetype, designmethod), data_field[:,i])
    end
end

data_syn_rms = sqrt(sum(data_syn.^2))
data_field_rms = sqrt(sum(data_field.^2))

coef_amp = data_field_rms/data_syn_rms

data_syn *= coef_amp

coef_color_syn = 0.1
coef_color_field = 0.5

vmin_syn = minimum(data_syn)*coef_color_syn
vmax_syn = abs(minimum(data_syn))*coef_color_syn
vmin_field = minimum(data_field)*coef_color_field
vmax_field = abs(minimum(data_field))*coef_color_field

#' Plot the shot record
fig, (ax1, ax2) = PyPlot.subplots(1, 2, sharex=false, sharey=true)
ax1.imshow(data_syn[:,1:end], 
            vmin=vmin_syn, 
            vmax=vmax_syn, 
            cmap="PuOr", 
            extent=[xrec[1], xrec[end], timeD, 0], 
            aspect="auto")
ax2.imshow(data_field[:,1:end], 
            vmin=vmin_field, 
            vmax=vmax_field, 
            cmap="PuOr", 
            extent=[xrec[1], xrec[end], timeD, 0], 
            aspect="auto")
ax1.set_title("Synthetic, shot: $src_num, frq: $(frq*1000)Hz")
ax1.set_xlabel("Receiver position (km)")
ax1.set_ylabel("Time (s)")
# ax1.set_ylim(3,1)
ax2.set_title("Field, shot: $src_num, frq: $(frq*1000)Hz")
ax2.set_xlabel("Receiver position (km)")
# ax2.set_ylim(3,1)
display(fig)