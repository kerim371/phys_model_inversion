using SegyIO, Plots

prestk_dir = "/home/data-volume/phys_model/"
prestk_file = "009_3D_georeshetka_float_fixed_scale.sgy"
dt = 2.5                # ms
ns = 1081               # limit number of samples
dir_out = "$(@__DIR__)/../data/trim_segy/"

container = segy_scan(prestk_dir, prestk_file, ["SourceX", "SourceY", "GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])

# prepare folder for output data
mkpath(dir_out)

I = length(container)
progress = 0
for i in 1:I
  block = container[i]
  block_out = SeisBlock(Float32.(block.data[1:ns,:]))
  # without copying (copy is not supported for fileheader)
  block_out.fileheader = block.fileheader
  block_out.fileheader.bfh.DataSampleFormat = 5
  block_out.traceheaders = block.traceheaders
  block_out.fileheader.bfh.dt = Int(round(dt*1000f0))
  block_out.fileheader.bfh.ns = ns
  set_header!(block_out, "dt", Int(round(dt*1000f0)))
  set_header!(block_out, "ns", ns)
  # make elevation scalar multiplier to be 1 instead of 0 (0 -> sets all elevations to NAN)
  set_header!(block_out, "ElevationScalar", Int16(1))
  segy_write(dir_out * "shot_$i.sgy", block_out)
  if round(i/I*100f0) > progress
    global progress = round(i/I*100f0)
    @info "progress: ($progress)%"
  end
end
