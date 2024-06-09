
library(scImpute)
args <- commandArgs()
baseName <- args[6]

count_path <- "intermediateFiles/raw.txt"


raw_count = read.table(count_path, header = TRUE, row.names = 1)
raw_count = as.matrix(raw_count)

outfile_format <- "txt"
out_dir <- baseName
labeled <- FALSE
drop_thre <- 0.5
Kcluster <- 4
ncores <- 1

imputed_data <- scImpute::scimpute(
  count_path = count_path,
  infile = outfile_format,
  outfile = outfile_format,
  out_dir = out_dir,
  labeled = labeled,
  drop_thre = drop_thre,
  Kcluster = Kcluster,
  ncores = ncores,
)




