library(ConsensusClusterPlus)
covmat <- as.matrix(read.csv('smaps_cov.csv', header = FALSE))
ConsensusClusterPlus(covmat, maxK=7, clusterAlg = "pam", reps = 20, plot = "png",
                     title ="DLPFC_output", writeTable = TRUE)
