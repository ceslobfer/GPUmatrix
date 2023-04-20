
installTorch <- tryCatch({
  res <- FALSE
  installedTORCH <- requireNamespace("torch", quietly = T)
  if (installedTORCH){
    res <- torch::torch_is_installed()
    if(res) attachNamespace("torch")
  }
  res
})

.onAttach <- function (libname, pkgname) {

  installedTENSORFLOW <- requireNamespace("tensorflow", quietly = T)

  if (installedTENSORFLOW)attachNamespace("tensorflow")

  if(installTorch) packageStartupMessage("Torch tensors allowed")
  if (installedTENSORFLOW) packageStartupMessage("Tensorflow tensors allowed")
  if(installTorch==F & installedTENSORFLOW==F) packageStartupMessage("Not torch or tensorflow installed")

}


