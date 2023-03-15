

.onAttach <- function (libname, pkgname) {

  installedTORCH <- requireNamespace("torch", quietly = T)
  installedTENSORFLOW <- requireNamespace("tensorflow", quietly = T)
  if (installedTORCH) attachNamespace("torch")
  if (installedTENSORFLOW) attachNamespace("tensorflow")

  if (installedTORCH) packageStartupMessage("Torch tensors allowed")
  if (installedTENSORFLOW) packageStartupMessage("Tensorflow tensors allowed")
  if(!installedTORCH & !installedTENSORFLOW) packageStartupMessage("Not torch or tensorflow installed")

}


