
installTorch <- function(){
  installedTORCH <- requireNamespace("torch", quietly = TRUE)
  tryCatch({
    res <- FALSE
    if (installedTORCH){
      res <- torch::torch_is_installed()
      if(res & !isNamespaceLoaded("torch")) attachNamespace("torch")
    }
    return(res)
  })
}

# Function called when a package is attached
.onAttach <- function (libname, pkgname) {

  installedTORCH <- requireNamespace("torch", quietly = TRUE)

  # Check if Tensorflow is installed
  installTorch <- tryCatch({
    res <- FALSE
    if (installedTORCH){
      res <- torch::torch_is_installed()
      if(res & !isNamespaceLoaded("torch")) attachNamespace("torch")
    }
    res
  })


  installedTENSORFLOW <- requireNamespace("tensorflow", quietly = TRUE)

  # Attach Tensorflow namespace if it is installed
  if (installedTENSORFLOW & !isNamespaceLoaded("tensorflow")) attachNamespace("tensorflow")

  # Display startup message depending on the installed packages
  if(installTorch) packageStartupMessage("Torch tensors allowed")
  if (installedTENSORFLOW) packageStartupMessage("Tensorflow tensors allowed")
  if(installTorch == FALSE & installedTENSORFLOW == FALSE) packageStartupMessage("Not torch or tensorflow installed")
  if(installTorch){
    if(torch::cuda_is_available()) packageStartupMessage("Your Torch installation have CUDA tensors available.")
    if(!torch::cuda_is_available()) packageStartupMessage("Your Torch installation does not have CUDA tensors available. Please check the Torch requirements and installation if you want to use CUDA tensors.")
  }

  # Set typeTensor option to "torch"
  options(typeTensor = "torch")



}


