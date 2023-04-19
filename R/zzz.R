
# installedTORCH <- requireNamespace("torch", quietly = T)
#
# installedTENSORFLOW <- requireNamespace("tensorflow", quietly = T)

# functionTorchInstalled<- function(){
#   if(requireNamespace('torch') & installedTORCH){
#     return(torch::torch_is_installed())
#   }
# }
# installTorch<- functionTorchInstalled()
installTorch<-F
.onAttach <- function (libname, pkgname) {

  installedTORCH <- requireNamespace("torch", quietly = T)
  installedTENSORFLOW <- requireNamespace("tensorflow", quietly = T)
  # print(installedTORCH)
  # if (installedTORCH){
  #   tryCatch(attachNamespace("torch"))
    # installTorch <- torch::torch_is_installed()
    # print(installTorch)
    # if (installTorch) {
    #   attachNamespace("torch")
    # }

  # if (installedTORCH){
  #
  #   installTorch <- torch::torch_is_installed()
  #   if(installTorch) attachNamespace("torch")
  # }
  if (installedTENSORFLOW)attachNamespace("tensorflow")


  if(installedTORCH & installTorch) packageStartupMessage("Torch tensors allowed")
  if (installedTENSORFLOW) packageStartupMessage("Tensorflow tensors allowed")
  if(!installedTORCH & !installedTENSORFLOW) packageStartupMessage("Not torch or tensorflow installed")

}


