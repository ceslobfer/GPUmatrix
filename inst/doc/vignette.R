## ----eval=F-------------------------------------------------------------------
#  install.packages("torch")
#  library(torch)
#  install_torch() # In some cases is required.

## ---- eval=F------------------------------------------------------------------
#  install.packages("tensorflow")
#  library(tensorflow)
#  install_tensorflow(version = "nightly-gpu")

## ----eval=FALSE---------------------------------------------------------------
#  install.packages("GPUmarix")

## ----eval=FALSE---------------------------------------------------------------
#  devtools::install_github(" ceslobfer/GPUmatrix")

## ---- eval=T------------------------------------------------------------------
library(GPUmatrix)
#R matrix initialization
if(installTorch){
  m <- matrix(c(1:20)+40,10,2)
  #Show CPU matrix
  m
  #GPU matrix initialization
  Gm <- gpu.matrix(c(1:20)+40,10,2)
  #Show GPU matrix
  Gm
}

## ----eval=T-------------------------------------------------------------------
if(installTorch){
  Gm[c(2,3),1]
  
  Gm[,2]
   
  Gm2 <- cbind(Gm[c(1,2),], Gm[c(6,7),])
  Gm2
  
  Gm2[1,3] <- 0
  Gm2
}

## ---- echo=FALSE--------------------------------------------------------------
knitr::kable(NULL,caption = "Table 1. Casting operations between different packages")

## ---- eval=T------------------------------------------------------------------
if(installTorch){
  #Create 'Gm' from 'm' matrix
  m <- matrix(c(1:20)+40,10,2)
  Gm <- gpu.matrix(m)
  Gm
  
  #Create 'Gm' from 'M' with Matrix package
  library(Matrix)
  M <- Matrix(c(1:20)+40,10,2)
  Gm <- gpu.matrix(M)
  Gm
  
  #Create 'Gm' from 'mfloat32' with float package
  library(float)
  mfloat32 <- fl(m)
  Gm <- gpu.matrix(mfloat32)
  Gm
   
  #Create 'Gms' type sparse from 'Ms' type sparse dgCMatrix with Matrix package
  Ms <- Matrix(sample(0:1, 20, replace = TRUE), nrow=10, ncol=2, sparse=TRUE)
  Ms
   
  Gms <- gpu.matrix(Ms)
  Gms
}

## ---- eval=T------------------------------------------------------------------
if(installTorch){
  #Creating a float32 matrix
  Gm32 <- gpu.matrix(c(1:20)+40,10,2, dtype = "float32")
  Gm32
  
  #Creating a non sparse martix with data type float32 from a sparse matrix type float64
  Ms <- Matrix(sample(0:1, 20, replace = TRUE), nrow=10, ncol=2, sparse=TRUE)
  Gm32 <- gpu.matrix(Ms, dtype = "float32", sparse = F)
  Gm32
   
  #Convert Gm32 in sparse matrix Gms32
  Gms32 <- to_sparse(Gm32)
  Gms32
  
  ##Convert data type Gms32 in float64
  Gms64 <- Gms32
  dtype(Gms64) <- "float64"
  Gms64
}


## ---- eval=T------------------------------------------------------------------
if(installTorch){
  (Gm + Gm) == (m + m)
  
  (Gm + M) == (mfloat32 + Gm)
  
  (M + M) == (mfloat32 + Gm)
  
  (Ms + Ms) > (Gms + Gms)*2
}

## ---- echo=FALSE--------------------------------------------------------------
knitr::kable(NULL,caption = "Table 2. Mathematical operators that accept a gpu.matrix as input")

## ----eval=T-------------------------------------------------------------------
if(installTorch){
  m <- matrix(c(1:20)+40,10,2)
  Gm <- gpu.matrix(c(1:20)+40,10,2)
  
  head(tcrossprod(m),1)
  
  head(tcrossprod(Gm),1)
  
  Gm <- tail(Gm,3)
  rownames(Gm) <- c("a","b","c")
  tail(Gm,2)
  
  colMaxs(Gm)
}

## ---- echo=FALSE--------------------------------------------------------------
knitr::kable(NULL,caption = "Table 3. Functions that accept one or several gpu.matrix matrices as input")

## ----eval=T-------------------------------------------------------------------
updateH <- function(V,W,H) {
  H <- H * (t(W) %*% V)/((t(W) %*% W) %*% H)}
updateW <- function(V,W,H) {
  W <- W * (V %*% t(H))/(W %*% (H %*% t(H)) )}

## ----eval=T-------------------------------------------------------------------
A <- matrix(runif(200*10),200,10)
B <- matrix(runif(10*100),10,100)
V <- A %*% B

W <- W1 <- matrix(runif(200*10),200,10)
H <- H1 <- matrix(runif(10*100),10,100)

for (iter in 1:100) {
  W <- updateW(V,W,H)
  H <- updateH(V,W,H)
}
print(W[1,1])
print(H[1,1])

## ----eval=T-------------------------------------------------------------------
if(installTorch){
  library(GPUmatrix)
  Vg <- gpu.matrix(V)
  
  Wg <- gpu.matrix(W1)
  Hg <- gpu.matrix(H1)
  
  for (iter in 1:100) {
    Wg <- updateW(Vg,Wg,Hg)
    Hg <- updateH(Vg,Wg,Hg)
  }
  
  print(Wg[1,1])
  print(Hg[1,1])
}

## ---- eval=T------------------------------------------------------------------
if(installTorch){
  #GPUmatrix initialization with CPU option
  Gm <- gpu.matrix(c(1:20)+40,10,2,device="cpu")
  #Show CPU matrix from GPUmatrix
  Gm
}


## ---- eval=F------------------------------------------------------------------
#  
#  # library(GPUmatrix)
#  tensorflowGPUmatrix <- gpu.matrix(c(1:20)+40,10,2, type = "tensorflow")
#  tensorflowGPUmatrix

