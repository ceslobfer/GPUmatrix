
a <- matrix(c(5,1,1,3),2,2)
a <- matrix(rnorm(16),4,4)

x <- gpu.matrix(a, type = "tensorflow")
xS <- gpu.matrix(a, type = "tensorflow",sparse = T)
y <- gpu.matrix(a)
yS <- gpu.matrix(a,sparse = T)


generar_strings_aleatorios <- function(n, k) {
  caracteres <- c("a","c","f","u")
  res <- replicate(n, paste(sample(caracteres, k, replace = T), collapse = ""))
  return(res)
}
generar_strings_aleatorios(10, 8)


testFunctions <- function(y){

  determinant(y)
  determinant(y,logarithm = F)
  det(y)
  fft(y)
  sort(y)
  sort(y,decreasing = T)
  round(y)
  round(y,digits = 1)
  length(y)
  dim(y) <- dim(y)
  rownames(y) <- generar_strings_aleatorios(nrow(y),8)
  colnames(y) <- generar_strings_aleatorios(ncol(y),8)
  dimnames(y) <- list(generar_strings_aleatorios(nrow(y),8), generar_strings_aleatorios(ncol(y),8))
  y[,c(1,2)]
  y[,c(1,1,2)]
  y[nrow(y),ncol(y)]
  y[c(1,2),]
  y[c(1,1,2),]
  y[c(1,2)]
  y[c(1,1,2)]
  y[nrow(y)*ncol(y)]
  rowSums(y)
  colSums(y)
  head(y)
  head(y,1)
  tail(y)
  tail(y,1)
  t(y)
  if(y@sparse){
    yS <- to_dense(y)
  }else{
    yS <- to_sparse(y)
  }


  a <- matrix(rnorm(nrow(y)*ncol(y)),nrow(y),ncol(y))

  library(Matrix)
  M <- as(a, "dgeMatrix")
  crossprod(y,M)
  crossprod(M,y)
  tcrossprod(y,M)
  tcrossprod(M,y)
  solve(y,M)
  solve(M,y)
  cbind(y,M)
  cbind(M,y)
  rbind(y,M)
  rbind(M,y)
  chol_solve(y,M)
  chol_solve(M,y)
  cor(y,M)
  cor(M,y)
  cor(y,M,method = "spearman")
  cor(M,y,method = "spearman")
  cov(y,M)
  cov(M,y)
  y * M
  M * y
  y + M
  M + y
  y - M
  M - y
  y %*% M
  M %*% y
  y %x% M
  M %x% y
  y %% M
  M %% y
  y / M
  M / y
  M <- as(a, "dgCMatrix")
  crossprod(y,M)
  crossprod(M,y)
  tcrossprod(y,M)
  tcrossprod(M,y)
  solve(y,M)
  solve(M,y)
  cbind(y,M)
  cbind(M,y)
  rbind(y,M)
  rbind(M,y)
  chol_solve(y,M)
  chol_solve(M,y)
  cor(y,M)
  cor(M,y)
  cor(y,M,method = "spearman")
  cor(M,y,method = "spearman")
  cov(y,M)
  cov(M,y)
  y * M
  M * y
  y + M
  M + y
  y - M
  M - y
  y %*% M
  M %*% y
  y %x% M
  M %x% y
  y %% M
  M %% y
  y / M
  M / y
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  crossprod(y,M)
  crossprod(M,y)
  tcrossprod(y,M)
  tcrossprod(M,y)
  solve(y,M)
  solve(M,y)
  cbind(y,M)
  cbind(M,y)
  rbind(y,M)
  rbind(M,y)
  chol_solve(y,M)
  chol_solve(M,y)
  cor(y,M)
  cor(M,y)
  cor(y,M,method = "spearman")
  cor(M,y,method = "spearman")
  cov(y,M)
  cov(M,y)
  y * M
  M * y
  y + M
  M + y
  y - M
  M - y
  y %*% M
  M %*% y
  y %x% M
  M %x% y
  y %% M
  M %% y
  y / M
  M / y
  library(float)
  floatA <- fl(a)
  crossprod(y,floatA)
  crossprod(floatA,y)
  tcrossprod(y,floatA)
  tcrossprod(floatA,y)
  solve(y,floatA)
  solve(gpu.matrix(floatA, device = GPUmatrix:::device(y)),y)
  cbind(y,gpu.matrix(floatA, device = GPUmatrix:::device(y)))
  cbind(gpu.matrix(floatA, device = GPUmatrix:::device(y)),y)
  rbind(y,gpu.matrix(floatA, device = GPUmatrix:::device(y)))
  rbind(gpu.matrix(floatA, device = GPUmatrix:::device(y) ),y)
  chol_solve(y,gpu.matrix(floatA, device = GPUmatrix:::device(y)))
  chol_solve(gpu.matrix(floatA, device = GPUmatrix:::device(y)),y)
  cor(y,floatA)
  cor(floatA,y)
  cor(y,floatA,method = "spearman")
  cor(floatA,y,method = "spearman")
  cov(y,floatA)
  cov(floatA,y)
  y * floatA
  floatA * y
  y + floatA
  floatA + y
  y - floatA
  floatA - y
  y %*% floatA
  floatA %*% y
  y %x% floatA
  floatA %x% y
  y %% floatA
  floatA %% y
  y / floatA
  floatA / y
  #Sparse
  crossprod(y,yS)
  tcrossprod(y,yS)
  solve(y,yS)
  cbind(y,yS)
  rbind(y,yS)
  crossprod(yS,y)
  tcrossprod(yS,y)
  solve(yS,y)
  cbind(yS,y)
  rbind(yS,y)
  chol_solve(y,yS)
  chol_solve(yS,y)
  cor(y,yS)
  cor(yS,y)
  cor(y,yS,method = "spearman")
  cor(yS,y,method = "spearman")
  cov(y,yS)
  cov(yS,y)
  y * yS
  yS * y
  y + yS
  yS + y
  y - yS
  yS - y
  y %*% yS
  yS %*% y
  y %x% yS
  yS %x% y
  y %% yS
  yS %% y
  y / yS
  yS / y
  #SINGLE
  crossprod(y)
  tcrossprod(y)
  solve(y)
  chol_solve(M,y)
  cor(y)
  cor(y,method = "spearman")
  cov(y)
  #A
  crossprod(y,a)
  crossprod(a,y)
  tcrossprod(y,a)
  tcrossprod(a,y)
  solve(y,a)
  solve(a,y)
  cbind(y,a)
  cbind(a,y)
  rbind(y,a)
  rbind(a,y)
  chol_solve(y,a)
  chol_solve(a,y)
  cor(y,a)
  cor(a,y)
  cor(y,a,method = "spearman")
  cor(a,y,method = "spearman")
  cov(y,a)
  cov(a,y)
  y * a
  a * y
  y + a
  a + y
  y - a
  a - y
  y %*% a
  a %*% y
  y %x% a
  a %x% y
  y %% a
  a %% y
  y / a
  a / y

  log(y)
  log2(y)
  log10(y)
  log1p(y)
  cos(y)
  cosh(y)
  acos(y)
  acosh(y)
  sin(y)
  sinh(y)
  asin(y)
  asinh(y)
  tan(y)
  atan(y)
  tanh(y)
  atanh(y)
  sqrt(y)
  abs(y)
  sign(y)
  ceiling(y)
  floor(y)
  cumsum(y)
  cumprod(y)
  exp(y)
  expm1(y)

  diag(y) <- diag(y)
  qr(y)
  rankMatrix(y)
  eigen(y)
  svd(y)
  ginv(y)

  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  device = "cuda"
  if(!y@gm$is_cuda) device="cpu"
  X <- gpu.matrix(X, dtype = dtype(y), device = device, dimnames = dimnames(y) )
  chol(X)

  mean(y)
  density(y)
  plot <- hist(y)
  colMeans(y)
  rowMeans(y)
  sum(y)
  min(y)
  max(y)
  which.max(y)
  which.min(y)
  cov(y)
  cov2cor(y)
  rowVars(y)
  colVars(y)
  colMaxs(y)
  colMins(y)
  rowRanks(y)
  res <- colRanks(y)

}


testFunctions_tensorflow <- function(y){

  determinant(y)
  determinant(y,logarithm = F)
  det(y)
  fft(y)
  sort(y)
  sort(y,decreasing = T)
  round(y)
  round(y,digits = 1)
  length(y)
  dim(y) <- dim(y)
  rownames(y) <- generar_strings_aleatorios(nrow(y),8)
  colnames(y) <- generar_strings_aleatorios(ncol(y),8)
  dimnames(y) <- list(generar_strings_aleatorios(nrow(y),8), generar_strings_aleatorios(ncol(y),8))
  y[,c(1,2)]
  y[,c(1,1,2)]
  y[nrow(y),ncol(y)]
  y[c(1,2),]
  y[c(1,1,2),]
  y[c(1,2)]
  y[c(1,1,2)]
  y[nrow(y)*ncol(y)]
  rowSums(y)
  colSums(y)
  head(y)
  head(y,1)
  tail(y)
  tail(y,1)
  t(y)
  if(y@sparse){
    yS <- to_dense(y)
  }else{
    yS <- to_sparse(y)
  }


  a <- matrix(rnorm(nrow(y)*ncol(y)),nrow(y),ncol(y))


  library(Matrix)
  M <- as(a, "dgeMatrix")
  crossprod(y,M)
  crossprod(M,y)
  tcrossprod(y,M)
  tcrossprod(M,y)
  solve(y,M)
  solve(M,y)
  cbind(y,M)
  cbind(M,y)
  rbind(y,M)
  rbind(M,y)
  chol_solve(y,M)
  chol_solve(M,y)
  cor(y,M)
  cor(M,y)
  cor(y,M,method = "spearman")
  cor(M,y,method = "spearman")
  cov(y,M)
  cov(M,y)
  y * M
  M * y
  y + M
  M + y
  y - M
  M - y
  y %*% M
  M %*% y
  y %x% M
  M %x% y
  y %% M
  M %% y
  y / M
  M / y
  M <- as(a, "dgCMatrix")
  crossprod(y,M)
  crossprod(M,y)
  tcrossprod(y,M)
  tcrossprod(M,y)
  solve(y,M)
  solve(M,y)
  cbind(y,M)
  cbind(M,y)
  rbind(y,M)
  rbind(M,y)
  chol_solve(y,M)
  chol_solve(M,y)
  cor(y,M)
  cor(M,y)
  cor(y,M,method = "spearman")
  cor(M,y,method = "spearman")
  cov(y,M)
  cov(M,y)
  y * M
  M * y
  y + M
  M + y
  y - M
  M - y
  y %*% M
  M %*% y
  y %x% M
  M %x% y
  y %% M
  M %% y
  y / M
  M / y
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  crossprod(y,M)
  crossprod(M,y)
  tcrossprod(y,M)
  tcrossprod(M,y)
  solve(y,M)
  solve(M,y)
  cbind(y,M)
  cbind(M,y)
  rbind(y,M)
  rbind(M,y)
  chol_solve(y,M)
  chol_solve(M,y)
  cor(y,M)
  cor(M,y)
  cor(y,M,method = "spearman")
  cor(M,y,method = "spearman")
  cov(y,M)
  cov(M,y)
  y * M
  M * y
  y + M
  M + y
  y - M
  M - y
  y %*% M
  M %*% y
  y %x% M
  M %x% y
  y %% M
  M %% y
  y / M
  M / y
  library(float)
  floatA <- fl(a)
  crossprod(y,floatA)
  crossprod(floatA,y)
  tcrossprod(y,floatA)
  tcrossprod(floatA,y)
  solve(y,floatA)
  solve(gpu.matrix(floatA, type="tensorflow"),y)
  cbind(y,gpu.matrix(floatA, type = "tensorflow"))
  cbind(gpu.matrix(floatA, type = "tensorflow"),y)
  rbind(y,gpu.matrix(floatA, type = "tensorflow"))
  rbind(gpu.matrix(floatA, type = "tensorflow"),y)
  chol_solve(y,gpu.matrix(floatA, type = "tensorflow"))
  chol_solve(gpu.matrix(floatA, type = "tensorflow"),y)
  cor(y,floatA)
  cor(floatA,y)
  cor(y,floatA,method = "spearman")
  cor(floatA,y,method = "spearman")
  cov(y,floatA)
  cov(floatA,y)
  y * floatA
  floatA * y
  y + floatA
  floatA + y
  y - floatA
  floatA - y
  y %*% floatA
  floatA %*% y
  y %x% floatA
  floatA %x% y
  y %% floatA
  floatA %% y
  y / floatA
  floatA / y
  #Sparse
  crossprod(y,yS)
  tcrossprod(y,yS)
  solve(y,yS)
  cbind(y,yS)
  rbind(y,yS)
  crossprod(yS,y)
  tcrossprod(yS,y)
  solve(yS,y)
  cbind(yS,y)
  rbind(yS,y)
  chol_solve(y,yS)
  chol_solve(yS,y)
  cor(y,yS)
  cor(yS,y)
  cor(y,yS,method = "spearman")
  cor(yS,y,method = "spearman")
  cov(y,yS)
  cov(yS,y)
  y * yS
  yS * y
  y + yS
  yS + y
  y - yS
  yS - y
  y %*% yS
  yS %*% y
  y %x% yS
  yS %x% y
  y %% yS
  yS %% y
  y / yS
  yS / y
  #SINGLE
  crossprod(y)
  tcrossprod(y)
  solve(y)
  chol_solve(M,y)
  cor(y)
  cor(y,method = "spearman")
  cov(y)
  #A
  crossprod(y,a)
  crossprod(a,y)
  tcrossprod(y,a)
  tcrossprod(a,y)
  solve(y,a)
  solve(a,y)
  cbind(y,a)
  cbind(a,y)
  rbind(y,a)
  rbind(a,y)
  chol_solve(y,a)
  chol_solve(a,y)
  cor(y,a)
  cor(a,y)
  cor(y,a,method = "spearman")
  cor(a,y,method = "spearman")
  cov(y,a)
  cov(a,y)
  y * a
  a * y
  y + a
  a + y
  y - a
  a - y
  y %*% a
  a %*% y
  y %x% a
  a %x% y
  y %% a
  a %% y
  y / a
  a / y

  log(y)
  log2(y)
  log10(y)
  log1p(y)
  cos(y)
  cosh(y)
  acos(y)
  acosh(y)
  sin(y)
  sinh(y)
  asin(y)
  asinh(y)
  tan(y)
  atan(y)
  tanh(y)
  atanh(y)
  sqrt(y)
  abs(y)
  sign(y)
  ceiling(y)
  floor(y)
  cumsum(y)
  cumprod(y)
  exp(y)
  expm1(y)


  y %^% 2

  diag(y) <- diag(y)
  qr(y)
  rankMatrix(y)
  eigen(y)
  svd(y)
  ginv(y)

  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  X <- gpu.matrix(X, dtype = dtype(y), dimnames = dimnames(y) )
  chol(X)

  mean(y)
  density(y)
  plot <- hist(y)
  colMeans(y)
  rowMeans(y)
  sum(y)
  min(y)
  max(y)
  which.max(y)
  which.min(y)
  cov(y)
  cov2cor(y)
  rowVars(y)
  colVars(y)
  colMaxs(y)
  colMins(y)
  rowRanks(y)
  res <- colRanks(y)

}


##MATRICES TENSORFLOW
y <- gpu.matrix(rnorm(9),3,3, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "int", type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", type = "tensorflow")

y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T, type = "tensorflow")

# dgeMatrix: matriz densa
M <- matrix(c(1, 2, 3, 4), nrow = 2)
M <- as(M, "dgeMatrix")

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = F, type = "tensorflow")

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T, type = "tensorflow")

# dpoMatrix: matriz simétrica definida positiva
M <- matrix(c(2, 1, 1, 2), nrow = 2)
M <- as(M, "dpoMatrix")

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = F, type = "tensorflow")

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T, type = "tensorflow")

# dgCMatrix: matriz comprimida esparsa
M <- matrix(c(1, 1, 2, 1), nrow = 2)
M <- as(M, "dgCMatrix")

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = F, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = F, type = "tensorflow")

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T, type = "tensorflow")
suppressWarnings(testFunctions_tensorflow(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T, type = "tensorflow")



##MATRICES TORCH
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")))
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "int")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float64")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float32")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "bool")

y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T)

y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T, device = "cpu")

library(Matrix)

# dgeMatrix: matriz densa
M <- matrix(c(1, 2, 3, 4), nrow = 2)
M <- as(M, "dgeMatrix")

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = F)

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T)

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = F, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T, device = "cpu")


# dpoMatrix: matriz simétrica definida positiva
M <- matrix(c(2, 1, 1, 2), nrow = 2)
M <- as(M, "dpoMatrix")

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = F)

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T)

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = F, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T, device = "cpu")


# dgCMatrix: matriz comprimida esparsa
M <- matrix(c(1, 1, 2, 1), nrow = 2)
M <- as(M, "dgCMatrix")

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = F)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = F)

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T)
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T)

y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = F, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "int", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float64", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "float32", sparse = T, device = "cpu")
suppressWarnings(testFunctions(y))
y <- gpu.matrix(M, dimnames = list(c("a","b"),c("c","d")), dtype = "bool", sparse = T, device = "cpu")
