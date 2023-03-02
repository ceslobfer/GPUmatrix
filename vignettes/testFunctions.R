
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
  M <- as(a, "dgCMatrix")
  crossprod(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  crossprod(y,M)
  library(float)
  floatA <- fl(a)
  crossprod(y,floatA)
  crossprod(y)
  crossprod(y,y)
  crossprod(y,a)
  crossprod(y,yS)


  M <- as(a, "dgeMatrix")
  tcrossprod(y,M)
  M <- as(a, "dgCMatrix")
  tcrossprod(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  tcrossprod(y,M)
  floatA <- fl(a)
  tcrossprod(y,floatA)
  tcrossprod(y)
  tcrossprod(y,y)
  tcrossprod(y,a)
  tcrossprod(y,yS)

  M <- as(a, "dgeMatrix")
  solve(y,M)
  M <- as(a, "dgCMatrix")
  solve(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  solve(t(y),M)
  floatA <- fl(a)
  solve(t(y),floatA)
  solve(t(y))
  solve(t(y),y)
  solve(t(y),a)
  solve(t(y),yS)


  M <- as(a, "dgeMatrix")
  cbind(y,M)
  M <- as(a, "dgCMatrix")
  cbind(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  cbind(y,M)
  floatA <- fl(a)
  cbind(y,gpu.matrix(floatA, device = GPUmatrix:::device(y)))
  cbind(y,y)
  cbind(y,a)
  cbind(y,yS)

  M <- as(a, "dgeMatrix")
  rbind(y,M)
  M <- as(a, "dgCMatrix")
  rbind(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  rbind(y,M)
  floatA <- fl(a)
  rbind(y,gpu.matrix(floatA, device = GPUmatrix:::device(y)))
  rbind(y,y)
  rbind(y,a)
  rbind(y,yS)

  M <- as(a, "dgeMatrix")
  chol_solve(y,M)
  M <- as(a, "dgCMatrix")
  chol_solve(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  chol_solve(y,M)
  floatA <- fl(a)
  chol_solve(y,gpu.matrix(floatA, device = GPUmatrix:::device(y)))
  chol_solve(y,y)
  chol_solve(y,a)
  chol_solve(y,yS)

  M <- as(a, "dgeMatrix")
  cor(y,M)
  M <- as(a, "dgCMatrix")
  cor(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  cor(y,M)
  floatA <- fl(a)
  cor(y,floatA)
  cor(y)
  cor(y,y)
  cor(y,a)
  cor(y,yS)

  M <- as(a, "dgeMatrix")
  cor(y,M, method = "spearman")
  M <- as(a, "dgCMatrix")
  cor(y,M, method = "spearman")
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  cor(y,M, method = "spearman")
  floatA <- fl(a)
  cor(y,floatA, method = "spearman")
  cor(y, method = "spearman")
  cor(y,y, method = "spearman")
  cor(y,a, method = "spearman")
  cor(y,yS, method = "spearman")

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

  M <- as(a, "dgeMatrix")
  y * M
  M <- as(a, "dgCMatrix")
  y * M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y * M
  floatA <- fl(a)
  y * floatA
  y * y
  y * a
  y * yS

  M <- as(a, "dgeMatrix")
  y + M
  M <- as(a, "dgCMatrix")
  y + M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y + M
  floatA <- fl(a)
  y + floatA
  y + y
  y + a
  y + yS

  M <- as(a, "dgeMatrix")
  y - M
  M <- as(a, "dgCMatrix")
  y - M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y - M
  floatA <- fl(a)
  y - floatA
  y - y
  y - a
  y - yS

  M <- as(a, "dgeMatrix")
  y / M
  M <- as(a, "dgCMatrix")
  y / M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y / M
  floatA <- fl(a)
  y / floatA
  y / y
  y / a
  y / yS

  M <- as(a, "dgeMatrix")
  y %*% M
  M <- as(a, "dgCMatrix")
  y %*% M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y %*% M
  floatA <- fl(a)
  y %*% floatA
  y %*% y
  y %*% a
  y %*% yS

  M <- as(a, "dgeMatrix")
  y %x% M
  M <- as(a, "dgCMatrix")
  y %x% M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y %x% M
  floatA <- fl(a)
  y %x% floatA
  y %x% y
  y %x% a
  y %x% yS

  M <- as(a, "dgeMatrix")
  y %% M
  M <- as(a, "dgCMatrix")
  y %% M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y %% M
  floatA <- fl(a)
  y %% floatA
  y %% y
  y %% a
  y %% yS

  M <- as(a, "dgeMatrix")
  y %o% M
  M <- as(a, "dgCMatrix")
  y %o% M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y %o% M
  floatA <- fl(a)
  y %o% gpu.matrix(floatA, device = GPUmatrix:::device(y))
  y %o% y
  y %o% a
  y %o% yS

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
  M <- as(a, "dgCMatrix")
  crossprod(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  crossprod(y,M)
  library(float)
  floatA <- fl(a)
  crossprod(y,floatA)
  crossprod(y)
  crossprod(y,y)
  crossprod(y,a)
  crossprod(y,yS)


  M <- as(a, "dgeMatrix")
  tcrossprod(y,M)
  M <- as(a, "dgCMatrix")
  tcrossprod(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  tcrossprod(y,M)
  floatA <- fl(a)
  tcrossprod(y,floatA)
  tcrossprod(y)
  tcrossprod(y,y)
  tcrossprod(y,a)
  tcrossprod(y,yS)

  M <- as(a, "dgeMatrix")
  solve(y,M)
  M <- as(a, "dgCMatrix")
  solve(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  solve(t(y),M)
  floatA <- fl(a)
  solve(t(y),floatA)
  solve(t(y))
  solve(t(y),y)
  solve(t(y),a)
  solve(t(y),yS)


  M <- as(a, "dgeMatrix")
  cbind(y,M)
  M <- as(a, "dgCMatrix")
  cbind(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  cbind(y,M)
  floatA <- fl(a)
  cbind(y,gpu.matrix(floatA, type = "tensorflow"))
  cbind(y,y)
  cbind(y,a)
  cbind(y,yS)

  M <- as(a, "dgeMatrix")
  rbind(y,M)
  M <- as(a, "dgCMatrix")
  rbind(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  rbind(y,M)
  floatA <- fl(a)
  rbind(y,gpu.matrix(floatA, type = "tensorflow"))
  rbind(y,y)
  rbind(y,a)
  rbind(y,yS)

  M <- as(a, "dgeMatrix")
  chol_solve(y,M)
  M <- as(a, "dgCMatrix")
  chol_solve(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  chol_solve(y,M)
  floatA <- fl(a)
  chol_solve(y,gpu.matrix(floatA, type = "tensorflow"))
  chol_solve(y,y)
  chol_solve(y,a)
  chol_solve(y,yS)

  M <- as(a, "dgeMatrix")
  cor(y,M)
  M <- as(a, "dgCMatrix")
  cor(y,M)
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  cor(y,M)
  floatA <- fl(a)
  cor(y,floatA)
  cor(y)
  cor(y,y)
  cor(y,a)
  cor(y,yS)

  M <- as(a, "dgeMatrix")
  cor(y,M, method = "spearman")
  M <- as(a, "dgCMatrix")
  cor(y,M, method = "spearman")
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  cor(y,M, method = "spearman")
  floatA <- fl(a)
  cor(y,floatA, method = "spearman")
  cor(y, method = "spearman")
  cor(y,y, method = "spearman")
  cor(y,a, method = "spearman")
  cor(y,yS, method = "spearman")

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

  M <- as(a, "dgeMatrix")
  y * M
  M <- as(a, "dgCMatrix")
  y * M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y * M
  floatA <- fl(a)
  y * floatA
  y * y
  y * a
  y * yS

  M <- as(a, "dgeMatrix")
  y + M
  M <- as(a, "dgCMatrix")
  y + M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y + M
  floatA <- fl(a)
  y + floatA
  y + y
  y + a
  y + yS

  M <- as(a, "dgeMatrix")
  y - M
  M <- as(a, "dgCMatrix")
  y - M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y - M
  floatA <- fl(a)
  y - floatA
  y - y
  y - a
  y - yS

  M <- as(a, "dgeMatrix")
  y / M
  M <- as(a, "dgCMatrix")
  y / M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y / M
  floatA <- fl(a)
  y / floatA
  y / y
  y / a
  y / yS

  M <- as(a, "dgeMatrix")
  y %*% M
  M <- as(a, "dgCMatrix")
  y %*% M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y %*% M
  floatA <- fl(a)
  y %*% floatA
  y %*% y
  y %*% a
  y %*% yS

  M <- as(a, "dgeMatrix")
  y %x% M
  M <- as(a, "dgCMatrix")
  y %x% M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y %x% M
  floatA <- fl(a)
  y %x% floatA
  y %x% y
  y %x% a
  y %x% yS

  M <- as(a, "dgeMatrix")
  y %% M
  M <- as(a, "dgCMatrix")
  y %% M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y %% M
  floatA <- fl(a)
  y %% floatA
  y %% y
  y %% a
  y %% yS

  M <- as(a, "dgeMatrix")
  y %o% M
  M <- as(a, "dgCMatrix")
  y %o% M
  X <- matrix(rnorm(nrow(y)*ncol(y), mean=0, sd=1), nrow(y), ncol(y))
  X <- X+1
  X <- nearPD(X, corr=TRUE)$mat
  M <- as(X, "dpoMatrix")
  y %o% M
  floatA <- fl(a)
  y %o% gpu.matrix(floatA, type = "tensorflow")
  y %o% y
  y %o% a
  y %o% yS

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
  # device = "cuda"
  # if(!y@gm$is_cuda) device="cpu"
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
y <- gpu.matrix(1:4,2,2, dimnames = list(c("a","b"),c("c","d")), type = "tensorflow")
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
