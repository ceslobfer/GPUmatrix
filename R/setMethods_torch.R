
to_dense_torch<-function(x){
  if(x@sparse){
    x@gm <-x@gm$to_dense()
    x@sparse <- F
  }
  return(x)
}

# to_sparse_torch<-function(x){
#   if(!x@sparse){
#     x@gm <-x@gm$to_sparse()
#     x@sparse <- T
#   }
#   return(x)
# }

to_sparse_torch <- function(x) {
  # Encontrar los índices de los elementos no cero
  non_zero_indices <- which(x != 0, arr.ind = TRUE)
  device <- "cuda"
  if(!x@gm$is_cuda) device <- "cpu"

  # indices <- torch::torch_tensor(t(non_zero_indices),device = x@gm$device,dtype = torch::torch_long())
  indices <- gpu.matrix(t(non_zero_indices),device = device,dtype = torch::torch_long())


  # Extraer los valores no cero
  values <- as.numeric(x)
  values <- torch::torch_tensor(values[values != 0],device = x@gm$device,dtype = x@gm$dtype)

  # Crear el tensor disperso
  sparse_tensor <- torch::torch_sparse_coo_tensor(indices@gm, values, size = dim(x))
  res <- gpu.matrix(sparse_tensor,sparse = T,dtype = dtype(x), device = device)
  x@gm <- res@gm
  x@sparse <- T
  return(x)
}


typeGPUmatrix <- function(x){
  objectClass <- class(x)
  if(objectClass == "gpu.matrix.torch"){
    res <- "torch"
  }else{
    res <- "tensorflow"
  }
  return(res)
}

setClassUnion("numMatrixLike", members = c("logical", "integer", "numeric", "matrix"))
c.GPUmatrix <- function(...) unlist(lapply(list(...), as.vector))


setMethod("to_dense", signature(x = "gpu.matrix.torch"), function(x) to_dense_torch(x) )

setMethod("to_sparse", signature(x = "gpu.matrix.torch"), function(x) to_sparse_torch(x) )


setMethod("c", "gpu.matrix.torch", function(x, ..., recursive) c.GPUmatrix(x, ...))
setMethod("c", "numMatrixLike", function(x, ..., recursive) c.GPUmatrix(x, ...))

logdetTensor_torch <- function(x){
  value <- x@gm$slogdet()
  logAbDet <- as.numeric(value[[2]]$cpu())
  attr(logAbDet, which = "logarithm") <- TRUE
  sign<-sign(as.numeric(value[[1]]$cpu()))
  res <- list("modulus"=logAbDet, "sign"=sign)
  attr(res, which = "class") <- "det"
  return(res)
}

setMethod("determinant", signature(x = "gpu.matrix.torch", logarithm = "missing"), function(x, logarithm, ...){
  x <- warningSparseTensor_torch(x)
  x <- warningInteger(x)
  res <- logdetTensor_torch(x)
  return(res)
})
setMethod("determinant", signature(x = "gpu.matrix.torch", logarithm = "logical"), function(x, logarithm, ...){
  x <- warningSparseTensor_torch(x)
  x <- warningInteger(x)
  if (logarithm) {
    res <- logdetTensor_torch(x)
  }else{
    value <- x@gm$det()
    logAbDet <- as.numeric(value$cpu())
    attr(logAbDet, which = "logarithm") <- FALSE
    sign<-sign(as.numeric(logAbDet))
    res <- list("modulus"=abs(logAbDet), "sign"=sign)
    attr(res, which = "class") <- "det"
  }
  return(res)
})

setMethod("det", signature(x = "gpu.matrix.torch"), function(x, ...){
  x <- warningSparseTensor_torch(x)
  x <- warningInteger(x)
  res <- as.numeric(x@gm$det()$cpu())

  return(res)
})

setMethod("fft", signature(z="gpu.matrix.torch", inverse="missing"), function(z,inverse=F){
  z <- warningSparseTensor_torch(z)
  if(!(ncol(z)>1 & nrow(z)>1)){
    if(ncol(z)>1){
      z@gm <- torch::torch_fft_fft(z@gm,dim = 2)
    }else{
      z@gm <- torch::torch_fft_fft(z@gm,dim = 1)
    }
  }else{
    stop("FFT in gpu.matrix with 2 dimensions is not allowed yet")
  }

  return(z)
})


setMethod("fft", signature(z="gpu.matrix.torch", inverse="logical"), function(z,inverse=F){
  z <- warningSparseTensor_torch(z)

  if(!(ncol(z)>1 & nrow(z)>1)){
    if(ncol(z)>1){
      if(inverse){
        z@gm <- torch::torch_fft_ifft(z@gm, norm = "forward",dim = 2)
      }else{
        z@gm <- torch::torch_fft_fft(z@gm,dim = 2)
      }
    }else{
      if(inverse){
        z@gm <- torch::torch_fft_ifft(z@gm, norm = "forward",dim = 1)
      }else{
        z@gm <- torch::torch_fft_fft(z@gm,dim = 1)
      }
    }
  }else{
    stop("FFT in gpu.matrix with 2 dimensions is not allowed yet")
  }

  # if(inverse){
  #   z@gm <- torch::torch_fft_ifft(z@gm, norm = "forward",dim = 1)
  # }else{
  #   z@gm <- torch::torch_fft_fft(z@gm,dim = 1)
  # }
  return(z)
})

setMethod("mvfft", signature(z="gpu.matrix.torch", inverse="missing"), function(z,inverse=F){
  z <- warningSparseTensor_torch(z)
  z@gm <- torch::torch_fft_fft(z@gm,dim = 1)

  return(z)
})

setMethod("mvfft", signature(z="gpu.matrix.torch", inverse="logical"), function(z,inverse=F){
  z <- warningSparseTensor_torch(z)
  if(inverse){
    z@gm <- torch::torch_fft_ifft(z@gm, norm = "forward",dim = 1)
  }else{
    z@gm <- torch::torch_fft_fft(z@gm,dim = 1)
  }
  return(z)
})

setMethod("sort", signature(x="gpu.matrix.torch"), function(x,decreasing=FALSE,...){

  if (!decreasing) {
    if (x@sparse) {
      res <- torch::torch_sort(x@gm$values())[[1]]
    }else{
      res<- torch::torch_sort(x@gm$reshape(length(x)))[[1]]
    }
  }else{
    if (x@sparse) {
      res <- torch::torch_sort(x@gm$values(),descending =T)[[1]]
    }else{
      res <- torch::torch_sort(x@gm$reshape(length(x)),descending =T)[[1]]
    }


  }

  return(res)
})

setMethod("round", signature(x= "gpu.matrix.torch",digits="missing"), function(x,digits){
  x <- warningInteger(x)
  if(x@sparse){
    oldDtype <- dtype(x)
    indices <- torch::torch_tensor(x@gm$indices()+1, dtype = torch::torch_long())
    x@gm <- torch::torch_sparse_coo_tensor(indices = indices, values = torch::torch_round(x@gm$values(),decimals = 0), size = x@gm$size(), device = torch::torch_device(type = device(x)))
    x@gm <- x@gm$coalesce()
    dtype(x) <- oldDtype
  }else{
    x@gm <- torch::torch_round(x@gm,decimals = 0)
  }
  return(x)
})

setMethod("round", signature(x= "gpu.matrix.torch",digits="numeric"), function(x,digits){
  x <- warningInteger(x)
  if(x@sparse){
    oldDtype <- dtype(x)
    indices <- torch::torch_tensor(x@gm$indices()+1, dtype = torch::torch_long())

    x@gm <- torch::torch_sparse_coo_tensor(indices = indices, values = torch::torch_round(x@gm$values(),decimals = digits), size = x@gm$size(), device = torch::torch_device(type = device(x)))
    x@gm <- x@gm$coalesce()
    dtype(x) <- oldDtype
  }else{
    x@gm <- torch::torch_round(x@gm,decimals = digits)
  }
  return(x)
})

setMethod(f = "show", signature = "gpu.matrix.torch", definition = function(object){
  if (dtype(object) == "complex64" | dtype(object) == "complex32"){
    cat("GPUmatrix\n")
    cat("Real (Re function):\n")
    print(Re(object))
    cat("Imag (Im function):\n")
    print(Im(object))
  }else{
    cat("GPUmatrix\n")
    print(object@gm)
    if (!is.null(object@rownames)) cat(paste(c("rownames:",object@rownames,"\n")))
    if (!is.null(object@colnames)) cat(paste(c("colnames:",object@colnames,"\n")))
  }
})

setMethod("print", signature = "gpu.matrix.torch", definition = function(x){
  object <- x
  if (dtype(object) == "complex64" | dtype(object) == "complex32"){
    cat("GPUmatrix\n")
    cat("Real (Re function):\n")
    print(Re(object))
    cat("Imag (Im function):\n")
    print(Im(object))
  }else{
    print(object@gm)
    if (!is.null(object@rownames)) cat(paste(c("rownames:",object@rownames,"\n")))
    if (!is.null(object@colnames)) cat(paste(c("colnames:",object@colnames,"\n")))
  }
})

setMethod("length", signature(x = "gpu.matrix.torch"), function(x){
  return(length(x@gm))
} )


setMethod("dim", signature(x = "gpu.matrix.torch"), function(x){dim(x@gm)})
setMethod("dim<-", signature(x = "gpu.matrix.torch",value="vector"), function(x,value){
  x <- t(x)

  if (x@sparse) {
    x <- warningSparseTensor_torch(x)
    x@gm <- x@gm$reshape(rev(value))
    x <- t(x)
    x <- to_sparse(x)
    x@gm <- x@gm$coalesce()
  }else{
    x@gm <- x@gm$reshape(rev(value))
    x <- t(x)
  }
  return(x)
})


setMethod("dimnames", signature(x = "gpu.matrix.torch"), function(x){
  if (is.null(c(x@rownames,x@colnames))) {
    res <- NULL
  }else{
    res <- list(x@rownames,x@colnames)
  }
  return(res)
})

setMethod("dimnames<-", signature(x = "gpu.matrix.torch", value="vector"), function(x,value){

  if (is.null(value[[1]]) & is.null(value[[2]])){
    x@rownames <- NULL
    x@colnames <- NULL
  }else if (is.null(value[[1]]) & length(value[[2]]) == ncol(x)){
    x@colnames <- value[[2]]
    x@rownames <- NULL
  }else if (is.null(value[[2]]) & length(value[[1]]) == nrow(x)){
    x@rownames <- value[[1]]
    x@colnames <- NULL
  }else if (length(value[[1]]) == nrow(x) & length(value[[2]]) == ncol(x)) {
    x@rownames <- value[[1]]
    x@colnames <- value[[2]]
  }else{
    stop("Error dimension not match")
  }

  return(x)
})

setMethod("rownames", signature(x = "gpu.matrix.torch"), function(x){
  return(x@rownames)
})
setMethod("row.names", signature(x = "gpu.matrix.torch"), function(x){
  return(rownames(x))
})
setMethod("rownames<-", signature(x = "gpu.matrix.torch", value="vector"), function(x,value){
  if (length(value) != nrow(x))  stop("length of 'colnames' not equal to array extent")

  if (is.null(value)) value <- c()
  x@rownames <- value
  return(x)
})
setMethod("row.names<-", signature(x = "gpu.matrix.torch", value="vector"), function(x,value){
  return(rownames(x) <- value)
})
setMethod("colnames", signature(x = "gpu.matrix.torch"), function(x){
  return(x@colnames)
})
setMethod("colnames<-", signature(x = "gpu.matrix.torch", value="vector"), function(x,value){
  if (length(value) != ncol(x))  stop("length of 'colnames' not equal to array extent")
  if (is.null(value)) value <- c()
  x@colnames <- value
  return(x)
})

setMethod("rowSums", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  return(as.numeric(torch::torch_sum(x@gm,2)$cpu()))
})
setMethod("colSums", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  return(as.numeric(torch::torch_sum(x@gm, 1)$cpu()))
})


setMethod("cbind2",signature(x = "gpu.matrix.torch", y = "ANY"), function(x,y,...){

  castMatrix <- castTypeOperations_torch(x,y, todense=F)
  yOrigin <- y
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  if (x@sparse != y@sparse) {
    x <- warningSparseTensor_torch(x)
    y <- warningSparseTensor_torch(y)
  }
  res <- gpu.matrix.torch(torch::torch_cat(tensors = c(x@gm,y@gm), dim = 2), device=device(x))
  if (is.null(colnames(x)) & !is.null(colnames(y))) colnames(x) <- rep(NA,ncol(x))
  if (is.null(colnames(y)) & !is.null(colnames(x)) & !is.vector(yOrigin)) colnames(y) <- rep(NA,ncol(y))
  if (is.null(colnames(y)) & !is.null(colnames(x)) & is.vector(yOrigin)) y@colnames <- NA

  rNames <- NULL
  if (!is.null(rownames(x))) rNames <- rownames(x)
  if (is.null(rownames(x)) & !is.null(rownames(y))) rNames <- rownames(y)
  if (!is.null(rownames(x)) & !is.null(rownames(y))) rNames <- c(rownames(x),rownames(y))[c(1:nrow(x))]


  dimnames(res) <- list(rNames,c(colnames(x), colnames(y)))


  return(res)
})

setMethod("cbind2",signature(x = "ANY", y = "gpu.matrix.torch"), function(x,y){

  castMatrix <- castTypeOperations_torch(x,y, todense=F)
  xOrigin <- x
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  if (x@sparse != y@sparse) {
    x <- warningSparseTensor_torch(x)
    y <- warningSparseTensor_torch(y)
  }
  res <- gpu.matrix.torch(torch::torch_cat(tensors = c(x@gm,y@gm), dim = 2), device=device(y))
  if (is.null(colnames(x)) & !is.null(colnames(y)) & !is.vector(xOrigin)) colnames(x) <- rep(NA,ncol(x))
  if (is.null(colnames(x)) & !is.null(colnames(y)) & is.vector(xOrigin)) x@colnames<- NA
  if (is.null(colnames(y)) & !is.null(colnames(x))) colnames(y) <- rep(NA,ncol(y))
  rNames <- NULL
  if (!is.null(rownames(x))) rNames <- rownames(x)
  if (is.null(rownames(x)) & !is.null(rownames(y))) rNames <- rownames(y)
  if (!is.null(rownames(x)) & !is.null(rownames(y))) rNames <- c(rownames(x),rownames(y))[c(1:nrow(x))]

  dimnames(res) <- list(rNames,c(colnames(x), colnames(y)))


  return(res)
})

setMethod("rbind2", signature(x = "gpu.matrix.torch", y = "ANY"), function(x,y){
  if(is.vector(y)) y <- matrix(y, ncol = length(y), nrow = 1)
  castMatrix <- castTypeOperations_torch(x,y, todense=F)
  yOrigin <- y
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  if (x@sparse != y@sparse) {
    x <- warningSparseTensor_torch(x)
    y <- warningSparseTensor_torch(y)
  }
  res <- gpu.matrix.torch(torch::torch_cat(tensors = c(x@gm,y@gm), dim = 1), device=device(x))


  if (is.null(rownames(x)) & !is.null(rownames(y))) rownames(x) <- rep(NA,nrow(x))
  if (is.null(rownames(y)) & !is.null(rownames(x)) & !is.vector(yOrigin)) rownames(y) <- rep(NA,nrow(y))
  if (is.null(rownames(y)) & !is.null(rownames(x)) & is.vector(yOrigin)) y@rownames <- NA
  cNames <- NULL
  if (!is.null(colnames(x))) cNames <- colnames(x)
  if (is.null(colnames(x)) & !is.null(colnames(y))) cNames <- colnames(y)
  if (!is.null(colnames(x)) & !is.null(colnames(y))) cNames <- c(colnames(x),colnames(y))[c(1:ncol(x))]
  dimnames(res) <- list(c(rownames(x),rownames(y)),cNames)
  return(res)
})

setMethod("rbind2",signature(x = "ANY", y = "gpu.matrix.torch"), function(x,y){
  if(is.vector(x)) x <- matrix(x, ncol = length(x), nrow = 1)
  castMatrix <- castTypeOperations_torch(x,y, todense=F)
  xOrigin <- x
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  if (x@sparse != y@sparse) {
    x <- warningSparseTensor_torch(x)
    y <- warningSparseTensor_torch(y)
  }
  res <- gpu.matrix.torch(torch::torch_cat(tensors = c(x@gm,y@gm), dim = 1), device=device(y))


  if (is.null(rownames(x)) & !is.null(rownames(y)) & !is.vector(xOrigin)) rownames(x) <- rep(NA,nrow(x))
  if (is.null(rownames(x)) & !is.null(rownames(y)) & is.vector(xOrigin)) x@rownames <- NA
  if (is.null(rownames(y)) & !is.null(rownames(x))) rownames(y) <- rep(NA,nrow(y))
  cNames <- NULL
  if (!is.null(colnames(x))) cNames <- colnames(x)
  if (is.null(colnames(x)) & !is.null(colnames(y))) cNames <- colnames(y)
  if (!is.null(colnames(x)) & !is.null(colnames(y))) cNames <- c(colnames(x),colnames(y))[c(1:ncol(x))]

  dimnames(res) <- list(c(rownames(x),rownames(y)),cNames)

  return(res)
})



setMethod("head", signature(x = "gpu.matrix.torch"), function(x, ...){
  x <- warningSparseTensor_torch(x)
  x@gm <- head(x@gm,...)
  rownames(x) <- head(x@rownames,...)
  return(x)
  })


setMethod("tail", signature(x = "gpu.matrix.torch"), function(x, ...){
  x <- warningSparseTensor_torch(x)
  x@gm <- tail(x@gm,...)
  rownames(x) <- tail(x@rownames,...)
  return(x)
  })

setMethod("nrow", signature(x = "gpu.matrix.torch"), function(x){
  return(nrow(x@gm))
} )

setMethod("ncol", signature(x = "gpu.matrix.torch"), function(x){
  return(ncol(x@gm))
} )

setMethod("t", signature(x = "gpu.matrix.torch"), function(x){
  if (length(dim(x@gm))<2){
    x <- gpu.matrix.torch(x@gm,nrow=nrow(x),ncol=ncol(x),sparse=x@sparse,rownames = rownames(x),colnames = colnames(x), dtype=x@gm$dtype, device = device(x))
  }
  res <- gpu.matrix.torch(torch::torch_transpose(self = x@gm,dim0 = 1,dim1 = 2),sparse=x@sparse,rownames = colnames(x),colnames = rownames(x), dtype=x@gm$dtype, device = device(x))

  return(res)
})

setMethod("crossprod", signature(x = "gpu.matrix.torch", y = "ANY"), function(x,y, ...){
  if (is.null(y)) {
    return(t(x) %*% x)
  }else{
    castMatrix <- castTypeOperations_torch(x,y, todense=FALSE)
    x <- castMatrix[[1]]
    y <- castMatrix[[2]]
    return(t(x) %*% y)
  }

} )

setMethod("crossprod", signature(x = "gpu.matrix.torch", y = "missing"), function(x,y, ...){
  return(t(x) %*% x)

} )

setMethod("crossprod", signature(x = "ANY", y = "gpu.matrix.torch"), function(x,y, ...){
  if (is.null(y)) {
    return(t(x) %*% x)
  }else{
    castMatrix <- castTypeOperations_torch(x,y, todense = FALSE)
    x <- castMatrix[[1]]
    y <- castMatrix[[2]]
    return(t(x) %*% y)
  }
} )

setMethod("tcrossprod", signature(x = "gpu.matrix.torch", y = "ANY"), function(x,y, ...){

  if (is.null(y)) {
    return(x %*% t(x))
  }else{
    castMatrix <- castTypeOperations_torch(x,y, todense = FALSE)
    x <- castMatrix[[1]]
    y <- castMatrix[[2]]
    return(x %*% t(y))
  }

} )

setMethod("tcrossprod", signature(x = "gpu.matrix.torch", y = "missing"), function(x,y, ...){

  return(x %*% t(x))
} )

setMethod("tcrossprod", signature(x = "ANY", y = "gpu.matrix.torch"), function(x,y, ...){
  if (is.null(y)) {
    return(x %*% t(x))
  }else{
    castMatrix <- castTypeOperations_torch(x,y, todense = FALSE)
    x <- castMatrix[[1]]
    y <- castMatrix[[2]]
    return(x %*% t(y))
  }
} )

tf_kron_torch <- function(X,Y){
  castMatrix <- castTypeOperations_torch(X,Y,todense=T)
  X <- castMatrix[[1]]
  Y <- castMatrix[[2]]
  X@gm<- X@gm$contiguous()
  Y@gm<- Y@gm$contiguous()

  res <- torch::torch_kron(X@gm,Y@gm)
  return(gpu.matrix.torch(res,device=device(X)))
}
setMethod("%x%", signature(X = "gpu.matrix.torch", Y = "ANY"), function(X,Y){
  return(tf_kron_torch(X, Y))
})
setMethod("%x%", signature(X = "ANY", Y = "gpu.matrix.torch"), function(X,Y){
  return(tf_kron_torch(X, Y))
})

# setGeneric("%^%", function(x,k) standardGeneric("%^%"))
setMethod("%^%", signature(x = "gpu.matrix.torch", k = "numeric"), function(x,k){
  if (k < 0) stop("power must be a positive integer; use solve() directly for negative powers")
  res <- x
  i <- 1
  while (i < k) {
    res <- res %*% x
    i = i+1
  }
  return(res)
})
setGeneric("expmGPU", function(x) standardGeneric("expmGPU"))
setMethod("expmGPU", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  x <- warningInteger(x)
  x@gm <- torch::torch_matrix_exp(x@gm)
  # message("The exponential is computed using a combination of the scaling and squaring method and the Pade approximation.SIAM J. Matrix Anal. Applic., 26:1179-1193, 2005")
  return(x)
})

setMethod("diag", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  res <- as.numeric(x@gm$diag()$cpu())

  return(res)
})
setMethod("diag<-", signature(x = "gpu.matrix.torch", value = "numeric"), function(x,value){
  if (x@sparse) {
    x <- warningSparseTensor_torch(x)
    x@gm$fill_diagonal_(0)
    value <-torch::torch_diag_embed(value)
    if(device(x) == "cuda") value <- value$cuda()
    x@gm <- x@gm + value
    x@gm <- (to_sparse_torch(x@gm))@gm
    x@sparse <- T
  }else{
    x@gm$fill_diagonal_(0)
    value <-torch::torch_diag_embed(value)
    if(device(x) == "cuda") value <- value$cuda()
    x@gm <- x@gm + value
  }

  return(x)
})

setMethod("solve", signature(a = "gpu.matrix.torch", b = "missing"), function(a){
  a <- warningSparseTensor_torch(a)
  a <- warningInteger(a)
  a@gm <- a@gm$inverse()

  return(a)
})
setMethod("solve", signature(a = "gpu.matrix.torch", b = "ANY"), function(a, b){
  a <- warningInteger(a)
  castMatrix <- castTypeOperations_torch(a,b,sameType = T)
  a <- castMatrix[[1]]
  b <- castMatrix[[2]]
  des_lu <- torch::torch_lu(a@gm,pivot = T)
  LU <- des_lu[[1]]
  pivots <- des_lu[[2]]
  res <- torch::torch_lu_solve(b@gm, LU, pivots)
  res <- gpu.matrix.torch(res, device=device(a))

  return(res)
})
setMethod("solve", signature(a = "ANY", b = "gpu.matrix.torch"), function(a, b){
  b <- warningInteger(b)
  castMatrix <- castTypeOperations_torch(a,b,sameType = T)
  a <- castMatrix[[1]]
  b <- castMatrix[[2]]
  des_lu <- torch::torch_lu(a@gm,pivot = T)
  LU <- des_lu[[1]]
  pivots <- des_lu[[2]]
  res <- torch::torch_lu_solve(b@gm, LU, pivots)
  res <- gpu.matrix.torch(res,device=device(b))

  return(res)
})

setMethod("qr", signature(x="gpu.matrix.torch"), function(x,...){
  x <- warningSparseTensor_torch(x)
  qrTorch <- torch::linalg_qr(x@gm,mode = "complete")
  res <- list(q=gpu.matrix.torch(qrTorch[[1]], device=device(x)), r=gpu.matrix.torch(qrTorch[[2]],device=device(x)), x=x)
  return(res)
})

setMethod("qr.Q", signature(qr="list"), function(qr, complete=F,Dvec){
  objectClass <- class(qr[[1]])[[1]]
  if((objectClass == "gpu.matrix.torch") | (objectClass == "gpu.matrix.tensorflow")){
    if(complete){
      res <- qr$q
    }else{
      res <- qr$q[,1:min(dim(qr$x))]
    }
    if (missing(Dvec)){
      Dvec <- diag(rep(1,ncol(res)))
    }else{
      Dvec <- diag(Dvec,nrow = ncol(res),ncol = ncol(res))
    }

    res <- res %*% Dvec

  }else{
    res <- base::qr.Q(qr,complete,Dvec)
  }
  return(res)
})

utils::globalVariables(c("R"))
setMethod("qr.X", signature(qr="list"),function(qr, complete=F){
  objectClass <- class(qr[[1]])[[1]]
  if((objectClass == "gpu.matrix.torch") | (objectClass == "gpu.matrix.tensorflow")){
    if(complete){
      if(nrow(qr$x)>ncol(qr$x)){
        toadd <- nrow(qr$x)-ncol(qr$x)
        res <- cbind(qr$x,qr$q[,(nrow(qr$x)-toadd+1):nrow(qr$x)])
      }else{
        res <- qr$x[,1:min(dim(qr$x))]
      }

    }else{
      res <- qr$x[,1:min(dim(qr$x))]
    }

  }else{
    res <- base::qr.X(qr,complete)
  }
  return(res)
})

setMethod("qr.R", signature(qr="list"), function(qr, complete=F){
  objectClass <- class(qr[[1]])[[1]]
  if((objectClass == "gpu.matrix.torch") | (objectClass == "gpu.matrix.tensorflow")){
    if(complete){
      res <- qr$r
    }else{
      res <- qr$r[1:min(dim(qr$x)),]
    }

  }else{
    res <- base::qr.R(qr)
  }
  return(res)
})

setMethod("qr.coef", signature(qr="list", y="ANY"), function(qr, y){
  objectClass <- class(qr[[1]])[[1]]
  if((objectClass == "gpu.matrix.torch") | (objectClass == "gpu.matrix.tensorflow")){
    cf <- qr.solve(a = qr,b=y)
    if(ncol(qr$x) > nrow(qr$x)){
      toadd <- matrix(0,nrow=ncol(qr$x) - nrow(qr$x),ncol=1)
      cf <- rbind(cf,toadd)
    }
    res <- cf

  }else{
    res <- base::qr.coef(qr,y)
  }
  return(res)
})

setMethod("qr.qy", signature(qr="list", y="ANY"), function(qr, y){
  objectClass <- class(qr[[1]])[[1]]
  if((objectClass == "gpu.matrix.torch") | (objectClass == "gpu.matrix.tensorflow")){
    res <- qr$q %*% y
  }else{
    res <- base::qr.qy(qr,y)
  }
  return(res)
})

setMethod("qr.qty", signature(qr="list", y="ANY"), function(qr, y){
  objectClass <- class(qr[[1]])[[1]]
  if((objectClass == "gpu.matrix.torch") | (objectClass == "gpu.matrix.tensorflow")){
    res <- t(qr$q) %*% y
  }else{
    res <- base::qr.qty(qr,y)
  }
  return(res)
})

setMethod("qr.resid", signature(qr="list", y="ANY"), function(qr, y){
  objectClass <- class(qr[[1]])[[1]]
  if((objectClass == "gpu.matrix.torch") | (objectClass == "gpu.matrix.tensorflow")){
    res <- y - (qr$x %*% qr.coef(qr,y))
  }else{
    res <- base::qr.resid(qr,y)
  }
  return(res)
})

setMethod("qr.solve", signature(a="gpu.matrix.torch", b="gpu.matrix.torch"), function(a,b){
  b <- warningInteger(b)
  castMatrix <- castTypeOperations_torch(a,b,sameType = T)
  a <- castMatrix[[1]]
  b <- castMatrix[[2]]
  qr_gpu <- qr(a)

  res <- qr.solve(a=qr_gpu,b=b)
  return(res)
})

setMethod("qr.solve", signature(a="gpu.matrix.torch", b="ANY"), function(a,b){
  b <- warningInteger(b)
  castMatrix <- castTypeOperations_torch(a,b,sameType = T)
  a <- castMatrix[[1]]
  b <- castMatrix[[2]]
  qr_gpu <- qr(a)

  res <- qr.solve(a=qr_gpu,b=b)
  return(res)
})

setMethod("qr.solve", signature(a="ANY", b="gpu.matrix.torch"), function(a,b){
  b <- warningInteger(b)
  castMatrix <- castTypeOperations_torch(a,b,sameType = T)
  a <- castMatrix[[1]]
  b <- castMatrix[[2]]
  qr_gpu <- qr(a)

  res <- qr.solve(a=qr_gpu,b=b)
  return(res)
})


setMethod("qr.solve", signature(a="list", b="ANY"), function(a,b){
  objectClass <- class(a[[1]])[[1]]
  if((objectClass == "gpu.matrix.torch")){
    castMatrix <- castTypeOperations_torch(a[[1]], b, sameType = T)
    b <- castMatrix[[2]]

    qr_gpu <- a

    res_solve <- torch::torch_triangular_solve((t(qr.Q(qr_gpu)) %*% b)@gm,qr.R(qr_gpu)@gm[,1:min(dim(qr_gpu$x))])[[1]]
    res <- gpu.matrix.torch(res_solve, dtype = dtype(a[[1]]))
  }else if((objectClass == "gpu.matrix.tensorflow")){
    castMatrix <- castTypeOperations(a[[1]], b, sameType = T)
    b <- castMatrix[[2]]

    qr_gpu <- a
    res_solve <- tensorflow::tf$linalg$triangular_solve(qr.R(qr_gpu)@gm[,1:min(dim(qr_gpu$x))], (t(qr.Q(qr_gpu)) %*% b)@gm, lower = F)
    res <- gpu.matrix.tensorflow(res_solve)
  }else{
    res <- base::qr.solve(a,b)
  }

  return(res)
})




setMethod("eigen", signature(x="gpu.matrix.torch"), function(x){

  x <- warningSparseTensor_torch(x)

  res <- torch::linalg_eig(x@gm)
  values <- gpu.matrix(res[[1]], device = device(x))
  vectors <- gpu.matrix(res[[2]], device = device(x))
  res <- list("values"=values, "vectors"=vectors)

  return(res)
})


setMethod("svd", signature(x="gpu.matrix.torch"), function(x){

  x <- warningSparseTensor_torch(x)
  res <- torch::torch_svd(x@gm)
  res <- list("d"=gpu.matrix.torch(res[[2]]), "u"=gpu.matrix.torch(res[[1]]), "v"=gpu.matrix.torch(res[[3]]), device=device(x))

  return(res)
})

setMethod("ginv", signature(X="gpu.matrix.torch", tol="ANY"), function (X, tol = sqrt(.Machine$double.eps))
{
  X <- warningSparseTensor_torch(X)
  X@gm <- torch::torch_pinverse(X@gm)
  return(X)

})

setMethod("chol", signature(x="gpu.matrix.torch"), function(x){

  x <- warningSparseTensor_torch(x)
  res <- gpu.matrix.torch(torch::torch_cholesky(x@gm,upper = T), device=device(x))

  return(res)
})
# setGeneric("chol_solve", function(x,y) standardGeneric("chol_solve"))

setMethod("chol_solve", signature(x="gpu.matrix.torch", y="ANY"), function(x, y){
  x <- warningInteger(x)
  castMatrix <- castTypeOperations_torch(x,y,sameType = T)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  res <- gpu.matrix.torch(torch::torch_cholesky_solve(y@gm,x@gm), device=device(x))
  return(res)
})

setMethod("chol_solve", signature(x="ANY", y="gpu.matrix.torch"), function(x, y){
  y <- warningInteger(y)
  castMatrix <- castTypeOperations_torch(x,y,sameType = T)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  res <- gpu.matrix.torch(torch::torch_cholesky_solve(y@gm,x@gm), device=device(y))
  return(res)
})




setMethod("mean", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningInteger(x)
  if (x@sparse) {
    res <- x@gm$values()$sum()/length(x)
    res <- as.numeric(res$cpu())
  }else{
    res <- as.numeric(x@gm$mean()$cpu())
  }

  return(res)
})

setMethod("density", signature(x = "gpu.matrix.torch"), function(x){
  return(density(as.numeric(x)))
})

setMethod("hist", signature(x = "gpu.matrix.torch"), function(x,...){
  xmat <- as.numeric(x)
  return(hist(xmat,...))
})

setMethod("colMeans", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningInteger(x)
  if(x@sparse){
    reduced_sum = colSums(x)  # Sum of each row
    reduced_mean = reduced_sum / ncol(x)  # Mean of each row
    res <- as.vector(reduced_mean)
  }else{
    res <- as.numeric(torch::torch_mean(x@gm,1)$cpu())
  }
  names(res) <- colnames(x)

  return(res)
})

setMethod("rowMeans", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningInteger(x)
  if(x@sparse){
    reduced_sum = rowSums(x)  # Sum of each row
    reduced_mean = reduced_sum / nrow(x)  # Mean of each row
    res <- as.vector(reduced_mean)
  }else{
    res <- as.numeric(torch::torch_mean(x@gm,2)$cpu())
  }
  names(res) <- rownames(x)

  return(res)
})


setMethod("sum", signature(x = "gpu.matrix.torch"), function(x){
  if (x@sparse) {
    x@gm <- torch::torch_sum(x@gm$values())
  }else{
    x@gm <- x@gm$sum()
  }
  dimnames(x) <- c(NULL,NULL)
  return(x)
})

# setGeneric("dtype", function(x) standardGeneric("dtype"))

writeDType_torch <- function(dtype){
  dtype <- as.character(dtype)
  switch(dtype,
         "Float" = {
           res <- "float32"
         },
         "Double" = {
           res <- "float64"
         },
         "Int" = {
           res <- "int"
         },
         "Bool" = {
           res <- "bool"
         },
         "ComplexDouble"={
           res <- "complex64"
         },
         "ComplexFloat"={
           res <- "complex32"
         },
         "Long"={
           res <- "long"
         },
         stop("Invalid input type")
  )
  return(res)
}
setMethod("dtype", signature(x = "gpu.matrix.torch"), function(x){
  res <- x@gm$dtype
  return(writeDType_torch(res))
})

setMethod("dtype<-", signature(x = "gpu.matrix.torch", value="ANY"), function(x,value){
  if (is.character(value)) value <- castDtype_torch(value)
  x@gm <- x@gm$to(value)
  return(x)
})

setMethod("min", signature(x = "gpu.matrix.torch"), function(x){
  if(x@sparse){
    res <- as.numeric(torch::torch_min(x@gm$values())$cpu())
  } else{
    res <- as.numeric(torch::torch_min(x@gm)$cpu())
  }
  return(res)
})

setMethod("max", signature(x = "gpu.matrix.torch"), function(x){
  if(x@sparse){
    res <- as.numeric(torch::torch_max(x@gm$values())$cpu())
  } else{
    res <- as.numeric(torch::torch_max(x@gm)$cpu())
  }
  return(res)
})

setMethod("which.max", signature(x = "gpu.matrix.torch"), function(x){

  if (x@sparse) {
    x <- warningSparseTensor_torch(x)
  }
  vecSearch <- t(x)@gm$reshape(length(x))
  max_index <- vecSearch$max(dim=1)[[2]]

  res <- as.numeric(max_index$cpu())

  return(res)
})

setMethod("which.min", signature(x = "gpu.matrix.torch"), function(x){

  x <- warningSparseTensor_torch(x)

  vecSearch <- t(x)@gm$reshape(length(x))
  min_index <- vecSearch$min(dim=1)[[2]]

  res <- as.numeric(min_index$cpu())

  return(res)
})


# Se debe merjorar
applyTest <- function (X, MARGIN, FUN, ..., simplify = TRUE)
{
  FUN <- match.fun(FUN)
  simplify <- isTRUE(simplify)
  dl <- length(dim(X))
  if (!dl)
    stop("dim(X) must have a positive length")

  d <- dim(X)
  dn <- dimnames(X)
  ds <- seq_len(dl)
  if (is.character(MARGIN)) {
    if (is.null(dnn <- names(dn)))
      stop("'X' must have named dimnames")
    MARGIN <- match(MARGIN, dnn)
    if (anyNA(MARGIN))
      stop("not all elements of 'MARGIN' are names of dimensions")
  }
  d.call <- d[-MARGIN]
  d.ans <- d[MARGIN]
  if (anyNA(d.call) || anyNA(d.ans))
    stop("'MARGIN' does not match dim(X)")
  s.call <- ds[-MARGIN]
  s.ans <- ds[MARGIN]
  dn.call <- dn[-MARGIN]
  dn.ans <- dn[MARGIN]
  d2 <- prod(d.ans)
  if (d2 == 0L) {
    newX <- array(vector(typeof(X), 1L), dim = c(prod(d.call),
                                                 1L))
    ans <- forceAndCall(1, FUN, if (length(d.call) < 2L) newX[,
                                                              1] else array(newX[, 1L], d.call, dn.call), ...)
    return(if (is.null(ans)) ans else if (length(d.ans) <
                                          2L) ans[1L][-1L] else array(ans, d.ans, dn.ans))
  }
  if(MARGIN==1){
    newX <- t(X)
  }else if(MARGIN==2){
    newX <- X
  }

  dim(newX) <- c(prod(d.call), d2)
  ans <- vector("list", d2)
  if (length(d.call) < 2L) {
    if (length(dn.call))
      dimnames(newX) <- c(dn.call, list(NULL))
    for (i in 1L:d2) {
      tmp <- forceAndCall(1, FUN, newX[, i], ...)
      if (!is.null(tmp))
        ans[[i]] <- tmp
    }
  }else for (i in 1L:d2) {
    tmp <- forceAndCall(1, FUN, array(newX[, i], d.call,
                                      dn.call), ...)
    if (!is.null(tmp))
      ans[[i]] <- tmp
  }
  ans.list <- !simplify || is.recursive(ans[[1L]])
  l.ans <- length(ans[[1L]])
  ans.names <- names(ans[[1L]])
  if (!ans.list)
    ans.list <- any(lengths(ans) != l.ans)
  if (!ans.list && length(ans.names)) {
    all.same <- vapply(ans, function(x) identical(names(x),
                                                  ans.names), NA)
    if (!all(all.same))
      ans.names <- NULL
  }
  len.a <- if (ans.list)
    d2
  else length(ans <- unlist(ans, recursive = FALSE))
  if (length(MARGIN) == 1L && len.a == d2) {
    names(ans) <- if (length(dn.ans[[1L]]))
      dn.ans[[1L]]
    ans
  }
  else if (len.a == d2)
    array(ans, d.ans, dn.ans)
  else if (len.a && len.a%%d2 == 0L) {
    if (is.null(dn.ans))
      dn.ans <- vector(mode = "list", length(d.ans))
    dn1 <- list(ans.names)
    if (length(dn.call) && !is.null(n1 <- names(dn <- dn.call[1])) &&
        nzchar(n1) && length(ans.names) == length(dn[[1]]))
      names(dn1) <- n1
    dn.ans <- c(dn1, dn.ans)
    array(ans, c(len.a%/%d2, d.ans), if (!is.null(names(dn.ans)) ||
                                         !all(vapply(dn.ans, is.null, NA)))
      dn.ans)
  }
  else ans
}

setMethod("apply", signature(X="gpu.matrix.torch"), function(X, MARGIN, FUN, ..., simplify = TRUE){
  warning("If the function applied to the GPU matrix returns a tensor or another GPU matrix, then the 'simplify' argument will always be FALSE.")
  applyTest(X, MARGIN, FUN, ..., simplify = simplify)

})


setMethod("cov", signature(x = "gpu.matrix.torch", y = "ANY"), function(x,y){
  x <- warningInteger(x)
  castMatrix <- castTypeOperations_torch(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  x_ <- t(x) - colMeans(x)
  y_ <- t(y) - colMeans(y)

  res <- tcrossprod(x_, y_)/(ncol(x_)-1)
  return(res)
})

setMethod("cov", signature(x = "ANY", y = "gpu.matrix.torch"), function(x,y){
  y <- warningInteger(y)
  castMatrix <- castTypeOperations_torch(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  x_ <- t(x) - colMeans(x)
  y_ <- t(y) - colMeans(y)

  res <- tcrossprod(x_, y_)/(ncol(x_)-1)
  return(res)
})

setMethod("cov", signature(x = "gpu.matrix.torch", y = "missing"), function(x,y){

  x <- warningSparseTensor_torch(x)
  x <- warningInteger(x)
  x_ <- t(x) - colMeans(x)
  res <- tcrossprod(x_)/(ncol(x_)-1)

  return(res)
})

setMethod("cov2cor", signature(V="gpu.matrix.torch"), function(V){
  V <- warningInteger(V)
  V <- warningSparseTensor_torch(V)

  p <- (d <- dim(V))[1L]
  Is <- sqrt(1/diag(V))
  r <- Is * V * rep(Is, each = p)
  r[cbind(1L:p, 1L:p)] <- 1
  dimnames(r) <- dimnames(V)
  return(r)
})

normalizeGPU <- function(x) {
  xcenter <- t(t(x)- colMeans(x))
  xnorm <- t(t(xcenter)/sqrt(colVars(xcenter)))
  return(xnorm)
}



setMethod("cor", signature(x = "gpu.matrix.torch", y = "ANY"), function(x,y){

  x <- warningSparseTensor_torch(x)
  castMatrix <- castTypeOperations_torch(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  xn <- normalizeGPU(x)
  yn <- normalizeGPU(y)
  res <- cov(xn,yn)

  return(res)
})

setMethod("cor", signature(x = "ANY", y = "gpu.matrix.torch"), function(x,y){

  y <- warningSparseTensor_torch(y)
  castMatrix <- castTypeOperations_torch(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  xn <- normalizeGPU(x)
  yn <- normalizeGPU(y)
  res <- cov(xn,yn)

  return(res)
})


setMethod("cor", signature(x = "gpu.matrix.torch", y = "missing"), function(x,y){

  x <- warningSparseTensor_torch(x)
  V <- cov(x)
  res <- cov2cor(V)
  dimnames(res) <- dimnames(V)
  # dimnames(res) <- dimnames(V)
  return(res)
})

setMethod("cor", signature(x = "gpu.matrix.torch", y = "ANY",use="missing", method = "character"), function(x,y,method){
  x <- warningSparseTensor_torch(x)

  castMatrix <- castTypeOperations_torch(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  if(method=="spearman"){
    x <- gpu.matrix.torch(t(colRanks(x))@gm,dtype = dtype(x), dimnames = dimnames(x), device=device(x))
    y <- gpu.matrix.torch(t(colRanks(y))@gm,dtype = dtype(y), dimnames = dimnames(y), device=device(x))
  }

  xn <- normalizeGPU(x)
  yn <- normalizeGPU(y)
  res <- cov(xn,yn)
  return(res)
})

setMethod("cor", signature(x = "gpu.matrix.torch", y = "missing",use="missing", method = "character"), function(x,y,method){
  x <- warningSparseTensor_torch(x)

  if(method=="spearman"){
    x <- gpu.matrix.torch(t(colRanks(x))@gm,dtype = dtype(x), dimnames = dimnames(x), device=device(x))
  }
  V <- cov(x)
  res <- cov2cor(V)
  dimnames(res) <- dimnames(V)
  return(res)
})


setMethod("rowVars", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningInteger(x)
  x <- warningSparseTensor_torch(x)
  res <- gpu.matrix(torch::torch_var(x@gm,dim = 2), device = device(x))
  rownames(res) <- rownames(x)
  return(res)
})

setMethod("colVars", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningInteger(x)
  x <- warningSparseTensor_torch(x)
  res <- gpu.matrix(torch::torch_var(x@gm,dim = 1), device = device(x))
  colnames(res) <- colnames(x)
  return(res)
})


setMethod("colMaxs", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  res <- as.numeric(torch::torch_max(x@gm,dim=1)[[1]]$cpu())
  names(res) <- colnames(x)
  return(res)
})

setMethod("rowMaxs", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  res <- as.numeric(torch::torch_max(x@gm,dim=2)[[1]]$cpu())
  names(res) <- rownames(x)
  return(res)
})


setMethod("rowRanks", signature(x="gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  return(gpu.matrix.torch(torch::torch_argsort(torch::torch_argsort(x@gm,dim = 2),dim = 2), dtype=dtype(x), device=device(x)))
} )

setMethod("colRanks", signature(x="gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  return(t(gpu.matrix.torch(torch::torch_argsort(torch::torch_argsort(x@gm,dim = 1),dim = 1),dtype=dtype(x), device=device(x))))
} )

setMethod("colMins", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  res <- as.numeric(torch::torch_min(x@gm,dim=1)[[1]]$cpu())
  names(res) <- colnames(x)
  return(res)
})

setMethod("rowMins", signature(x = "gpu.matrix.torch"), function(x){
  x <- warningSparseTensor_torch(x)
  res <- as.numeric(torch::torch_min(x@gm,dim=2)[[1]]$cpu())
  names(res) <- rownames(x)
  return(res)
})

setMethod("dist", signature(x = "gpu.matrix.torch"), function(x,method = "euclidean", diag = FALSE, upper = FALSE, p = 2){
  if (!is.na(pmatch(method, "euclidian")))
    method <- "euclidean"
  METHODS <- c("euclidean", "maximum", "manhattan", "minkowski")
  method <- pmatch(method, METHODS)
  p <- (method == 1)*2 + (method==3)*1+(method==4)*p
  if(method==2) p <- Inf
  if (is.na(method))
    stop("invalid distance method")

  output <- torch::torch_cdist(x@gm, x@gm,p)
  return(gpu.matrix(output, device = device(x)))
})
