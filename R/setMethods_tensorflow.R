

to_dense_tensorflow<-function(x){
  if(x@sparse){
    x@gm <-tensorflow::tf$sparse$to_dense(x@gm)
    x@sparse <- F
  }
  return(x)
}

to_sparse_tensorflow<-function(x){
  if(!x@sparse){
    x@gm <- tensorflow::tf$sparse$from_dense(x@gm)
    x@sparse <- T
  }
  return(x)
}
setClassUnion("numMatrixLike", members = c("logical", "integer", "numeric", "matrix"))
c.GPUmatrix <- function(...) unlist(lapply(list(...), as.vector))

setMethod("c", "gpu.matrix.tensorflow", function(x, ..., recursive) c.GPUmatrix(x, ...))
setMethod("c", "numMatrixLike", function(x, ..., recursive) c.GPUmatrix(x, ...))

setGeneric("to_dense", function(x) standardGeneric("to_dense"))
setMethod("to_dense", signature(x = "gpu.matrix.tensorflow"), function(x) to_dense_tensorflow(x) )
setGeneric("to_sparse", function(x) standardGeneric("to_sparse"))
setMethod("to_sparse", signature(x = "gpu.matrix.tensorflow"), function(x) to_sparse_tensorflow(x) )

logdetTensor <- function(x){
  value <- tensorflow::tf$linalg$slogdet(x@gm)
  logAbDet <- as.numeric(value$log_abs_determinant)
  attr(logAbDet, which = "logarithm") <- TRUE
  sign<-as.numeric(value$sign)
  res <- list("modulus"=logAbDet, "sign"=sign)
  attr(res, which = "class") <- "det"
  return(res)
}


warningInteger <- function(x){
  typeTensor <- dtype(x)
  if (typeTensor == "int"){
    dtype(x) <- "float64"
    warning(message = "Not allowed with int32, parse to float64 by default")
  }
  return(x)
}

setMethod("determinant", signature(x = "gpu.matrix.tensorflow", logarithm = "missing"), function(x, logarithm, ...){
  x <- warningSparseTensor(x)
  x <- warningInteger(x)
  res <- logdetTensor(x)
  return(res)
})
setMethod("determinant", signature(x = "gpu.matrix.tensorflow", logarithm = "logical"), function(x, logarithm, ...){
  x <- warningSparseTensor(x)
  x <- warningInteger(x)
  if (logarithm) {
    res <- logdetTensor(x)
  }else{
    value <- tensorflow::tf$linalg$det(x@gm)
    logAbDet <- as.numeric(value)
    attr(logAbDet, which = "logarithm") <- FALSE
    sign<-sign(as.numeric(logAbDet))
    res <- list("modulus"=abs(logAbDet), "sign"=sign)
    attr(res, which = "class") <- "det"
  }
  return(res)
})

setMethod("det", signature(x = "gpu.matrix.tensorflow"), function(x, ...){
  res <- determinant(x, logarithm = F)

  return(as.numeric(res$modulus))
})

setMethod("fft", signature(z="gpu.matrix.tensorflow"), function(z){
  z <- warningSparseTensor(z)
  res <- gpu.matrix.tensorflow(tensorflow::tf$signal$fft(tensorflow::tf$cast(z@gm,tensorflow::tf$complex128)),dtype=tensorflow::tf$complex128)
  return(res)
})

setMethod("xtfrm", signature(x="gpu.matrix.tensorflow"), function(x){
  return(as.numeric(x))
})

setMethod("sort", signature(x="gpu.matrix.tensorflow", decreasing = "missing"), function(x,decreasing){
  if (x@sparse) {
    res <- as.vector(sort(x@gm$values))
  }else{
    res <- as.vector(tensorflow::tf$sort(tensorflow::tf$reshape(x@gm,length(x))))
  }
  return(res)
})

setMethod("sort", signature(x="gpu.matrix.tensorflow", decreasing = "logical"), function(x,decreasing){
  if(decreasing){
    decreasing="DESCENDING"
  }else{
    decreasing="ASCENDING"
    }
  if (x@sparse) {
    res <- as.vector(tensorflow::tf$sort(x@gm$values, direction = decreasing))
  }else{
    res <- as.vector(tensorflow::tf$sort(tensorflow::tf$reshape(x@gm,length(x)),direction=decreasing))
  }
  return(res)
})

setMethod("round", signature(x= "gpu.matrix.tensorflow",digits="missing"), function(x,digits){
  x<-warningInteger(x)
  if (x@sparse) {
    x@gm <- tensorflow::tf$SparseTensor(indices = x@gm$indices,
                            values = tensorflow::tf$round(x@gm$values),
                            dense_shape = x@gm$shape)
  }else{
    x@gm <- tensorflow::tf$round(x@gm)
  }
  return(x)
})

my_tf_round<- function(x, decimals = 0){
  multiplier = tensorflow::tf$constant(10**decimals, dtype=x$dtype)
  return(tensorflow::tf$round(x * multiplier) / multiplier)
}

setMethod("round", signature(x= "gpu.matrix.tensorflow",digits="numeric"), function(x,digits){
  x<-warningInteger(x)
  if (x@sparse) {
    x@gm <- tensorflow::tf$SparseTensor(indices = x@gm$indices,
                            values = my_tf_round(x@gm$values, digits),
                            dense_shape = x@gm$shape)
  }else{
    x@gm <- my_tf_round(x@gm, digits)
  }
  return(x)
})


setMethod(f = "show", signature = "gpu.matrix.tensorflow", definition = function(object){
  cat("GPUmatrix\n")
  print(object@gm)
  if (!is.null(object@rownames)) cat(paste(c("rownames:",object@rownames,"\n")))
  if (!is.null(object@colnames)) cat(paste(c("colnames:",object@colnames,"\n")))
})

setMethod("length", signature(x = "gpu.matrix.tensorflow"), function(x){
  return(length(x@gm))
} )



setMethod("dim", signature(x = "gpu.matrix.tensorflow"), function(x){dim(x@gm)})
setMethod("dim<-", signature(x = "gpu.matrix.tensorflow",value="vector"), function(x,value){
  x <- t(x)
  if (x@sparse) {
    x@gm <- tensorflow::tf$sparse$reshape(x@gm,as.integer(rev(value)))
  }else{
    x@gm <- tensorflow::tf$reshape(x@gm,as.integer(rev(value)))
  }
  return(t(x))
})


setMethod("dimnames", signature(x = "gpu.matrix.tensorflow"), function(x){
  if (is.null(c(x@rownames,x@colnames))) {
    res <- NULL
  }else{
    res <- list(x@rownames,x@colnames)
  }
  return(res)
})
setMethod("dimnames<-", signature(x = "gpu.matrix.tensorflow", value="vector"), function(x,value){

  if (length(value[[1]]) == nrow(x) & length(value[[2]]) == ncol(x)) {
    x@rownames <- value[[1]]
    x@colnames <- value[[2]]
  }else if (is.null(value[[1]]) & length(value[[2]]) == ncol(x)){
    x@colnames <- value[[2]]
    x@rownames <- c()
  }else if (is.null(value[[2]]) & length(value[[1]]) == nrow(x)){
    x@rownames <- value[[1]]
    x@colnames <- c()
  }else if (is.null(value[[1]]) & is.null(value[[2]])){
    x@rownames <- c()
    x@colnames <- c()
  }else{
    stop("Error dimension not match")
  }

  return(x)
})

setMethod("rownames", signature(x = "gpu.matrix.tensorflow"), function(x){
  return(x@rownames)
})
setMethod("row.names", signature(x = "gpu.matrix.tensorflow"), function(x){
  return(rownames(x))
})
setMethod("rownames<-", signature(x = "gpu.matrix.tensorflow", value="vector"), function(x,value){
  if (length(value) != nrow(x))  stop("length of 'colnames' not equal to array extent")

  if (is.null(value)) value <- c()
  x@rownames <- value
  return(x)
})
setMethod("row.names<-", signature(x = "gpu.matrix.tensorflow", value="vector"), function(x,value){
  return(rownames(x) <- value)
})
setMethod("colnames", signature(x = "gpu.matrix.tensorflow"), function(x){
  return(x@colnames)
})
setMethod("colnames<-", signature(x = "gpu.matrix.tensorflow", value="vector"), function(x,value){
  if (length(value) != ncol(x))  stop("length of 'colnames' not equal to array extent")
  if (is.null(value)) value <- c()
  x@colnames <- value
  return(x)
})

setMethod("rowSums", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  return(as.vector(tensorflow::tf$math$reduce_sum(x@gm, 1L)))
})
setMethod("colSums", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  return(as.vector(tensorflow::tf$math$reduce_sum(x@gm, 0L)))
})


setMethod("cbind2",signature(x = "gpu.matrix.tensorflow", y = "ANY"), function(x,y,...){

  castMatrix <- castTypeOperations(x,y, todense=F, sameType = T)
  yOrigin <- y
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  if (x@sparse & y@sparse) {
    res <- gpu.matrix.tensorflow(tensorflow::tf$sparse$concat(sp_inputs = list(x@gm,y@gm), axis = 1L))
  }else{
    if (x@sparse) x <- warningSparseTensor(x)
    if (y@sparse) y <- warningSparseTensor(y)
    res <- gpu.matrix.tensorflow(cbind(x@gm,y@gm))
  }

  if (is.null(colnames(x)) & !is.null(colnames(y))) colnames(x) <- rep(NA,ncol(x))
  if (is.null(colnames(y)) & !is.null(colnames(x)) & !is.vector(yOrigin)) colnames(y) <- rep(NA,ncol(y))
  if (is.null(colnames(y)) & !is.null(colnames(x)) & is.vector(yOrigin)) y@colnames <- NA
  # rNames <- c(rownames(x),rownames(y))[c(1:nrow(res))]
  rNames <- NULL
  if (!is.null(rownames(x))) rNames <- rownames(x)
  if (is.null(rownames(x)) & !is.null(rownames(y))) rNames <- rownames(y)
  if (!is.null(rownames(x)) & !is.null(rownames(y))) rNames <- c(rownames(x),rownames(y))[c(1:nrow(x))]
  # dimnames(res) <- list(rNames,c(colnames(x), colnames(y)))

  dimnames(res) <- list(rNames,c(colnames(x), colnames(y)))


  return(res)
})

setMethod("cbind2",signature(x = "ANY", y = "gpu.matrix.tensorflow"), function(x,y){

  castMatrix <- castTypeOperations(x,y, todense=F, sameType = T)
  xOrigin <- x
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]


  if (x@sparse & y@sparse) {
    res <- gpu.matrix.tensorflow(tensorflow::tf$sparse$concat(sp_inputs = list(x@gm,y@gm), axis = 1L))
  }else{
    if (x@sparse) x <- warningSparseTensor(x)
    if (y@sparse) y <- warningSparseTensor(y)
    res <- gpu.matrix.tensorflow(cbind(x@gm,y@gm))
  }

  if (is.null(colnames(x)) & !is.null(colnames(y)) & !is.vector(xOrigin)) colnames(x) <- rep(NA,ncol(x))
  if (is.null(colnames(x)) & !is.null(colnames(y)) & is.vector(xOrigin)) x@colnames <- NA
  if (is.null(colnames(y)) & !is.null(colnames(x))) colnames(y) <- rep(NA,ncol(y))
  # rNames <- c(rownames(x),rownames(y))[c(1:nrow(res))]
  rNames <- NULL
  if (!is.null(rownames(x))) rNames <- rownames(x)
  if (is.null(rownames(x)) & !is.null(rownames(y))) rNames <- rownames(y)
  if (!is.null(rownames(x)) & !is.null(rownames(y))) rNames <- c(rownames(x),rownames(y))[c(1:nrow(x))]
  # dimnames(res) <- list(rNames,c(colnames(x), colnames(y)))

  dimnames(res) <- list(rNames,c(colnames(x), colnames(y)))


  return(res)
})

setMethod("rbind2", signature(x = "gpu.matrix.tensorflow", y = "ANY"), function(x,y){
  castMatrix <- castTypeOperations(x,y, todense=F, sameType = T)
  yOrigin <- y
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  if (x@sparse & y@sparse) {
    res <- gpu.matrix.tensorflow(tensorflow::tf$sparse$concat(sp_inputs = list(x@gm,y@gm), axis = 0L))
  }else{
    if (x@sparse) x <- warningSparseTensor(x)
    if (y@sparse) y <- warningSparseTensor(y)
    res <- gpu.matrix.tensorflow(rbind(x@gm,y@gm))
  }


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

setMethod("rbind2",signature(x = "ANY", y = "gpu.matrix.tensorflow"), function(x,y){
  castMatrix <- castTypeOperations(x,y, todense=F, sameType = T)
  xOrigin <- x
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  if (x@sparse & y@sparse) {
    res <- gpu.matrix.tensorflow(tensorflow::tf$sparse$concat(sp_inputs = list(x@gm,y@gm), axis = 0L))
  }else{
    if (x@sparse) x <- warningSparseTensor(x)
    if (y@sparse) y <- warningSparseTensor(y)
    res <- gpu.matrix.tensorflow(rbind(x@gm,y@gm))
  }


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



setMethod("head", signature(x = "gpu.matrix.tensorflow"), function(x, ...){
  x <- warningSparseTensor(x)
  head(x@gm,...)
  })


setMethod("tail", signature(x = "gpu.matrix.tensorflow"), function(x, ...){
  x <- warningSparseTensor(x)
  tail(x@gm,...)
  })

setMethod("nrow", signature(x = "gpu.matrix.tensorflow"), function(x){
  return(nrow(x@gm))
} )

setMethod("ncol", signature(x = "gpu.matrix.tensorflow"), function(x){
  return(ncol(x@gm))
} )

setMethod("t", signature(x = "gpu.matrix.tensorflow"), function(x){
  if (x@sparse) {
    res <- gpu.matrix.tensorflow(data = tensorflow::tf$sparse$transpose(x@gm),rownames = colnames(x),colnames = rownames(x), dtype=dtype(x))
  }else{
    res <- gpu.matrix.tensorflow(tensorflow::tf$transpose(x@gm),rownames = colnames(x),colnames = rownames(x), dtype=dtype(x))
  }
  return(res)
})

setMethod("crossprod", signature(x = "gpu.matrix.tensorflow", y = "ANY"), function(x,y, ...){
    castMatrix <- castTypeOperations(x,y, todense=FALSE)
    x <- castMatrix[[1]]
    y <- castMatrix[[2]]
    return(t(x) %*% y)
} )

setMethod("crossprod", signature(x = "ANY", y = "gpu.matrix.tensorflow"), function(x,y, ...){
  castMatrix <- castTypeOperations(x,y, todense = FALSE)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  return(t(x) %*% y)
} )

setMethod("crossprod", signature(x = "gpu.matrix.tensorflow", y = "missing"), function(x,y, ...){
    return(t(x) %*% x)
} )

setMethod("tcrossprod", signature(x = "gpu.matrix.tensorflow", y = "ANY"), function(x,y, ...){
    castMatrix <- castTypeOperations(x,y, todense = FALSE)
    x <- castMatrix[[1]]
    y <- castMatrix[[2]]
    return(x %*% t(y))
} )

setMethod("tcrossprod", signature(x = "ANY", y = "gpu.matrix.tensorflow"), function(x,y, ...){
    castMatrix <- castTypeOperations(x,y, todense = FALSE)
    x <- castMatrix[[1]]
    y <- castMatrix[[2]]
    return(x %*% t(y))
} )

setMethod("tcrossprod", signature(x = "gpu.matrix.tensorflow", y = "missing"), function(x,y, ...){

    return(x %*% t(x))
} )



# setMethod("outer", signature(X = "gpu.matrix.tensorflow", Y = "ANY"), function(X,Y, ...){
#
#   castMatrix <- castTypeOperations(X,Y)
#   X <- castMatrix[[1]]
#   Y <- castMatrix[[2]]
#
#   return(as.array(tensorflow::tf$tensordot(X@gm, Y@gm, axes=0L)))
#
# } )
# setMethod("outer", signature(X = "ANY", Y = "gpu.matrix.tensorflow"), function(X,Y, ...){
#
#   castMatrix <- castTypeOperations(X,Y)
#   X <- castMatrix[[1]]
#   Y <- castMatrix[[2]]
#
#   return(as.array(tensorflow::tf$tensordot(X@gm, Y@gm, axes=0L)))
#
# } )
#
# setMethod("%o%", signature(X = "gpu.matrix.tensorflow", Y = "ANY"), function(X,Y){
#   return(outer(X,Y))
# })
# setMethod("%o%", signature(X = "ANY", Y = "gpu.matrix.tensorflow"), function(X,Y){
#   return(outer(X,Y))
# })

tf_kron <- function(X,Y){
  castMatrix <- castTypeOperations(X,Y, sameType = T)
  X <- castMatrix[[1]]
  Y <- castMatrix[[2]]

  a <- X@gm
  b <- Y@gm

  a_shape = c(as.integer(a$shape[1]),as.integer(a$shape[2]))
  b_shape = c(as.integer(b$shape[1]),as.integer(b$shape[2]))
  res <- tensorflow::tf$reshape(tensorflow::tf$reshape(a,c(a_shape[1],1L,a_shape[2],1L))*tensorflow::tf$reshape(b,c(1L,b_shape[1],1L,b_shape[2])),c(a_shape[1]*b_shape[1],a_shape[2]*b_shape[2]))
  return(gpu.matrix.tensorflow(res))
}
setMethod("%x%", signature(X = "gpu.matrix.tensorflow", Y = "ANY"), function(X,Y){
  return(tf_kron(X, Y))
})
setMethod("%x%", signature(X = "ANY", Y = "gpu.matrix.tensorflow"), function(X,Y){
  return(tf_kron(X, Y))
})

setGeneric("%^%", function(x,k) standardGeneric("%^%"))
setMethod("%^%", signature(x = "gpu.matrix.tensorflow", k = "numeric"), function(x,k){
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
setMethod("expmGPU", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  x@gm <- tensorflow::tf$linalg$expm(x@gm)
  message("The exponential is computed using a combination of the scaling and squaring method and the Pade approximation.SIAM J. Matrix Anal. Applic., 26:1179-1193, 2005")
  return(x)
})

setMethod("diag", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  res <- as.vector(tensorflow::tf$linalg$diag_part(x@gm))

  return(res)
})
setMethod("diag<-", signature(x = "gpu.matrix.tensorflow", value = "numeric"), function(x,value){
  x<-warningInteger(x)
  if (x@sparse) {
    x <- warningSparseTensor(x)
    x@gm <- tensorflow::tf$linalg$set_diag(x@gm, value)
    x@gm <- tensorflow::tf$sparse$from_dense(x@gm)
    x@sparse <- T
  }else{
    x <- warningSparseTensor(x)
    x@gm <- tensorflow::tf$linalg$set_diag(x@gm, value)
  }

  return(x)
})

setMethod("solve", signature(a = "gpu.matrix.tensorflow", b = "missing"), function(a){
  a <- warningSparseTensor(a)
  a <- warningInteger(a)
  res <- gpu.matrix.tensorflow(tensorflow::tf$linalg$inv(a@gm), dimnames = list(colnames(a),rownames(a)))

  return(res)
})
setMethod("solve", signature(a = "gpu.matrix.tensorflow", b = "ANY"), function(a, b){
  a <- warningSparseTensor(a)
  a <- warningInteger(a)
  castMatrix <- castTypeOperations(a,b,sameType = T)
  a <- castMatrix[[1]]
  b <- castMatrix[[2]]

  res <- gpu.matrix.tensorflow(tensorflow::tf$linalg$solve(a@gm, b@gm))

  return(res)
})
setMethod("solve", signature(a = "ANY", b = "gpu.matrix.tensorflow"), function(a, b){
  b <- warningSparseTensor(b)
  b <- warningInteger(b)
  castMatrix <- castTypeOperations(a,b,sameType = T)
  a <- castMatrix[[1]]
  b <- castMatrix[[2]]

  res <- gpu.matrix.tensorflow(tensorflow::tf$linalg$solve(a@gm, b@gm))

  return(res)
})

setMethod("qr", signature(x="gpu.matrix.tensorflow"), function(x){

  x <- warningSparseTensor(x)
  res <- qr(x@gm)
  res$qr <- gpu.matrix.tensorflow(res$qr)

  return(res)
})

# setGeneric("rankMatrix", function(x) standardGeneric("rankMatrix"))
# setMethod("rankMatrix", signature(x="gpu.matrix.tensorflow"), function(x){
#   x <- warningSparseTensor(x)
#   res <- Matrix::rankMatrix(x@gm)
#   return(res)
# })

#Se debe mejorar
setMethod("eigen", signature(x="gpu.matrix.tensorflow"), function(x){

  x <- warningSparseTensor(x)
  res <- eigen(x@gm)

  return(res)
})

setMethod("svd", signature(x="gpu.matrix.tensorflow"), function(x){

  x <- warningSparseTensor(x)
  res <- tensorflow::tf$linalg$svd(x@gm)
  res <- list("d"=gpu.matrix.tensorflow(res[[1]]), "u"=gpu.matrix.tensorflow(res[[2]]), "v"=gpu.matrix.tensorflow(res[[3]]))

  return(res)
})
setGeneric("ginv", function(X,tol) standardGeneric("ginv"))
setMethod("ginv", signature(X="gpu.matrix.tensorflow", tol="ANY"), function (X, tol = sqrt(.Machine$double.eps))
{
  X <- warningSparseTensor(X)
  X@gm <- tensorflow::tf$linalg$pinv(X@gm)
  return(X)

})

setMethod("chol", signature(x="gpu.matrix.tensorflow"), function(x){

  x <- warningSparseTensor(x)
  res <- gpu.matrix.tensorflow(tensorflow::tf$linalg$cholesky(x@gm))

  return(res)
})
setGeneric("chol_solve", function(x,y) standardGeneric("chol_solve"))

setMethod("chol_solve", signature(x="gpu.matrix.tensorflow", y="ANY"), function(x, y){
  x <- warningInteger(x)
  castMatrix <- castTypeOperations(x,y,sameType = T)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  res <- gpu.matrix.tensorflow(tensorflow::tf$linalg$cholesky_solve(x@gm,y@gm))
  return(res)
})

setMethod("chol_solve", signature(x="ANY", y="gpu.matrix.tensorflow"), function(x, y){
  y <- warningInteger(y)
  castMatrix <- castTypeOperations(x,y,sameType = T)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]

  res <- gpu.matrix.tensorflow(tensorflow::tf$linalg$cholesky_solve(x@gm,y@gm))
  return(res)
})

setMethod("mean", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningInteger(x)
  if (x@sparse) {
    res <- as.numeric(tensorflow::tf$sparse$reduce_sum(x@gm)/length(x))
  }else{
    res <- as.numeric(tensorflow::tf$reduce_mean(x@gm))
  }

  return(res)
})
setMethod("density", signature(x = "gpu.matrix.tensorflow"), function(x){
  return(density(as.numeric(x)))
})
setMethod("hist", signature(x = "gpu.matrix.tensorflow"), function(x,...){
  xmat <- as.numeric(x)
  return(hist(xmat,...))
})
setMethod("colMeans", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningInteger(x)
  if(x@sparse){
    reduced_sum = tensorflow::tf$sparse$reduce_sum(x@gm, 0L)  # Sum of each row
    reduced_mean = reduced_sum / tensorflow::tf$cast(x@gm$dense_shape[2], castDtype_tensorflow(dtype(x)))  # Mean of each row
    res <- as.vector(reduced_mean)
  }else{
    res <- as.vector(tensorflow::tf$reduce_mean(x@gm,axis=0L))
  }
  names(res) <- colnames(x)

  return(res)
})
setMethod("rowMeans", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningInteger(x)
  if(x@sparse){
    reduced_sum = tensorflow::tf$sparse$reduce_sum(x@gm, 1L)  # Sum of each row
    reduced_mean = reduced_sum / tensorflow::tf$cast(x@gm$dense_shape[1], castDtype_tensorflow(dtype(x)))  # Mean of each row
    res <- as.vector(reduced_mean)
  }else{
    res <- as.vector(tensorflow::tf$reduce_mean(x@gm,axis=1L))
  }
  names(res) <- rownames(x)

  return(res)
})
setMethod("sum", signature(x = "gpu.matrix.tensorflow"), function(x){
  if (x@sparse) {
    res <- as.numeric(tensorflow::tf$sparse$reduce_sum(x@gm))
  }else{
    res <- as.numeric(tensorflow::tf$reduce_sum(x@gm))
  }
  return(res)
})

writeDType_tensorflow <- function(dtype){
  dtype <- as.character(dtype)
  switch(dtype,
         "<dtype: 'float32'>" = {
           res <- "float32"
         },
         "<dtype: 'float64'>" = {
           res <- "float64"
         },
         "<dtype: 'int32'>" = {
           res <- "int"
         },
         "<dtype: 'int64'>" = {
           res <- "int"
         },
         "<dtype: 'bool'>" = {
           res <- "bool"
         },
         "<dtype: 'complex64'>"={
           res <- "complex64"
         },
         "<dtype: 'complex128'>"={
           res <- "complex128"
         },
         stop("Invalid input type")
  )
  return(res)
}

setGeneric("dtype", function(x) standardGeneric("dtype"))

setMethod("dtype", signature(x = "gpu.matrix.tensorflow"), function(x){
  res <- writeDType_tensorflow(x@gm$dtype)
  return(res)
})

setGeneric("dtype<-", function(x,value) standardGeneric("dtype<-"))
setMethod("dtype<-", signature(x = "gpu.matrix.tensorflow", value="ANY"), function(x,value){
  if (is.character(value)) value <- castDtype_tensorflow(value)
  x@gm <- tensorflow::tf$cast(x@gm,value)
  return(x)
})

# setGeneric("checkGPU", function() standardGeneric("checkGPU"))
# setMethod("checkGPU", function(){
#   if (length(tensorflow::tf$config$list_physical_devices("GPU")) > 0){
#     cat("Tensorflow dependence is installed using GPU")
#   }else{
#     cat("Tensorflow dependence is not installed using GPU")
#   }
# })

# setMethod("colSums", signature(x = "gpu.matrix.tensorflow"), function(x){
#   print("hola tensor")
#   if (x@sparse) {
#     res <- as.numeric(tensorflow::tf$sparse$reduce_sum(x@gm, 0L))
#   }else{
#     res <- as.vector(tensorflow::tf$reduce_sum(x@gm,axis=0L))
#   }
#   names(res) <- colnames(x)
#   return(res)
#
#   return(res)
# })
# setMethod("rowSums", signature(x = "gpu.matrix.tensorflow"), function(x){
#   print("hola tensor")
#   if (x@sparse) {
#     res <- as.numeric(tensorflow::tf$sparse$reduce_sum(x@gm, 1L))
#   }else{
#     res <- as.vector(tensorflow::tf$reduce_sum(x@gm,axis=1L))
#   }
#   names(res) <- rownames(x)
#   return(res)
# })

setMethod("min", signature(x = "gpu.matrix.tensorflow"), function(x){
  if(x@sparse){
    res <- as.numeric(tensorflow::tf$reduce_min(x@gm$values))
  } else{
    res <- as.numeric(tensorflow::tf$reduce_min(x@gm))
  }
  return(res)
})

setMethod("max", signature(x = "gpu.matrix.tensorflow"), function(x){
  if (x@sparse) {
    res <- as.numeric(tensorflow::tf$sparse$reduce_max(x@gm))
  }else{
    res <-as.numeric(tensorflow::tf$reduce_max(x@gm))
  }
  return(res)
})

setMethod("which.max", signature(x = "gpu.matrix.tensorflow"), function(x){

  if (x@sparse) {
    vecSearch <- tensorflow::as_tensor(as.vector(x),dtype = castDtype_tensorflow(dtype(x)))
    max_val <- tensorflow::tf$reduce_max(vecSearch, keepdims=F)
    cond <- tensorflow::tf$equal(vecSearch, max_val)
    res <- as.numeric(tensorflow::tf$where(cond)) + 1
  }else{
    vecSearch <- tensorflow::as_tensor(as.vector(x),dtype = castDtype_tensorflow(dtype(x)))
    max_val <- tensorflow::tf$reduce_max(vecSearch, keepdims=F)
    cond <- tensorflow::tf$equal(vecSearch, max_val)
    res <- as.numeric(tensorflow::tf$where(cond)) + 1
  }
  return(res)
})

setMethod("which.min", signature(x = "gpu.matrix.tensorflow"), function(x){

  if (x@sparse) {
    vecSearch <- tensorflow::as_tensor(as.vector(x),dtype = castDtype_tensorflow(dtype(x)))
    min_val <- tensorflow::tf$reduce_min(vecSearch, keepdims=F)
    cond <- tensorflow::tf$equal(vecSearch, min_val)
    res <- as.numeric(tensorflow::tf$where(cond)) + 1
  }else{
    vecSearch <- tensorflow::as_tensor(as.vector(x),dtype = castDtype_tensorflow(dtype(x)))
    min_val <- tensorflow::tf$reduce_min(vecSearch, keepdims=F)
    cond <- tensorflow::tf$equal(vecSearch, min_val)
    res <- as.numeric(tensorflow::tf$where(cond)) + 1
  }
  return(res)
})

setMethod("aperm", signature(a="gpu.matrix.tensorflow"), function(a,perm,...){
  res <- aperm(as.matrix(a),perm)

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
  # if (is.object(X))
  #   X <- if (dl == 2L)
  #     as.matrix(X)
  # else{
  #   as.array(X)
  # }

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
  newX <- aperm(X, c(s.call, s.ans))
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
  }
  else for (i in 1L:d2) {
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

# setGeneric("apply", function(X, MARGIN, FUN, ..., simplify = TRUE) standardGeneric("apply"))
setMethod("apply", signature(X="gpu.matrix.tensorflow"), function(X, MARGIN, FUN, ..., simplify = TRUE){
  applyTest(X, MARGIN, FUN, ..., simplify = TRUE)

})


setMethod("cov", signature(x = "gpu.matrix.tensorflow", y = "ANY"), function(x,y){
  x <- warningInteger(x)
  x <- warningSparseTensor(x)

  castMatrix <- castTypeOperations(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  x_ <- t(x) - colMeans(x)
  y_ <- t(y) - colMeans(y)

  res <- tcrossprod(x_, y_)/(ncol(x)-1)
  return(res)
})

setMethod("cov", signature(x = "ANY", y = "gpu.matrix.tensorflow"), function(x,y){
  y <- warningInteger(y)
  y <- warningSparseTensor(y)

  castMatrix <- castTypeOperations(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  x_ <- t(x) - colMeans(x)
  y_ <- t(y) - colMeans(y)

  res <- tcrossprod(x_, y_)/(ncol(x)-1)
  return(res)
})

setMethod("cov", signature(x = "gpu.matrix.tensorflow", y = "missing"), function(x,y){
  x <- warningInteger(x)
  x <- warningSparseTensor(x)
  res <- tcrossprod(t(x) - colMeans(x))/(ncol(x)-1)
  return(res)
})

setMethod("cov2cor", signature(V="gpu.matrix.tensorflow"), function(V){
  V <- warningInteger(V)
  p <- (d <- dim(V))[1L]
  Is <- sqrt(1/diag(V))
  r<-V
  r <- Is * V * rep(Is, each = p)
  r[cbind(1L:p, 1L:p)] <- 1
  dimnames(r) <- dimnames(V)
  return(r)
})



setMethod("cor", signature(x = "gpu.matrix.tensorflow", y = "ANY"), function(x,y){
  x <- warningSparseTensor(x)
  castMatrix <- castTypeOperations(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  V <- cov(x,y)
  res <- cov2cor(V)
  dimnames(res) <- dimnames(V)
  return(res)
})

setMethod("cor", signature(x = "ANY", y = "gpu.matrix.tensorflow"), function(x,y){
  y <- warningSparseTensor(y)
  castMatrix <- castTypeOperations(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  V <- cov(x,y)
  res <- cov2cor(V)
  dimnames(res) <- dimnames(V)
  return(res)
})

setMethod("cor", signature(x = "gpu.matrix.tensorflow", y = "missing"), function(x,y){
  x <- warningSparseTensor(x)
  V <- cov(x)
  res <- cov2cor(V)
  dimnames(res) <- dimnames(V)
  return(res)
})

setMethod("cor", signature(x = "gpu.matrix.tensorflow", y = "ANY",use="missing", method = "character"), function(x,y,method){
  x <- warningSparseTensor(x)

  castMatrix <- castTypeOperations(x,y)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  if(method=="spearman"){
    x <- gpu.matrix.tensorflow(t(colRanks(x))@gm,dtype = castDtype_tensorflow(dtype(x)), dimnames = dimnames(x))
    y <- gpu.matrix.tensorflow(t(colRanks(y))@gm,dtype = castDtype_tensorflow(dtype(y)), dimnames = dimnames(y))
  }

  V <- cov(x,y)

  res <- cov2cor(V)
  dimnames(res) <- dimnames(V)
  return(res)
})

setMethod("cor", signature(x = "gpu.matrix.tensorflow", y = "missing", use="missing", method = "character"), function(x,y,method){
  x <- warningSparseTensor(x)
  if(method=="spearman"){
    x <- gpu.matrix.tensorflow(t(colRanks(x))@gm,dtype = castDtype_tensorflow(dtype(x)), dimnames = dimnames(x))
  }
  V <- cov(x)
  res <- cov2cor(V)
  dimnames(res) <- dimnames(V)
  return(res)
})
# library(matrixStats)
setGeneric("rowVars", function(x) standardGeneric("rowVars"))
setMethod("rowVars", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  x <- warningInteger(x)
  res <- (as.numeric(tensorflow::tf$math$reduce_variance(x@gm,axis=1L))*nrow(x))/(nrow(x)-1)
  return(res)
})
# library(matrixStats)
setGeneric("colVars", function(x,y) standardGeneric("colVars"))
setMethod("colVars", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  x <- warningInteger(x)
  res <- (as.numeric(tensorflow::tf$math$reduce_variance(x@gm,axis=0L))*ncol(x))/(ncol(x)-1)
  return(res)
})

setGeneric("colMaxs", function(x,y) standardGeneric("colMaxs"))
setMethod("colMaxs", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  res <- as.vector(tensorflow::tf$reduce_max(x@gm,axis=0L))
  names(res) <- colnames(x)
  return(res)
})
setGeneric("rowMaxs", function(x,y) standardGeneric("rowMaxs"))
setMethod("rowMaxs", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  res <- as.vector(tensorflow::tf$reduce_max(x@gm,axis=1L))
  names(res) <- rownames(x)
  return(res)
})

setGeneric("rowRanks", function(x) standardGeneric("rowRanks"))
setMethod("rowRanks", signature(x="gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  return(gpu.matrix.tensorflow(tensorflow::tf$argsort(tensorflow::tf$argsort(x@gm,axis=1L), axis=1L) + 1))
} )
setGeneric("colRanks", function(x) standardGeneric("colRanks"))
setMethod("colRanks", signature(x="gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  return(t(gpu.matrix.tensorflow(tensorflow::tf$argsort(tensorflow::tf$argsort(x@gm,axis=0L), axis=0L) + 1)))
} )
setGeneric("colMins", function(x) standardGeneric("colMins"))
setMethod("colMins", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  res <- as.vector(tensorflow::tf$reduce_min(x@gm,axis=0L))
  names(res) <- colnames(x)
  return(res)
})
setGeneric("rowMins", function(x) standardGeneric("rowMins"))
setMethod("rowMins", signature(x = "gpu.matrix.tensorflow"), function(x){
  x <- warningSparseTensor(x)
  res <- as.vector(tensorflow::tf$reduce_min(x@gm,axis=1L))
  names(res) <- rownames(x)
  return(res)
})
