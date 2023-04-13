warningSparseTensor <- function(x){
  if(x@sparse){
    x <- to_dense_tensorflow(x)
    warning(message = "Not allowed with sparse matrix, matrix will be cast to dense for the operation. May cause memory problems")
  }
  return(x)
}
castTypeOperations <- function(m1, m2, operator=FALSE, todense=TRUE, sameType = F){

  m1Class <- class(m1)[1]
  m2Class <- class(m2)[1]
  if (m1Class=="float32") {
    m1 <- gpu.matrix.tensorflow(data = m1, dtype = tensorflow::tf$float32, dimnames = dimnames(m1))
    m1Class <- class(m1)[1]
  }else if (m2Class=="float32"){
    m2 <- gpu.matrix.tensorflow(data = m2, dtype = tensorflow::tf$float32, dimnames = dimnames(m2))
    m2Class <- class(m2)[1]
  }

  if (operator & (m1Class=="integer" | m1Class=="numeric")) {
    m1 <- gpu.matrix.tensorflow(data = m1, nrow = nrow(m2), ncol = ncol(m2), dimnames = dimnames(m1), dtype = m2@gm$dtype )
    m1Class <- class(m1)[1]
  }else if (operator & (m2Class=="integer" | m2Class=="numeric")){
    m2 <- gpu.matrix.tensorflow(data = m2, nrow = nrow(m1), ncol = ncol(m1), dimnames = dimnames(m2),dtype = m1@gm$dtype)
    m2Class <- class(m2)[1]
  }



  if (m2Class[1]!="gpu.matrix.tensorflow") {
    m2 <- gpu.matrix.tensorflow(data = m2, dimnames = dimnames(m2),dtype = m1@gm$dtype)
  }
  if (m1Class[1]!="gpu.matrix.tensorflow") {
    m1 <- gpu.matrix.tensorflow(data = m1, dimnames = dimnames(m1),dtype = m2@gm$dtype)
  }
  if (todense) {
    m1 <- warningSparseTensor(m1)
    m2 <- warningSparseTensor(m2)
  }

  if (sameType) {
    if (is_dtype_greater(dtype(m1),dtype(m2))) {
      dtype(m1) <- dtype(m2)
    }else{
      dtype(m2) <- dtype(m1)
    }
  }

  return(list(m1,m2))
}

prodGPUmat <- function(e1,e2){
  # e1Class <- class(e1)[1]
  # e2Class <- class(e2)[1]

  castMatrix <- castTypeOperations(e1,e2, operator = TRUE, todense = T, sameType = T)
  e1 <- castMatrix[[1]]
  e2 <- castMatrix[[2]]
  resTensor <- e1@gm * e2@gm

  res <- gpu.matrix.tensorflow(data = resTensor)
  return(suppressWarnings(res))

}

divisionGPUmat <- function(e1,e2){

  castMatrix <- castTypeOperations(e1,e2,operator = T, todense = F, sameType = T)
  e1 <- castMatrix[[1]]
  e2 <- castMatrix[[2]]
  e2 <- warningInteger(e2)
  #One sparse
  if (e2@sparse) {
    warning(message = "Not allowed with sparse matrix as denominator, matrix will be cast to dense for the operation. May cause memory problems")
    e2 <- to_dense_tensorflow(e2)
  }

  resTensor <- tensorflow::tf$divide(e1@gm, e2@gm)
  e1@gm <- resTensor

  return(e1)

}

sumGPUmat <- function(e1,e2, operator){

  castMatrix <- castTypeOperations(e1,e2,operator = T,todense = T, sameType = T)
  e1 <- castMatrix[[1]]
  e2 <- castMatrix[[2]]

  if (operator == "+") {
    res <- e1@gm + e2@gm
  }else{
    res <- e1@gm - e2@gm
  }
  e1@gm <- res

  return(e1)
}


MatProdGPUmat <- function(x,y){
  if(requireNamespace('tensorflow')){
    castMatrix <- castTypeOperations(x,y, todense = FALSE, sameType = T)
    x <- castMatrix[[1]]
    y <- castMatrix[[2]]
    x <- warningInteger(x)
    y <- warningInteger(y)

    if (ncol(x)==nrow(y)){

      #Both sparse
      if (x@sparse & y@sparse) {

        warning(message = "Not allowed with two sparse matrix, the smallest matrix will be cast to dense for the operation. May cause memory problems")
        if (as.numeric(tensorflow::tf$size(x@gm)) < as.numeric(tensorflow::tf$size(y@gm))) {
          x <- to_dense_tensorflow(x)
        }else y <- to_dense_tensorflow(y)
      }

      #One sparse
      if ((x@sparse & !y@sparse) | (!x@sparse & y@sparse)) {

        resTensor <- tensorflow::tf$sparse$sparse_dense_matmul(x@gm,y@gm)
        x@gm <- resTensor

      }else{
        resTensor <- tensorflow::tf$matmul(x@gm,y@gm)
        x@gm <- resTensor
      }
      x@sparse <- F
      colnames(x) <- colnames(y)
      return(x)
    } else{
      stop("The matrix cannot be multiplied (check for compatible dimensions).")
    }
  }
}

setMethod("-", signature(e1 = "gpu.matrix.tensorflow", e2 = "missing"), function(e1, e2){
  e1@gm <- -e1@gm
  return(e1)
})

setMethod("%*%", signature(x = "gpu.matrix.tensorflow", y="ANY"), function(x, y) {
  MatProdGPUmat(x,y)
})
setMethod("%*%", signature(x = "ANY", y="gpu.matrix.tensorflow"), function(x, y) {
  MatProdGPUmat(x,y)
})

setMethod("Arith",
          c(e1="gpu.matrix.tensorflow", e2="ANY"),
          function(e1, e2)
          {
            op = .Generic[[1]]
            switch(op,
                   '+' = {
                     sumGPUmat(e1,e2, operator = "+")
                   },
                   '-' = {
                     sumGPUmat(e1,e2, operator = "-")
                   },
                   '*' = {
                     res <- prodGPUmat(e1,e2)
                     dimnames(res) <- dimnames(e1)
                     res
                   },
                   '/' = {
                     e1 <- warningInteger(e1)
                     divisionGPUmat(e1,e2)
                   },
                   '^'={
                     if (e1@sparse) e1<-to_dense_tensorflow(e1)
                     #Mejorar
                     if (inherits(e2,"gpu.matrix.tensorflow") ) {
                       if (e2@sparse) e2<-to_dense_tensorflow(e2)

                       res <- gpu.matrix.tensorflow(e1@gm ^ e2@gm, dimnames = dimnames(e1))
                     }else{
                       res <- gpu.matrix.tensorflow(e1@gm ^ e2, dimnames = dimnames(e1))
                     }

                     return(res)
                   }
            )
          }
)

setMethod("Arith",
          c(e1="ANY", e2="gpu.matrix.tensorflow"),
          function(e1, e2)
          {
            op = .Generic[[1]]
            switch(op,
                   '+' = {
                     sumGPUmat(e1,e2, operator = "+")
                   },
                   '-' = {
                     sumGPUmat(e1,e2, operator = "-")
                   },
                   '*' = {
                     res <- prodGPUmat(e1,e2)
                     dimnames(res) <- dimnames(e2)
                     res
                   },
                   '/' = {
                     e2 <- warningInteger(e2)
                     divisionGPUmat(e1,e2)
                   },
                   '^'={
                     if (e2@sparse) e2<-to_dense_tensorflow(e2)

                     res <- gpu.matrix.tensorflow(e1 ^ e2@gm, dimnames = dimnames(e2))
                     return(res)
                   }
            )
          }
)

setMethod("Arith",
          c(e1="numeric", e2="gpu.matrix.tensorflow"),
          function(e1, e2)
          {
            op = .Generic[[1]]
            switch(op,
                   '+' = {
                     sumGPUmat(e1,e2, operator = "+")
                   },
                   '-' = {
                     sumGPUmat(e1,e2, operator = "-")
                   },
                   '*' = {
                     res <- prodGPUmat(e1,e2)
                     dimnames(res) <- dimnames(e2)
                     res
                   },
                   '/' = {
                     divisionGPUmat(e1,e2)
                   },
                   '^'={
                     if (e2@sparse) {
                       e2 <- gpu.matrix.tensorflow(data = tensorflow::tf$sparse$to_dense(e2@gm),dimnames = dimnames(e2))
                     }
                     res <- gpu.matrix.tensorflow(e1 ^ e2@gm, dimnames = dimnames(e2))
                     return(res)
                   }
            )
          }
)




