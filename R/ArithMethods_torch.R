warningSparseTensor_torch <- function(x){
  if(x@sparse){
    x <- to_dense_torch(x)
    warning(message = "Not allowed with sparse matrix, matrix will be cast to dense for the operation. May cause memory problems")
  }
  return(x)
}

is_dtype_greater <- function(dtype1, dtype2) {
  dtype_order <- c("float64","float32","int")

  if(match(dtype1, dtype_order) > match(dtype2, dtype_order)) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}
castTypeOperations_torch <- function(m1, m2, operator=FALSE, todense=TRUE, sameType = FALSE){
  if(requireNamespace('torch')){
    m1Class <- class(m1)[1]
    m2Class <- class(m2)[1]
    defaultType = torch::torch_float64()
    if (m1Class=="float32") {
      m1 <- gpu.matrix.torch(data = m1, dtype = "float32", dimnames = dimnames(m1), device=device(m2))
      m1Class <- class(m1)[1]
    }else if (m2Class=="float32"){
      m2 <- gpu.matrix.torch(data = m2, dtype = "float32", dimnames = dimnames(m2), device=device(m1))
      m2Class <- class(m2)[1]
    }

    if (operator & (m1Class=="integer" | m1Class=="numeric")) {
      m1 <- gpu.matrix.torch(data = m1, nrow = nrow(m2), ncol = ncol(m2), dimnames = dimnames(m1), dtype = m2@gm$dtype, device=device(m2) )
      m1Class <- class(m1)[1]
    }else if (operator & (m2Class=="integer" | m2Class=="numeric")){
      m2 <- gpu.matrix.torch(data = m2, nrow = nrow(m1), ncol = ncol(m1), dimnames = dimnames(m2),dtype = m1@gm$dtype, device=device(m1))
      m2Class <- class(m2)[1]
    }


    if (m1Class[1]!="gpu.matrix.torch") {
      m1 <- gpu.matrix.torch(data = m1, dimnames = dimnames(m1), device=device(m2))
    }
    if (m2Class[1]!="gpu.matrix.torch") {
      m2 <- gpu.matrix.torch(data = m2, dimnames = dimnames(m2), device=device(m1))
    }

    if (sameType) {
      if (is_dtype_greater(dtype(m1),dtype(m2))) {
        dtype(m1) <- dtype(m2)
      }else{
        dtype(m2) <- dtype(m1)
      }
    }


    if (todense) {
      m1 <- warningSparseTensor_torch(m1)
      m2 <- warningSparseTensor_torch(m2)
    }
  }
  return(list(m1,m2))
}

prodGPUmat_torch <- function(e1,e2){

  if(e1@sparse & length(e2)==1){
    resTensor <- e1@gm * e2
  }else{
    castMatrix <- castTypeOperations_torch(e1,e2, operator = TRUE, todense = F)
    e1 <- castMatrix[[1]]
    e2 <- castMatrix[[2]]
    # GM((e1@sparse & length(e2)==1)|(e2@sparse & length(e1)==1) )
    if (!(e2@sparse & e1@sparse)) {
      e1 <- warningSparseTensor_torch(e1)
      e2 <- warningSparseTensor_torch(e2)
    }

    resTensor <- e1@gm * e2@gm
  }



  return(resTensor)

}

divisionGPUmat_torch <- function(e1,e2){

  castMatrix <- castTypeOperations_torch(e1,e2,operator = T, todense = F)
  e1 <- castMatrix[[1]]
  e2 <- castMatrix[[2]]
  e2 <- warningInteger(e2)
  #One sparse
  if (e2@sparse) {
    warning(message = "Not allowed with sparse matrix as denominator, matrix will be cast to dense for the operation. May cause memory problems")
    e2 <- to_dense_torch(e2)
  }
  if (e1@sparse & (sum(dim(e1))>1)) {
    e1 <- warningSparseTensor_torch(e1)
  }

  resTensor <- e1@gm/e2@gm

  e1@gm <- resTensor

  return(e1)

}

sumGPUmat_torch <- function(e1,e2, operator){

  castMatrix <- castTypeOperations_torch(e1,e2,operator = T,todense = F)
  e1 <- castMatrix[[1]]
  e2 <- castMatrix[[2]]
  if ((e1@sparse==T) & (e2@sparse==F)) e1<-warningSparseTensor_torch(e1)

  if (operator == "+") {
    res <- e1@gm + e2@gm
  }else{
    res <- e1@gm - e2@gm
  }
  e1@gm <- res

  return(e1)
}


MatprodGPUmat_torch <- function(x,y){

  castMatrix <- castTypeOperations_torch(x,y, todense = FALSE,sameType = T)
  x <- castMatrix[[1]]
  y <- castMatrix[[2]]
  x <- warningInteger(x)
  y <- warningInteger(y)
  if (ncol(x)==nrow(y)){

    y <- warningSparseTensor_torch(y)

    if(requireNamespace('torch')){
      x@gm <- torch::torch_matmul(self=x@gm,other=y@gm)
    }

    x@sparse <- F
    colnames(x) <- colnames(y)
    return(x)
  } else{
    stop("The matrix cannot be multiplied (check for compatible dimensions).")
  }
}

setMethod("-", signature(e1 = "gpu.matrix.torch", e2 = "missing"), function(e1, e2){
  if(e1@sparse){
    if(requireNamespace('torch')){
      e1@gm <- torch::torch_sparse_coo_tensor(indices = e1@gm$indices(), values = -e1@gm$values(), size = dim(e1))
    }
  }else{
    e1@gm <- -e1@gm
  }

  return(e1)
})

setMethod("%*%", signature(x = "gpu.matrix.torch", y="ANY"), function(x, y) {
  MatprodGPUmat_torch(x,y)
})
setMethod("%*%", signature(x = "ANY", y="gpu.matrix.torch"), function(x, y) {
  MatprodGPUmat_torch(x,y)
})

setMethod("Arith",
          c(e1="gpu.matrix.torch", e2="ANY"),
          function(e1, e2)
          {
            op = .Generic[[1]]
            if (length(e2)==1 & !e1@sparse){
              if(requireNamespace('torch')){
                e2 <- torch::torch_tensor(e2,dtype = e1@gm$dtype,device = device(e1))
              }
              switch(op,
                     '+' = {
                       e1@gm <- e1@gm + e2
                       return(e1)
                     },
                     '-' = {
                       e1@gm <- e1@gm - e2
                       return(e1)
                     },
                     '*' = {
                       e1@gm <- e1@gm * e2
                       return(e1)
                     },
                     '/' = {
                       e1@gm <- e1@gm / e2
                       return(e1)
                     },
                     '^'={
                      e1@gm <- e1@gm ^ e2
                       return(e1)
                     },
                     '%%'={
                       castMatrix <- castTypeOperations_torch(e1,e2, todense = T)
                       e1 <- castMatrix[[1]]
                       e2 <- castMatrix[[2]]
                       e1@gm <- e1@gm%%e2
                       return(e1)
                     },
                     '%/%'={
                       castMatrix <- castTypeOperations_torch(e1,e2, todense = T)
                       e1 <- castMatrix[[1]]
                       e2 <- castMatrix[[2]]
                       e1@gm <- e1@gm%/%e2
                       return(e1)
                     }
              )
            }else{

              switch(op,
                     '+' = {
                       sumGPUmat_torch(e1,e2, operator = "+")
                     },
                     '-' = {
                       sumGPUmat_torch(e1,e2, operator = "-")
                     },
                     '*' = {
                       e1@gm <- prodGPUmat_torch(e1,e2)

                       return(e1)
                     },
                     '/' = {
                       if(e1@sparse & length(e2)==1){
                         e1@gm/e2
                       }else{
                         divisionGPUmat_torch(e1,e2)
                       }

                     },
                     '^'={
                       # if (e1@sparse) e1<-to_dense_torch(e1)
                       #Mejorar
                       if (inherits(e2,"gpu.matrix.torch")) {
                         e2<-warningSparseTensor_torch(e2)
                         e1<-warningSparseTensor_torch(e1)

                         e1@gm <- e1@gm ^ e2@gm
                       }else{
                         e1@gm <- e1@gm ^ e2
                       }

                       return(e1)
                     },
                     '%%'={
                       castMatrix <- castTypeOperations_torch(e1,e2, todense = T)
                       e1 <- castMatrix[[1]]
                       e2 <- castMatrix[[2]]
                       e1@gm <- e1@gm%%e2@gm
                       return(e1)
                     },
                     '%/%'={
                       castMatrix <- castTypeOperations_torch(e1,e2, todense = T)
                       e1 <- castMatrix[[1]]
                       e2 <- castMatrix[[2]]
                       e1@gm <- e1@gm%/%e2@gm
                       return(e1)
                     }
              )
            }

          }
)

setMethod("Arith",
          c(e1="ANY", e2="gpu.matrix.torch"),
          function(e1, e2)
          {
            op = .Generic[[1]]
            if (length(e1)==1 & !e2@sparse){
              e1 <- torch::torch_tensor(e1,dtype = e2@gm$dtype,device = device(e2))
              switch(op,
                     '+' = {
                       e2@gm <- e2@gm + e1
                       return(e2)
                     },
                     '-' = {
                       e2@gm <- e1 - e2@gm
                       return(e2)
                     },
                     '*' = {
                       e2@gm <- e2@gm * e1
                       return(e2)
                     },
                     '/' = {
                       e2 <- warningInteger(e2)
                       e2@gm <- e1/e2@gm
                       return(e2)
                     },
                     '^'={
                       # if (e2@sparse) e2<-to_dense_torch(e2)
                       #Mejorar
                       e2@gm <- e1 ^ e2@gm

                       return(e2)
                     },
                     '%%'={
                       castMatrix <- castTypeOperations_torch(e1,e2, todense = T)
                       e1 <- castMatrix[[1]]
                       e2 <- castMatrix[[2]]
                       e1@gm <- e1%%e2@gm
                       return(e1)
                     },
                     '%/%'={
                       castMatrix <- castTypeOperations_torch(e1,e2, todense = T)
                       e1 <- castMatrix[[1]]
                       e2 <- castMatrix[[2]]
                       e1@gm <- e1%/%e2@gm
                       return(e1)
                     }
              )
            }else{
              switch(op,
                     '+' = {
                       sumGPUmat_torch(e1,e2, operator = "+")
                     },
                     '-' = {
                       sumGPUmat_torch(e1,e2, operator = "-")
                     },
                     '*' = {
                       e2@gm <- prodGPUmat_torch(e2,e1)

                       return(e2)
                     },
                     '/' = {
                       divisionGPUmat_torch(e1,e2)
                     },
                     '^'={
                       e2 <- warningSparseTensor_torch(e2)

                       e2@gm <- e1 ^ e2@gm
                       return(e2)
                     },
                     '%%'={
                       castMatrix <- castTypeOperations_torch(e1,e2, todense = T)
                       e1 <- castMatrix[[1]]
                       e2 <- castMatrix[[2]]
                       e1@gm <- e1@gm%%e2@gm
                       return(e1)
                     },
                     '%/%'={
                       castMatrix <- castTypeOperations_torch(e1,e2, todense = T)
                       e1 <- castMatrix[[1]]
                       e2 <- castMatrix[[2]]
                       e1@gm <- e1@gm%/%e2@gm
                       return(e1)
                     }
              )
            }

          }
)



