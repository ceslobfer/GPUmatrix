# setGeneric("as.double", function(x) standardGeneric("as.double"))
setMethod("as.numeric", signature(x = "gpu.matrix.torch"), function(x, ...) as(as.vector(x),"double") )
# setMethod("as.double", signature(x = "gpu.matrix.torch"), function(x, ...) as.numer(x),"double") )

setAs("ANY", "gpu.matrix.torch", function(from) gpu.matrix.torch(from))
setAs("gpu.matrix.torch", "matrix",
      function(from){
        from <- warningSparseTensor_torch(from)
        # typeData <- from@gm$dtype
        res <- as.matrix(from@gm$cpu())
        dimnames(res) <- dimnames(from)
        # if (typeData == tf$int32 | typeData == tf$int64) {
        #   mode(res) <- "integer"
        # }
        return(res)
      })
setGeneric("as.gpu.matrix", function(x,...) standardGeneric("as.gpu.matrix"))
setMethod("as.gpu.matrix", signature(x = "ANY"), function(x, ...) gpu.matrix(x,...) )
# setAs("gpu.matrix.torch","double", function(from){
#   return(as.double(as.vector(from)))
# })
setMethod("as.matrix", signature(x = "gpu.matrix.torch"), function(x, ...){
  res <- as(x,"matrix")
  dimnames(res) <- dimnames(x)
  return(res)
})
# setMethod("as.matrix.default", signature(x = "gpu.matrix.torch"), function(x, ...) as(x,"matrix") )



# setGeneric("as.gpu.matrix.torch", function(x) standardGeneric("as.gpu.matrix.torch"))
# setMethod("as.gpu.matrix.torch", signature(x = "ANY"), function(x){
#   return(gpu.matrix.torch(x))
# } )

# setAs("matrix", "gpu.matrix.torch", function(from) gpu.matrix.torch(as.matrix(from)))

setMethod("as.array",  signature(x = "gpu.matrix.torch"), function(x, ...){
  return(as(x@gm$cpu(),"matrix"))
} )

setMethod("as.vector", "gpu.matrix.torch", function(x, mode){
  x <- warningSparseTensor_torch(x)
  x <- as.matrix(x)
  res <- as.vector(x,mode)
  return(res)
})

setMethod("as.list", signature(x = "gpu.matrix.torch"), function(x, ...){
  return(as.list(as.matrix(x)))
})
# setMethod("is.matrix",  signature(x = "gpu.matrix.torch"), function(x){
#   return(T)
# })
setMethod("is.numeric",  signature(x = "gpu.matrix.torch"), function(x){
  dtype <-dtype(x)
  res <- F
  if (dtype(x) != "bool" & dtype(x) != "complex32" & dtype(x) != "complex64") {
    res <-T
  }
  return(res)
} )

# setMethod("is.complex",  signature(x = "gpu.matrix.torch"), function(x,...){
#   dtype <-dtype(x)
#   res <- F
#   if (dtype(x) == "complex32" & dtype(x) == "complex64") {
#     res <-T
#   }
#   return(res)
# } )
