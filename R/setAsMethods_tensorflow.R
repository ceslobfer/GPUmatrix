# setGeneric("as.double", function(x) standardGeneric("as.double"))
setMethod("as.numeric", signature(x = "gpu.matrix.tensorflow"), function(x, ...) as(as.vector(x),"double") )
# setMethod("as.double", signature(x = "gpu.matrix.tensorflow"), function(x, ...) as.numer(x),"double") )

setAs("ANY", "gpu.matrix.tensorflow", function(from) gpu.matrix.tensorflow(as.matrix(from)))
setAs("gpu.matrix.tensorflow", "matrix",
      function(from){
        if (from@sparse) from <- to_dense_tensorflow(from)
        typeData <- from@gm$dtype
        res <- as.matrix(from@gm)
        dimnames(res) <- dimnames(from)
        if (typeData == tensorflow::tf$int32 | typeData == tensorflow::tf$int64) {
          mode(res) <- "integer"
        }
        return(res)
      })
# setAs("gpu.matrix.tensorflow","double", function(from){
#   return(as.double(as.vector(from)))
# })
setMethod("as.matrix", signature(x = "gpu.matrix.tensorflow"), function(x, ...) as(x,"matrix") )
# setMethod("as.matrix.default", signature(x = "gpu.matrix.tensorflow"), function(x, ...) as(x,"matrix") )



# setGeneric("as.gpu.matrix.tensorflow", function(x) standardGeneric("as.gpu.matrix.tensorflow"))
# setMethod("as.gpu.matrix.tensorflow", signature(x = "ANY"), function(x){
#   return(gpu.matrix.tensorflow(x))
# } )

# setAs("matrix", "gpu.matrix.tensorflow", function(from) gpu.matrix.tensorflow(as.matrix(from)))

setMethod("as.array",  signature(x = "gpu.matrix.tensorflow"), function(x, ...){
  return(as(x@gm,"matrix"))
} )

# setMethod("is.matrix",  signature(x = "gpu.matrix.tensorflow"), function(x){
#   return(T)
# })

setMethod("is.numeric",  signature(x = "gpu.matrix.tensorflow"), function(x){
  dtype <-dtype(x)
  res <- F
  if (dtype(x) != "bool" & dtype(x) != "complex32" & dtype(x) != "complex64") {
    res <-T
  }
  return(res)
} )

setMethod("as.vector", "gpu.matrix.tensorflow", function(x, mode){
  x <- warningSparseTensor(x)
  x <- as.matrix(x)
  return(as.vector(x, mode))
})

setMethod("as.list", signature(x = "gpu.matrix.tensorflow"), function(x, ...){
  return(as.list(as.matrix(x)))
})
