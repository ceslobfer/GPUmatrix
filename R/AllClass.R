# library(tensorflow)
# install_tensorflow(gpu = TRUE)
# use_condaenv("tf")
# tfp <- import("tensorflow_probability")
# library(torch)

setClass("gpu.matrix.tensorflow",
         representation(
           # data="vector",
           rownames = "ANY",
           colnames = "ANY",
           gm = "ANY",
           sparse = "logical",
           type="character")
)
setClass("gpu.matrix.torch",
         representation(
           # data="vector",
           rownames = "ANY",
           colnames = "ANY",
           gm = "ANY",
           sparse = "logical",
           type="character")
)
# library(torch)

castDtype_torch <- function(type) {
  if(requireNamespace('torch')){
    switch(type,
           "float32" = {
             res <- torch::torch_float32()
           },
           "float64" = {
             res <- torch::torch_float64()
           },
           "int" = {
             res <- torch::torch_int()
           },
           "bool" = {
             res <- torch::torch_bool()
           },
           "complex64"={
             res <- torch::torch_cfloat64()
           },
           "complex32"={
             res <- torch::torch_cfloat32()
           },
           "logical" = {
             res <- torch::torch_bool()
           },
           stop("Invalid input type")
    )
  }
  return(res)
}

castDtype_tensorflow <- function(type) {
  switch(type,
         "float32" = {
           res <- tensorflow::tf$float32
         },
         "float64" = {
           res <- tensorflow::tf$float64
         },
         "int" = {
           res <- tensorflow::tf$int64
         },
         "bool" = {
           res <- tensorflow::tf$bool
         },
         "complex64"={
           res <- tensorflow::tf$complex64
         },
         "complex32"={
           res <- tensorflow::tf$complex64
         },
         "logical" = {
           res <- tensorflow::tf$bool
         },
         stop("Invalid input type")
  )
  return(res)
}

gpu.matrix.torch <- function(data = NA, nrow = NULL, ncol = NULL, byrow = FALSE,
                                  dimnames = NULL, dtype=NULL, sparse=NULL, colnames=c(), rownames=c(),device=NULL){
  if (byrow) ncol=length(data)/nrow
  if (is.null(dtype)) dtype <- "float64"
  charDtype <- dtype
  if (class(dtype)[[1]] != "torch_dtype") dtype <- castDtype_torch(dtype)
  #

  classData <- class(data)[1]
  # if (classData == "torch_tensor" & is.null(device)) device <- data$device

  if (classData == "torch_tensor" & is.null(device)){
    if (data$is_cuda){
      device <- "cuda"
    }else{
      device <- "cpu"
    }
  }

  if (is.null(device)){
    if (torch::cuda_is_available()){
      device <- "cuda"
    }else{
      warning(message = "Not cuda available")
      device<-"cpu"
    }

  }else{
    if (device=="cuda" & !torch::cuda_is_available()){
      warning(message = "Not cuda available")
      device<-"cpu"
    }
  }


  device_torch <- torch::torch_device(type = device)
  sparseCast <- F

  switch(classData,
         matrix={
           gm <- torch::torch_tensor(data,device = device_torch,dtype = dtype)
         },
         data.frame={
           gm <- torch::torch_tensor(as.matrix(data),device = device_torch,dtype = dtype)
         },
         dgeMatrix={
           gm <- torch::torch_tensor(as.matrix(data),device = device_torch,dtype = dtype)

         },
         ddiMatrix={
           data <- as(data,"dgCMatrix")
           i <- data@i + 1
           j <- findInterval(seq(data@x)-1,data@p[-1]) + 1
           indices <- torch::torch_tensor(as.matrix(rbind(i,j)), dtype = torch::torch_long())
           gm <- torch::torch_sparse_coo_tensor(indices = indices, values = data@x, size = data@Dim, device = device_torch)
           if (!gm$is_cuda & device=="cuda")  gm <- gm$cuda()
           if (gm$is_cuda & device!="cuda")  gm <- gm$cpu()
           gm <- gm$coalesce()
           # gm <- tensorflow::tf$sparse$reorder(gm)
           sparseCast <- T
           if (is.null(sparse)) sparse <- T
         },
         dpoMatrix={
           gm <- torch::torch_tensor(as.matrix(data),device = device,dtype = dtype)

         },
         dgCMatrix={
           i <- data@i + 1
           j <- findInterval(seq(data@x)-1,data@p[-1]) + 1
           indices <- torch::torch_tensor(as.matrix(rbind(i,j)),dtype=torch::torch_long())
           gm <- torch::torch_sparse_coo_tensor(indices = indices, values = data@x, size = data@Dim, device = device_torch)
           gm <- gm$coalesce()
           if (!gm$is_cuda & device=="cuda")  gm <- gm$cuda()
           if (gm$is_cuda & device!="cuda")  gm <- gm$cpu()
           sparseCast <- T
           if (is.null(sparse)) sparse <- T
         },
         float32={
           if(requireNamespace("float")){
            gm <- torch::torch_tensor(float::dbl(data),device = device_torch)
           }
           if (is.null(dtype)) {
             gm <- gm$to(torch::torch_float32())

           }
           if (!is.null(nrow) & !is.null(ncol)) gm$resize_(c(nrow,ncol))
           if (!is.null(nrow)) gm$resize_(c(nrow,ncol(gm)))
           if (!is.null(ncol)) gm$resize_(c(nrow(gm),ncol))

         },
         torch_tensor={
           if (!is.null(dtype) & data$dtype != dtype) {
             data <- data$to(dtype)
           }
           if (!is.null(nrow) & !is.null(ncol)) data$resize_(c(nrow,ncol))
           if (!is.null(nrow)) data$resize_(c(nrow,ncol(data)))
           if (!is.null(ncol)) data$resize_(c(nrow(data),ncol))
           if (!data$is_cuda & device=="cuda")  data <- data$cuda()
           if (data$is_cuda & device!="cuda")  data <- data$cpu()
           if (data$dim() == 1) data <- data$reshape(c(length(data),1))
           gm <- data
           device <- gm$device


         },
         integer={
           if (is.null(nrow)) nrow=length(data)
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow, ncol, byrow, dimnames)
           gm <- torch::torch_tensor(m,device = device_torch,dtype = torch::torch_int32())
         },
         numeric={

           if (is.null(nrow)) nrow=length(data)
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow, ncol, byrow, dimnames)

           gm <- torch::torch_tensor(m,device = device_torch,dtype = dtype)
         }
  )
  if (is.null(sparse)) sparse <- FALSE


  if(classData == "gpu.matrix.torch"){
    if (!is.null(sparse) & sparse != data@sparse) if(sparse) data <- to_sparse_torch(data)
    if (dtype(data) != writeDType_torch(dtype)) dtype(data) <- dtype
    if (!data@gm$is_cuda & device=="cuda")  data@gm <- data@gm$cuda()
    if (data@gm$is_cuda & device=="cpu")  data@gm <- data@gm$cpu()
    res <- data
  }else{
    if (gm$is_sparse()!=sparse & sparse) {
      gm <- gm$to_sparse()
    }
    if (gm$is_sparse()!=sparse & !sparse) {
      gm <- gm$to_dense()
    }
    res <- new("gpu.matrix.torch", gm=gm, sparse=sparse, colnames=colnames, rownames=rownames, type="torch")
  }
  if(sparseCast) dtype(res) <- dtype
  if (class(charDtype)[[1]] == "torch_dtype") charDtype <- writeDType_torch(charDtype)
  if (dtype(res) != charDtype & !is.null(charDtype)) dtype(res) <- charDtype
  if (is.null(dimnames)){
    dimnames(res) <- dimnames(data)
  }else dimnames(res) <- dimnames

  return(res)
}

gpu.matrix.tensorflow <- function(data = NA, nrow = NULL, ncol = NULL, byrow = FALSE,
                                  dimnames = NULL, dtype=NULL, sparse=FALSE, colnames=c(), rownames=c()){
  if (byrow) ncol=length(data)/nrow
  if (is.null(dtype)){
    dtype <- "float64"
  }
  charDtype <- dtype

  if (class(dtype)[[1]] != "tensorflow.python.framework.dtypes.DType") dtype <- castDtype_tensorflow(dtype)

  classData <- class(data)[1]
  switch(classData,
         matrix={
           gm <- tensorflow::as_tensor(data,dtype = dtype)
         },
         data.frame={
           gm <- tensorflow::as_tensor(as.matrix(data),dtype = dtype)
         },
         dgeMatrix={
           gm <- tensorflow::as_tensor(as.matrix(data),dtype = dtype)

         },
         ddiMatrix={
           data <- as(data,"dgCMatrix")
           i <- data@i
           j <- findInterval(seq(data@x)-1,data@p[-1])
           indices <- tensorflow::as_tensor(lapply(c(1:length(i)),function(x){return(c(i[x],j[x]))}),dtype = tensorflow::tf$int64)
           gm <- tensorflow::tf$SparseTensor(indices = indices, values = data@x, dense_shape = data@Dim)
           gm <- tensorflow::tf$sparse$reorder(gm)
           sparse <- TRUE
         },
         dpoMatrix={
           gm <- tensorflow::as_tensor(as.matrix(data),dtype = dtype)

         },
         dgCMatrix={
           i <- data@i
           j <- findInterval(seq(data@x)-1,data@p[-1])
           indices <- tensorflow::as_tensor(lapply(c(1:length(i)),function(x){return(c(i[x],j[x]))}),dtype = tensorflow::tf$int64)
           gm <- tensorflow::tf$SparseTensor(indices = indices, values = data@x, dense_shape = data@Dim)
           gm <- tensorflow::tf$sparse$reorder(gm)
           sparse <- TRUE
         },
         float32={
           if(requireNamespace("float")){
            gm <- tensorflow::as_tensor(float::dbl(data))
           }
           if (is.null(dtype)) {
             gm <- tensorflow::tf$cast(gm,tensorflow::tf$float32)
           }
           if (!is.null(nrow) & !is.null(ncol)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow,ncol)))
           if (!is.null(nrow)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow,ncol(gm))))
           if (!is.null(ncol)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow(gm),ncol)))

         },
         tensorflow.tensor={
           if (!is.null(dtype) & data$dtype != dtype) {
             data <- tensorflow::tf$cast(data, dtype)
           }
           gm <- data
           if (!is.null(nrow) & !is.null(ncol)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow,ncol)))
           if (!is.null(nrow)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow,ncol(gm))))
           if (!is.null(ncol)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow(gm),ncol)))
           if (class(gm)[2] == "tensorflow.python.framework.sparse_tensor.SparseTensor") {
             sparse <- TRUE
           }
         },
         integer={
           if (is.null(nrow)) nrow=length(data)
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow, ncol, byrow, dimnames)
           gm <- tensorflow::as_tensor(m,dtype = tensorflow::tf$int32)
         },
         numeric={

           if (is.null(nrow)) nrow=length(data)
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow, ncol, byrow, dimnames)

           gm <- tensorflow::as_tensor(m,dtype = dtype)
         }
  )
  if (is.null(sparse)) sparse <- FALSE

  if(classData == "gpu.matrix.tensorflow"){
    if(!is.null(sparse) & sparse != data@sparse) if(sparse) data <- to_sparse_tensorflow(data)
    if (dtype(data) != writeDType_tensorflow(dtype)) dtype(data) <- dtype
    res <- data
  }else{
    if (sparse & class(gm)[2] != "tensorflow.python.framework.sparse_tensor.SparseTensor") {
      gm <- tensorflow::tf$sparse$from_dense(gm)
    }
    if (class(gm)[2] == "tensorflow.python.framework.sparse_tensor.SparseTensor" & !sparse) {
      gm <- tensorflow::tf$sparse$to_dense(gm)
    }

    res <- new("gpu.matrix.tensorflow", gm=gm, sparse=sparse, colnames=colnames, rownames=rownames, type="tensorflow")
  }

  if (class(charDtype)[[1]] == "tensorflow.python.framework.dtypes.DType") charDtype <- writeDType_tensorflow(charDtype)
  if (dtype(res) != charDtype & !is.null(charDtype)) dtype(res) <- charDtype
  if (is.null(dimnames)) dimnames(res) <- dimnames(data)
  else dimnames(res) <- dimnames
  return(res)
}


#' @export
gpu.matrix <- function(data = NA, nrow = NULL, ncol = NULL, byrow = FALSE,
                       dimnames = NULL, dtype=NULL, sparse=NULL, colnames=c(), rownames=c(),device="cuda", type="torch") {
  if (type=="tensorflow") {

    res <- gpu.matrix.tensorflow(data , nrow , ncol , byrow ,
                          dimnames, dtype, sparse,
                          colnames, rownames)

  }else{
    res <- gpu.matrix.torch(data , nrow , ncol , byrow ,
                            dimnames, dtype, sparse,
                            colnames, rownames, device = device)
  }
  return(res)
}

# source("./R/ArithMethods.R")
# source("./R/CompareMethods.R")
# source("./R/IndexingGPUMatrix.R")
# source("./R/setAsMethods.R")
# source("./R/setMethods.R")

