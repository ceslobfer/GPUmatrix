
# Define a class for a GPU matrix using TensorFlow
setClass("gpu.matrix.tensorflow",
         representation(
           # data="vector",
           rownames = "ANY",
           colnames = "ANY",
           gm = "ANY",
           sparse = "ANY",
           type="character")
)
# Define a class for a GPU matrix using PyTorch
setClass("gpu.matrix.torch",
         representation(
           # data="vector",
           rownames = "ANY",
           colnames = "ANY",
           gm = "ANY",
           sparse = "ANY",
           type="character")
)

# Function to cast data type for PyTorch
castDtype_torch <- function(type,data=NULL) {
  if(requireNamespace('torch')){
    switch(type,
           "float32" = {
             res <- torch::torch_float32()
           },
           "float64" = {
             res <- torch::torch_float64()
           },
           "double" = {
             res <- torch::torch_float64()
           },
           "int" = {
             res <- torch::torch_int()
           },
           "integer" = {
             res <- torch::torch_int()
           },
           "bool" = {
             res <- torch::torch_bool()
           },
           "logical" = {
             res <- torch::torch_bool()
           },
           "complex64"={
             res <- torch::torch_cfloat64()
           },
           "complex"={
             res <- torch::torch_cfloat64()
           },
           "complex32"={
             res <- torch::torch_cfloat32()
           },
           "externalptr"={
             if(class(data)[1]=="torch_tensor"){
               res <- data$dtype
             }
             if(class(data)[1]=="torch_dtype"){
               res <- data
             }

           },
           NULL={
             res <- torch::torch_float64()
           },
           "logical" = {
             res <- torch::torch_bool()
           },
           "S4" = {
             if(class(data)[1]=="gpu.matrix.torch"){
               res <- data@gm$dtype
             }else{
               res <- castDtype_torch(typeof(data[1]))
             }

           },
           stop("Invalid input type")
    )
  }
  return(res)
}
# Function to cast data type for TensorFlow
castDtype_tensorflow <- function(type,data=NULL) {
  switch(type,
         "float32" = {
           res <- tensorflow::tf$float32
         },
         "float64" = {
           res <- tensorflow::tf$float64
         },
         "double" = {
           res <- tensorflow::tf$float64
         },
         "int" = {
           res <- tensorflow::tf$int64
         },
         "integer" = {
           res <- tensorflow::tf$int64
         },
         "bool" = {
           res <- tensorflow::tf$bool
         },
         "logical" = {
           res <- tensorflow::tf$bool
         },
         "complex"={
           res <- tensorflow::tf$complex64
         },
         "complex64"={
           res <- tensorflow::tf$complex64
         },
         "externalptr"={
           if(class(data)[1]=="tensorflow.tensor"){
             res <- data$dtype
           }
           if(class(data)[1]=="tensorflow.python.framework.dtypes.DType"){
             res <- data
           }
         },
         "environment"={
           if(class(data)[1]=="tensorflow.tensor"){
             res <- data$dtype
           }
           if(class(data)[1]=="tensorflow.python.framework.dtypes.DType"){
             res <- data
           }
         },
         NULL={
           res <- tensorflow::tf$float64
         },
         "S4" = {
           if(class(data)[1]=="gpu.matrix.tensorflow"){
             res <- data@gm$dtype
           }else{
             res <- castDtype_tensorflow(typeof(data[1]))
           }

         },
         "logical" = {
           res <- tensorflow::tf$bool
         },
         stop("Invalid input type")
  )
  return(res)
}

# Function to cast a Matrix to a PyTorch sparse tensor
dMatrixCast_torch <- function(data,device_torch){
  i <- data@i + 1
  j <- findInterval(seq(data@x)-1,data@p[-1]) + 1
  indices <- torch::torch_tensor(as.matrix(rbind(i,j)),dtype=torch::torch_long())
  gm <- torch::torch_sparse_coo_tensor(indices = indices, values = data@x, size = data@Dim)
  gm <- gm$coalesce()
  if (!gm$is_cuda & device_torch=="cuda")  gm <- gm$cuda()
  return(gm)
}
# Function to cast a Matrix to a TensorFlow sparse tensor
dMatrixCast_tensorflow <- function(data){
  i <- data@i
  j <- findInterval(seq(data@x)-1,data@p[-1])
  indices <- tensorflow::as_tensor(lapply(c(1:length(i)),function(x){return(c(i[x],j[x]))}),dtype = tensorflow::tf$int64)
  gm <- tensorflow::tf$SparseTensor(indices = indices, values = data@x, dense_shape = data@Dim)
  gm <- tensorflow::tf$sparse$reorder(gm)
  return(gm)
}
# Function to control dimensions of a tensor for PyTorch
dimControl_torch <- function(gm,nrow,ncol){
  if (!is.null(nrow) & !is.null(ncol)) gm$resize_(c(nrow, ncol))
  if (!is.null(nrow)) gm$resize_(c(nrow,ncol(gm)))
  if (!is.null(ncol)) gm$resize_(c(nrow(gm), ncol))

  return(gm)
}
# Function to control dimensions of a tensor for TensorFlow
dimControl_tensorflow <- function(gm,nrow,ncol){
  if (!is.null(nrow) & !is.null(ncol)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow,ncol)))
  if (!is.null(nrow)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow,ncol(gm))))
  if (!is.null(ncol)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow(gm),ncol)))
  return(gm)
}

# Main function to create a GPU matrix using PyTorch
gpu.matrix.torch <- function(data = NULL, nrow = NULL, ncol = NULL, byrow = FALSE,
                                  dimnames = NULL, dtype=NULL, sparse=NULL, colnames=c(), rownames=c(),device=NULL){
  if (byrow) ncol=length(data)/nrow

  classData <- class(data)[1]

  #dtype control
  if (class(dtype)[[1]] != "torch_dtype"){
    # dtype <- castDtype_torch(typeof(data), data)
    if (is.null(dtype) & classData == "float32") dtype <- "float32"
    if (is.null(dtype)){
      dtype <- castDtype_torch(typeof(data),data)
    }else{
      dtype <- castDtype_torch(dtype)
    }
  }

  #device control
  if (is.null(device)){
    if(torch::cuda_is_available()){
      device <- "cuda"
    }else{
      # warning("Your Torch installation does not have CUDA tensors available. Please check the Torch requirements and installation if you want to use CUDA tensors.")
      device <- "cpu"
    }
  }
  device_torch <- torch::torch_device(type = device)

  switch(classData,
         matrix={
           gm <- torch::torch_tensor(data, device = device_torch,dtype = dtype)
           gm <- dimControl_torch(gm, nrow = nrow, ncol = ncol)
         },
         data.frame={
           gm <- torch::torch_tensor(as.matrix(data), device = device_torch,dtype = dtype)
           gm <- dimControl_torch(gm, nrow = nrow, ncol = ncol)
         },
         dgeMatrix={
           gm <- torch::torch_tensor(as.matrix(data), device = device_torch,dtype = dtype)
         },
         ddiMatrix={
           data <- as(data,"dgCMatrix")
           gm <- dMatrixCast_torch(data, device)
           gm <- dimControl_torch(gm, nrow = nrow, ncol = ncol)
         },
         dpoMatrix={
           gm <- torch::torch_tensor(as.matrix(data), device = device,dtype = dtype)
           gm <- dimControl_torch(gm, nrow = nrow, ncol = ncol)
         },
         dgCMatrix={
           gm <- dMatrixCast_torch(data, device)
           gm <- dimControl_torch(gm,nrow,ncol)
         },
         float32={
           if(requireNamespace("float")){
            gm <- torch::torch_tensor(float::dbl(data), device = device_torch,dtype = dtype)
           }
           gm <- dimControl_torch(gm,nrow,ncol)

         },
         torch_tensor={
           if (data$dim() == 1) data <- data$reshape(c(length(data), 1))
           gm <- torch::torch_tensor(data,device=device_torch, dtype=dtype)
           device <- gm$device
           gm <- dimControl_torch(gm,nrow,ncol)
         },
         gpu.matrix.torch={
           gm <- torch::torch_tensor(data@gm, device=device_torch, dtype=dtype)
           gm <- dimControl_torch(gm,nrow,ncol)
         },
         integer={
           if (is.null(nrow) & is.null(ncol)) nrow=length(data)
           if (is.null(nrow)) nrow=length(data)/ncol
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow=nrow, ncol=ncol, byrow=byrow)
           gm <- torch::torch_tensor(m, device = device_torch, dtype = dtype)
         },
         numeric={
           if (is.null(nrow) & is.null(ncol)) nrow=length(data)
           if (is.null(nrow)) nrow=length(data)/ncol
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow=nrow, ncol=ncol, byrow=byrow)
           gm <- torch::torch_tensor(m, device = device_torch, dtype = dtype)
         },
         logical={
           if (is.null(nrow) & is.null(ncol)) nrow=length(data)
           if (is.null(nrow)) nrow=length(data)/ncol
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow=nrow, ncol=ncol, byrow=byrow)
           gm <- torch::torch_tensor(m, device = device_torch, dtype = dtype)
         },
         NULL={
           if (is.null(nrow) | nrow ==0) nrow=1
           if (is.null(ncol) | ncol ==0) ncol=1
           m <- matrix(NaN,nrow=nrow, ncol=ncol, byrow=byrow)
           gm <- torch::torch_tensor(m, device = device_torch, dtype = dtype)
         },
         complex={
           if (is.null(nrow) & is.null(ncol)) nrow=length(data)
           if (is.null(nrow)) nrow=length(data)/ncol
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow=nrow, ncol=ncol, byrow=byrow)
           gm <- torch::torch_tensor(m, device = device_torch, dtype = dtype)
         }
  )


  res <- new("gpu.matrix.torch", gm=gm, sparse=sparse, type="torch")

  #sparse Control
  if (!is.null(sparse)) {
    if(res@gm$is_sparse()!=sparse & sparse==T) res <- to_sparse_torch(res)
    if(res@gm$is_sparse()!=sparse & sparse==F) res@gm <- res@gm$to_dense()
  }else{
    sparse <- res@gm$is_sparse()
    res@sparse <- sparse
  }

  # res <- new("gpu.matrix.torch", gm=gm, sparse=sparse, type="torch")

  #dimnames Control
  if (!is.null(dimnames)){
    dimnames(res) <- dimnames
  }else{
    dimnames(res) <- dimnames(data)
  }

  if (!is.null(rownames)){
    rownames(res) <- rownames
  }

  if (!is.null(colnames)){
    colnames(res) <- colnames
  }

  return(res)
}
# Main function to create a GPU matrix using TensorFlow
gpu.matrix.tensorflow <- function(data = NA, nrow = NULL, ncol = NULL, byrow = FALSE,
                                  dimnames = NULL, dtype=NULL, sparse=NULL, colnames=c(), rownames=c()){

  if (byrow) ncol=length(data)/nrow
  classData <- class(data)[1]

  #dtype control
  if (class(dtype)[[1]] != "tensorflow.python.framework.dtypes.DType"){
    # dtype <- castDtype_torch(typeof(data), data)
    if (is.null(dtype) & classData == "float32") dtype <- "float32"
    if (is.null(dtype)){
      dtype <- castDtype_tensorflow(typeof(data),data)
    }else{
      dtype <- castDtype_tensorflow(dtype)
    }
  }

  switch(classData,
         matrix={
           # data <- matrix(data,nrow,ncol,byrow)
           gm <- tensorflow::as_tensor(data,dtype = dtype)
           gm <- dimControl_tensorflow(gm,nrow,ncol)
         },
         data.frame={
           gm <- tensorflow::as_tensor(as.matrix(data), dtype = dtype)
           gm <- dimControl_tensorflow(gm,nrow,ncol)
         },
         dgeMatrix={
           gm <- tensorflow::as_tensor(as.matrix(data), dtype = dtype)
           gm <- dimControl_tensorflow(gm,nrow,ncol)
         },
         ddiMatrix={
           data <- as(data,"dgCMatrix")
           gm <- dMatrixCast_tensorflow(data)
           sparse <- TRUE
           gm <- dimControl_tensorflow(gm,nrow,ncol)
         },
         dpoMatrix={
           gm <- tensorflow::as_tensor(as.matrix(data),dtype = dtype)
           gm <- dimControl_tensorflow(gm,nrow,ncol)

         },
         dgCMatrix={
           gm <- dMatrixCast_tensorflow(data)
           sparse <- TRUE
           gm <- dimControl_tensorflow(gm,nrow,ncol)

         },
         float32={
           if(requireNamespace("float")){
             gm <- tensorflow::as_tensor(float::dbl(data),dtype)
           }
           gm <- dimControl_tensorflow(gm,nrow,ncol)
         },
         tensorflow.tensor={
           if (!is.null(dtype) & data$dtype != dtype) {
             data <- tensorflow::tf$cast(data, dtype)
           }
           gm <- data
           if (class(gm)[2] == "tensorflow.python.framework.sparse_tensor.SparseTensor") {
             sparse <- TRUE
           }
         },
         NULL={
           if (is.null(nrow) | nrow ==0) nrow=1
           if (is.null(ncol) | ncol ==0) ncol=1
           m <- matrix(NaN,nrow, ncol, byrow, dimnames)
           gm <- tensorflow::as_tensor(m,dtype = dtype)
         },
         integer={
           if (is.null(nrow) & is.null(ncol)) nrow=length(data)
           if (is.null(ncol)) ncol=length(data)/nrow
           if (is.null(nrow)) nrow=length(data)/ncol
           m <- matrix(data, nrow, ncol, byrow, dimnames)
           gm <- tensorflow::as_tensor(m,dtype = dtype)
         },
         gpu.matrix.tensorflow={
           gm <- data@gm
           if (gm$dtype != dtype) {
             gm <- tensorflow::tf$cast(gm, dtype)
           }
           if (!is.null(nrow) & !is.null(ncol)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow,ncol)))
           if (!is.null(nrow)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow,ncol(gm))))
           if (!is.null(ncol)) gm <- tensorflow::tf$reshape(gm, as.integer(c(nrow(gm),ncol)))
           if (class(gm)[2] == "tensorflow.python.framework.sparse_tensor.SparseTensor") {
             sparse <- TRUE
           }
         },
         numeric={
           if (is.null(nrow) & is.null(ncol)) nrow=length(data)
           if (is.null(ncol)) ncol=length(data)/nrow
           if (is.null(nrow)) nrow=length(data)/ncol
           m <- matrix(data, nrow, ncol, byrow, dimnames)
           gm <- tensorflow::as_tensor(m,dtype = dtype)
         },
         complex={
           if (is.null(nrow) & is.null(ncol)) nrow=length(data)
           if (is.null(nrow)) nrow=length(data)/ncol
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow, ncol, byrow, dimnames)
           gm <- tensorflow::as_tensor(m,dtype)
         },
         logical={
           if (is.null(nrow) & is.null(ncol)) nrow=length(data)
           if (is.null(nrow)) nrow=length(data)/ncol
           if (is.null(ncol)) ncol=length(data)/nrow
           m <- matrix(data, nrow, ncol, byrow, dimnames)
           gm <- tensorflow::as_tensor(m, dtype)
         }
  )
  if (is.null(sparse)) sparse <- FALSE


  #Control SPARSE
  if(!is.null(sparse)){
    if(sparse & class(gm)[2] != "tensorflow.python.framework.sparse_tensor.SparseTensor"){
      gm <- tensorflow::tf$sparse$from_dense(gm)
    }
    if (class(gm)[2] == "tensorflow.python.framework.sparse_tensor.SparseTensor" & !sparse) {
      gm <- tensorflow::tf$sparse$to_dense(gm)
    }
  }

  res <- new("gpu.matrix.tensorflow", gm=gm, sparse=sparse, type="tensorflow")


  #dimnames Control
  if (!is.null(dimnames)){
    dimnames(res) <- dimnames
  }else{
    dimnames(res) <- dimnames(data)
  }

  if (!is.null(rownames)){
    rownames(res) <- rownames
  }

  if (!is.null(colnames)){
    colnames(res) <- colnames
  }

  return(res)
}

# Main function to create a GPU matrix using either PyTorch or TensorFlow
#' @export
gpu.matrix <- function(data = NULL, nrow = NULL, ncol = NULL, byrow = FALSE,
                       dimnames = NULL, dtype=NULL, sparse=NULL, colnames=c(), rownames=c(),device=NULL, type=NULL) {
  if (is.null(type)) type <- getOption("typeTensor")
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



