

device <- function(x){
  res <- "cuda"
  if (!x@gm$is_cuda) {
    res <- "cpu"
  }
  return(res)
}
select_rawIndex_torch <- function(A, rawIndex){
  rowsMatrix <- nrow(A)
  colIndex <- c()
  rowIndex <- c()
  for(i in rawIndex){
    colIndex <- c(colIndex,as.integer(ceiling(i/rowsMatrix)))
    valRow <- (i-(rowsMatrix*as.integer(i/rowsMatrix)))
    if (valRow==0) valRow<- rowsMatrix
    rowIndex <- c(rowIndex,valRow)
  }

  index <- list(rowIndex, colIndex)

  return(index)
}

putValuesIndex_torch <- function(x, i, j, values){
  if (typeof(i) == "character") i <- match(i, x@rownames)
  if (typeof(j) == "character") j <- match(j, x@colnames)
  # i <- i-1
  # j <- j-1
  if (length(i) < length(j)) {
    index <- as.matrix(expand.grid(j,i))
    index <- index[ , c(2,1)]
  }else{
    index <- as.matrix(expand.grid(i,j))
  }
  tensor_list <- lapply(1:ncol(index), function(i) torch::torch_tensor(index[,i],dtype = torch::torch_long(),device = device(x)))
  x@gm <- x@gm$index_put(indices=tensor_list, values = torch::torch_tensor(values,dtype = x@gm$dtype,device = device(x)))
  return(x@gm)
}

indexSparse_torch <- function(x, i, j){
  if (typeof(i) == "character") i <- match(i, x@rownames)
  if (typeof(j) == "character") j <- match(j, x@colnames)
  i <- i-1
  j <- j-1
  if (length(i) < length(j)) {
    index <- as.matrix(expand.grid(j,i))
    index <- index[ , c(2,1)]
  }else{
    index <- as.matrix(expand.grid(i,j))
  }
  keyValues <- abs(rnorm(dim(index)[2]))
  index <- index %*% keyValues
  indices <- as.vector(t(gpu.matrix.torch(x@gm$indices())) %*% keyValues)
  matchIndex <- match(index,indices)
  resValues <- as.numeric(x@gm$values()$cpu())[matchIndex]
  resValues[is.na(resValues)] <- 0
  resValues <- torch::torch_tensor(resValues,dtype = x@gm$dtype,device = device(x))

  ni <- c(1:length(i))
  nj <- c(1:length(j))
  if (length(ni) < length(nj)) {
    nindex <- as.matrix(expand.grid(nj,ni))
    nindex <- nindex[ , c(2,1)]
  }else{
    nindex <- as.matrix(expand.grid(ni,nj))
  }
  nindex <- t(gpu.matrix.torch(nindex, device = device(x)))
  dtype(nindex) <- torch::torch_long()

  resIndex <- torch::torch_sparse_coo_tensor(indices = nindex@gm, values = resValues, size = c(length(ni),length(nj)))
  # indices <- torch::torch_tensor(as.matrix(rbind(i,j)))
  # torch::torch_sparse_coo_tensor(indices = indices, values = resValues, size = data@Dim)
  res <- gpu.matrix.torch(resIndex,sparse = T,
                               colnames = x@colnames[j+1],
                               rownames = x@rownames[i+1],
                               dtype = x@gm$dtype, device=device(x))
  return(res)
}

assignValuesSparse_torch <- function(x, i, j, value){
  if (typeof(i) == "character") i <- match(i, x@rownames)
  if (typeof(j) == "character") j <- match(j, x@colnames)
  if (max(i)>nrow(x)) stop(gettextf("invalid row index"), domain=NA)
  if (max(j)>ncol(x)) stop(gettextf("invalid column index"), domain=NA)
  i <- as.integer(i - 1)
  j <- as.integer(j - 1)
  if (length(i) < length(j)) {
    index <- as.matrix(expand.grid(j,i))
    index <- index[ , c(2,1)]
  }else{
    index <- as.matrix(expand.grid(i,j))
  }
  keyValues <- rnorm(dim(index)[2])
  indexU <- index %*% keyValues
  indices <- gpu.matrix(torch::torch_transpose(self = x@gm$indices(),dim0 = 1,dim1 = 2))
  indicesU <- indices %*% keyValues
  matchIndex <- match(as.numeric(indexU),as.numeric(indicesU))
  newValuesIndex <- matchIndex[!is.na(matchIndex)]
  matchIndex[is.na(matchIndex)] <- 0
  replaceValues <- value[as.logical(matchIndex)]
  resValues <- as.numeric(x@gm$values()$cpu())
  resValues[newValuesIndex] <- replaceValues
  resValues <- c(resValues,value[!as.logical(matchIndex)] )
  newValues <- torch::torch_tensor(resValues,dtype = dtype(x), device = device(x))

  catIndex <- index[!as.logical(matchIndex),]
  if (length(catIndex) == 0 ){
    newIndices <- t(indices) + 1
  }else{
    newIndices <- torch::torch_cat(tensors=c(t(indices),torch::torch_tensor(catIndex,dtype = dtype(indices))),dim=1) + 1
  }
  dtype(newIndices) <- torch::torch_long()

  res <- torch::torch_sparse_coo_tensor(indices = newIndices@gm, values = newValues, size = dim(x))
  x@gm <- res$coalesce()
  res <- x
  return(res)
}
assignValues_torch <- function(x,i,j){

}
setMethod("[", signature(x = "gpu.matrix.torch", i = "matrix", j = "missing"),
          function(x,i,j,...){
            x <- as.matrix(x)
            if((na <- nargs()) == 3){
              mIndex <- x[i,]
              res <- gpu.matrix.torch(mIndex, dimnames = dimnames(mIndex))
            }else {
              if((na <- nargs()) == 2){
                # res <- gpu.matrix.torch(suppressWarnings(tf$reshape(tf$transpose(x@gm), as_tensor(c(length(x),1L),shape = c(2L),dtype = tf$int32))[i]))
                res <- x[i]
                return(res)
              }
            }

            return(res)
          })


setMethod("[", signature(x = "gpu.matrix.torch", i = "index", j = "missing"),
          function(x,i,j,...){
            # x <- as.matrix(x)
            if (x@sparse) {
              if((na <- nargs()) == 3){
                res <- indexSparse_torch(x,i,j=c(1:ncol(x)))

              }else{
                if ((na <- nargs()) == 2) {
                    listIndex <- select_rawIndex_torch(x,i)
                    index <- cbind(listIndex[[1]]-1,listIndex[[2]]-1)
                    keyValues <- rnorm(dim(index)[2])
                    index <- index %*% keyValues
                    indices <- t(gpu.matrix.torch(x@gm$indices())) %*% keyValues
                    matchIndex <- match(index,as.numeric(indices@gm$cpu()))
                    resValues <- as.numeric(x@gm$values()$cpu())[matchIndex]
                    resValues[is.na(resValues)] <- 0
                    res <- resValues

                }
              }
            }else{
              if((na <- nargs()) == 3){
                if (typeof(i) == "character") i <- match(i, x@rownames)
                if (length(i)>1) {
                  res <- gpu.matrix.torch(x@gm[i,],colnames=colnames(x), rownames = rownames(x)[i])
                }else{
                  res <- as.numeric(x@gm[i,]$cpu())
                }

              }else {
                if((na <- nargs()) == 2){
                  vecSearch <- t(x)@gm$reshape(length(x))
                  if (typeof(i) ==  "logical"){
                    i <- c(1:length(vecSearch))[i]
                  }
                  res <- as.numeric(vecSearch[i]$cpu())


                }
              }
            }

            return(res)
          })

setMethod("[[", signature(x = "gpu.matrix.torch", i = "index"),
          function(x,i,...){
            if (typeof(i) == "character") i <- match(i, x@rownames)
            listIndex <- select_rawIndex_torch(x,i)
            index <- cbind(listIndex[[1]]-1,listIndex[[2]]-1)
            keyValues <- rnorm(dim(index)[2])
            index <- index %*% keyValues
            indices <- t(gpu.matrix.torch(x@gm$indices())) %*% keyValues
            matchIndex <- match(index,as.numeric(indices@gm$cpu()))
            resValues <- as.numeric(x@gm$values()$cpu())[matchIndex]
            resValues[is.na(resValues)] <- 0
            res <- resValues

            return(res)
          })

setMethod("[", signature(x = "gpu.matrix.torch", i = "missing", j = "index"),
          function (x, i, j) {
            if (x@sparse) {
              res <- indexSparse_torch(x,i=c(1:nrow(x)),j)
            }else{
              if (typeof(j) == "character") j <- match(j, x@colnames)
              res <- gpu.matrix.torch(x@gm[,j])
              rownames(res) <- x@rownames

              colnames(res)<- x@colnames[j]
            }

            return(res)
          })

setMethod("[", signature(x = "gpu.matrix.torch", i = "index", j = "index"),
          function (x, i, j) {
            if (x@sparse) {
              x <- indexSparse_torch(x,i,j)
            }else{
              if (typeof(i) == "character") i <- match(i, x@rownames)
              if (typeof(j) == "character") j <- match(j, x@colnames)
              x@gm <- x@gm[i,j]
              # dim(x) <- c(length(i),length(j))
              x@rownames <- x@rownames[i]
              x@colnames <- x@colnames[j]
              x@gm <- x@gm$reshape(c(length(i),length(j)))
            }

            return(x)
          })



setReplaceMethod("[", signature(x = "gpu.matrix.torch", i = "index", j = "missing",
                                value = "ANY"),
                 function (x, i, j, ..., value) {
                   value <- as.vector(value)
                   if(x@sparse){
                     if((na <- nargs()) == 3){
                       if (max(i)>length(x)) stop(gettextf("invalid index"), domain=NA)
                       listIndex <- select_rawIndex_torch(x,i)
                       index <- cbind(listIndex[[1]],listIndex[[2]]) - 1
                       keyValues <- rnorm(dim(index)[2])
                       indexU <- index %*% keyValues
                       indices <- gpu.matrix(torch::torch_transpose(self = x@gm$indices(),dim0 = 1,dim1 = 2))
                       indicesU <- as.numeric(indices %*% matrix(keyValues))
                       matchIndex <- match(indexU,indicesU)
                       newValuesIndex <- matchIndex[!is.na(matchIndex)]
                       matchIndex[is.na(matchIndex)] <- 0
                       replaceValues <- value[as.logical(matchIndex)]
                       resValues <- as.numeric(x@gm$values()$cpu())
                       resValues[newValuesIndex] <- replaceValues
                       resValues <- c(resValues,value[!as.logical(matchIndex)] )
                       newValues <- torch::torch_tensor(resValues,dtype = x@gm$values()$dtype, device = device(x))
                       catIndex <- index[!as.logical(matchIndex),]
                       if (length(catIndex) == 0 ){
                         newIndices <- t(indices) + 1
                       }else{
                         newIndices <- torch::torch_cat(tensors=c(t(indices),torch::torch_tensor(catIndex,dtype = dtype(indices))),dim=1) + 1
                       }
                       dtype(newIndices) <- torch::torch_long()
                       res <- torch::torch_sparse_coo_tensor(indices = newIndices@gm, values = newValues, size = dim(x))
                       x@gm <- res$coalesce()
                       res <- x
                     }else if(na == 4){
                       res <- assignValuesSparse_torch(x, i, j=c(1:ncol(x)), value)

                     }


                   }else{
                     if((na <- nargs()) == 3){

                       listIndex <- select_rawIndex_torch(x,i)
                       index <- cbind(listIndex[[1]],listIndex[[2]])
                       tensor_list <- lapply(1:ncol(index), function(i) torch::torch_tensor(index[,i],dtype = torch::torch_long(),device = device(x)))
                       x@gm <- x@gm$index_put(indices=tensor_list, value = torch::torch_tensor(value,dtype = dtype(x),device = device(x)))
                       res <- x

                     }else if(na == 4){
                       x@gm <- putValuesIndex_torch(x,i,1:ncol(x),value)

                       res <- x
                     }else stop(gettextf("invalid nargs()= %d", na), domain=NA)
                   }


                   return(res)
                 })

setReplaceMethod("[", signature(x = "gpu.matrix.torch", i = "matrix", j = "missing",
                                value = "ANY"),## double/logical/...
                 function (x, i, j, ..., value) {
                   value <- as.vector(value)
                   res <- as.matrix(x)
                   if((na <- nargs()) == 3)
                     res[i] <- value
                   else if(na == 4)
                     res[i, ] <- value
                   else stop(gettextf("invalid nargs()= %d", na), domain=NA)

                   return(gpu.matrix.torch(res))
                 })

setReplaceMethod("[", signature(x = "gpu.matrix.torch", i = "missing", j = "index",
                                value = "ANY"),## double/logical/...
                 function (x, i, j, ..., value) {
                   value <- as.vector(value)
                   if (x@sparse){
                     res <- assignValuesSparse_torch(x, i=c(1:nrow(x)), j, value)

                   }else{
                     if (typeof(j) == "character") j <- match(j, x@colnames)
                     x@gm <- putValuesIndex_torch(x,1:nrow(x),j,value)
                     res <- x
                   }


                   return(res)
                 })

setReplaceMethod("[", signature(x = "gpu.matrix.torch", i = "index", j = "index",
                                value = "ANY"),## double/logical/...
                 function (x, i, j, ..., value) {
                   value <- as.vector(value)
                   if (x@sparse) {

                     res <- assignValuesSparse_torch(x, i, j, value)


                   }else{
                     x@gm <- putValuesIndex_torch(x,i,j,value)
                     res <- x
                   }


                   return(res)
                 })

setReplaceMethod("[[", signature(x = "gpu.matrix.torch", i = "index",
                                 value = "ANY"),## double/logical/...
                 function (x, i, ..., value) {
                   listIndex <- select_rawIndex_torch(x,i)
                   index <- cbind(listIndex[[1]],listIndex[[2]])
                   tensor_list <- lapply(1:ncol(index), function(i) torch::torch_tensor(index[,i],dtype = torch::torch_long(),device = device(x)))
                   x@gm <- x@gm$index_put(indices=tensor_list, value = torch::torch_tensor(value,dtype = dtype(x),device = device(x)))
                   res <- x

                   return(res)
                 })
