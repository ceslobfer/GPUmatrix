setClassUnion("index",
              members = c("logical", "numeric", "character"))
select_rows <- function(A,row_to_show){


  if (typeof(row_to_show) == "character") row_to_show <- match(row_to_show, A@rownames)


  if(length(row_to_show) == length(unique(row_to_show))) {
    gm <- A@gm[row_to_show,]

  }else{
    row_to_show <- row_to_show - 1
    gm <- tensorflow::tf$gather_nd(A@gm, indices = lapply(as.integer(row_to_show),function(x){list(x)}))
  }
  # gm <- tensorflow::tf$gather_nd(A@gm, indices = lapply(as.integer(row_to_show),function(x){list(x)}))
  # res <- gpu.matrix.tensorflow(gm , rownames = A@rownames[row_to_show], colnames = A@colnames)

  A@gm <- gm
  A@rownames <- A@rownames[row_to_show]

  return(A)

}

select_cols <- function(A,col_to_show){
  #

  if (typeof(col_to_show) == "character") col_to_show <- match(col_to_show, A@colnames)


  if(length(col_to_show) == length(unique(col_to_show))) {
    # New version
    gm <- A@gm[,col_to_show]
    A@gm <- gm


    # End new version
  }else{
    col_to_show <- col_to_show - 1
    A <- t(A)
    gm <- tensorflow::tf$gather_nd(A@gm, indices = lapply(as.integer(col_to_show),function(x){list(x)}))
    # res <- gpu.matrix.tensorflow(gm, rownames = A@rownames[col_to_show], colnames = A@colnames)
    A@gm <- gm
    A <- t(A)

  }
  A@colnames <- A@colnames[col_to_show]

  return(A)

}

select_rawIndex <- function(A, rawIndex){
  # rowsMatrix <- nrow(A)
  # index <- lapply(rawIndex,function(y,nrows=rowsMatrix){
  #   colIndex <- as.integer(ceiling(y/nrows) - 1)
  #   if (y > nrows){
  #     rowIndex <- as.integer(y/(nrows))
  #   }else{
  #     rowIndex <- as.integer(y-1)
  #   }
  #   list(rowIndex, colIndex)
  # })

  rowsMatrix <- nrow(A)
  colIndex <- c()
  rowIndex <- c()
  for(i in rawIndex){
    colIndex <- c(colIndex,as.integer(ceiling(i/rowsMatrix)))
    valRow <- (i-(rowsMatrix*as.integer(i/rowsMatrix)))
    if (valRow==0) valRow<- rowsMatrix
    rowIndex <- c(rowIndex,valRow)
  }

  index <- list(as.integer(rowIndex-1), as.integer(colIndex-1))
  return(index)
}

indexSparse <- function(x, i, j){
  if (typeof(i) == "character") i <- match(i, x@rownames)
  if (typeof(j) == "character") j <- match(j, x@colnames)
  i <- as.integer(i - 1)
  j <- as.integer(j - 1)
  if (length(i) < length(j)) {
    index <- as.matrix(expand.grid(j,i))
    index <- index[ , c(2,1)]
  }else{
    index <- as.matrix(expand.grid(i,j))
  }
  keyValues <- rnorm(dim(index)[2])
  index <- index %*% keyValues
  indices <- as.matrix(x@gm$indices) %*% keyValues
  matchIndex <- match(index,indices)
  resValues <- as.vector(x@gm$values)[matchIndex]
  resValues[is.na(resValues)] <- 0
  res <- gpu.matrix.tensorflow(matrix(resValues,nrow = length(i),
                           ncol = length(j)),sparse = T,
                    colnames = x@colnames[j+1],
                    rownames = x@rownames[i+1],
                    dtype = x@gm$dtype)
  return(res)
}

assignValuesSparse <- function(x, i, j, value){
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
  indices <- x@gm$indices
  indicesU <- as.matrix(indices) %*% keyValues
  matchIndex <- match(indexU,indicesU)
  newValuesIndex <- matchIndex[!is.na(matchIndex)]
  matchIndex[is.na(matchIndex)] <- 0
  replaceValues <- value[as.logical(matchIndex)]
  resValues <- as.vector(x@gm$values)
  resValues[newValuesIndex] <- replaceValues
  resValues <- c(resValues,value[!as.logical(matchIndex)] )
  newValues <- tensorflow::as_tensor(resValues,dtype = x@gm$values$dtype)

  catIndex <- index[!as.logical(matchIndex),]
  if (length(catIndex) <= 2 ){
    catIndex <- list(catIndex)
  }
  newIndices <- tensorflow::tf$concat(c(indices,tensorflow::as_tensor(catIndex,dtype = indices$dtype)),0L)
  res <- tensorflow::tf$SparseTensor(indices = newIndices,
                         values = newValues,
                         dense_shape = c(nrow(x),ncol(x)))
  res <- tensorflow::tf$sparse$reorder(res)
  return(res)
}
setMethod("[", signature(x = "gpu.matrix.tensorflow", i = "matrix", j = "missing"),
          function(x,i,j,...){
            x <- as.matrix(x)
            if((na <- nargs()) == 3){
              mIndex <- x[i,]
              res <- gpu.matrix.tensorflow(mIndex, dimnames = dimnames(mIndex))
            }else {
              if((na <- nargs()) == 2){
                # res <- gpu.matrix.tensorflow(suppressWarnings(tensorflow::tf$reshape(tensorflow::tf$transpose(x@gm), tensorflow::as_tensor(c(length(x),1L),shape = c(2L),dtype = tensorflow::tf$int32))[i]))
                res <- x[i]
                return(res)
              }
            }

            return(res)
          })


setMethod("[", signature(x = "gpu.matrix.tensorflow", i = "index", j = "missing"),
          function(x,i,j,...){
            # x <- as.matrix(x)
            if (x@sparse) {
              if((na <- nargs()) == 3){
                res <- indexSparse(x,i,j=c(1:ncol(x)))

              }else{
                if ((na <- nargs()) == 2) {
                  index <- select_rawIndex(x,i)
                  index <- do.call(cbind, index)
                  keyValues <- rnorm(dim(index)[2])
                  index <- index %*% keyValues
                  indices <- as.matrix(x@gm$indices) %*% keyValues
                  matchIndex <- match(index,indices)
                  resValues <- as.vector(x@gm$values)[matchIndex]
                  resValues[is.na(resValues)] <- 0
                  res <- resValues
                }
              }
            }else{
              if((na <- nargs()) == 3){
                res <- select_rows(x,i)

              }else {
                if((na <- nargs()) == 2){
                  if (typeof(i) ==  "logical"){
                    vecSearch <- as.numeric(x)
                    i <- c(1:length(vecSearch))[i]
                    res <- vecSearch[i]
                  }else{
                    index <- do.call(cbind,select_rawIndex(x,i))
                    res <- as.vector(tensorflow::tf$gather_nd(x@gm, indices = index))
                  }

                }
              }
            }

            return(res)
          })

setMethod("[[", signature(x = "gpu.matrix.tensorflow", i = "index"),
          function(x,i,...){
            index <- do.call(cbind,select_rawIndex(x,i))
            res <- as.vector(tensorflow::tf$gather_nd(x@gm, indices = index))

            return(res)
          })

setMethod("[", signature(x = "gpu.matrix.tensorflow", i = "missing", j = "index"),
          function (x, i, j) {
            if (x@sparse) {
              res <- indexSparse(x,i=c(1:nrow(x)),j)
            }else{
              res <- select_cols(x,j)
            }

            return(res)
          })

#' @name [
#' @aliases [,gpu.matrix.tensorflow-method
#' @docType methods
#' @rdname extract-methods
setMethod("[", signature(x = "gpu.matrix.tensorflow", i = "index", j = "index"),
          function (x, i, j) {
            if (x@sparse) {
              x <- indexSparse(x,i,j)
            }else{
              if((length(i) == length(unique(i))) & (length(j) == length(unique(j)))){
                x@gm <- x@gm[i,j]
                x@rownames <- x@rownames[i]
                x@colnames <- x@colnames[j]
              }else{
                x <- select_rows(x,i)
                if (length(i) == 1) dim(res) <- c(1,length(res))
                x <- select_cols(x,j)
              }

            }

            return(x)
          })

setReplaceMethod("[", signature(x = "gpu.matrix.tensorflow", i = "index", j = "missing",
                                value = "ANY"),
                 function (x, i, j, ..., value) {
                   value <- as.vector(value)
                   if(x@sparse){
                     if((na <- nargs()) == 3){
                       if (max(i)>length(x)) stop(gettextf("invalid index"), domain=NA)
                       index <- do.call(cbind,select_rawIndex(x,i))
                       keyValues <- rnorm(dim(index)[2])
                       indexU <- index %*% keyValues
                       indices <- x@gm$indices
                       indicesU <- as.matrix(indices) %*% keyValues
                       matchIndex <- match(indexU,indicesU)
                       newValuesIndex <- matchIndex[!is.na(matchIndex)]
                       matchIndex[is.na(matchIndex)] <- 0
                       replaceValues <- value[as.logical(matchIndex)]
                       resValues <- as.vector(x@gm$values)
                       resValues[newValuesIndex] <- replaceValues
                       resValues <- c(resValues,value[!as.logical(matchIndex)] )
                       newValues <- tensorflow::as_tensor(resValues,dtype = x@gm$values$dtype)
                       catIndex <- index[!as.logical(matchIndex),]
                       if (length(catIndex) <= 2 ){
                         catIndex <- list(catIndex)
                       }
                       newIndices <- tensorflow::tf$concat(c(indices,tensorflow::as_tensor(catIndex,dtype = indices$dtype)),0L)
                       res <- tensorflow::tf$SparseTensor(indices = newIndices,
                                              values = newValues,
                                              dense_shape = c(nrow(x),ncol(x)))
                       x@gm <- tensorflow::tf$sparse$reorder(res)
                       res <- x
                     }else if(na == 4){
                       x@gm <- assignValuesSparse(x, i, j=c(1:ncol(x)), value)
                       res <- x
                     }


                   }else{
                     if((na <- nargs()) == 3){
                       if (length(i) == 1) value <- list(value)

                       index <- do.call(cbind,select_rawIndex(x,i))
                       res <- tensorflow::tf$tensor_scatter_nd_update(
                         x@gm,
                         indices = index,
                         updates = value
                       )
                       res <- gpu.matrix.tensorflow(res, dimnames = dimnames(x))
                     }else if(na == 4){
                       if (typeof(i) == "character") i <- match(i, x@rownames)
                       i <- i - 1
                       indices = lapply(as.integer(i),function(y){list(y)})
                       res <- tensorflow::tf$tensor_scatter_nd_update(
                         x@gm,
                         indices = indices,
                         updates = matrix(value, ncol = ncol(x))
                       )
                       res <- gpu.matrix.tensorflow(res, dimnames = dimnames(x))
                     }else stop(gettextf("invalid nargs()= %d", na), domain=NA)
                   }


                   return(res)
                 })

setReplaceMethod("[", signature(x = "gpu.matrix.tensorflow", i = "matrix", j = "missing",
                                value = "ANY"),## double/logical/...
                 function (x, i, j, ..., value) {
                   value <- as.vector(value)
                   res <- as.matrix(x)
                   if((na <- nargs()) == 3)
                     res[i] <- value
                   else if(na == 4)
                     res[i, ] <- value
                   else stop(gettextf("invalid nargs()= %d", na), domain=NA)

                   return(gpu.matrix.tensorflow(res))
                 })

setReplaceMethod("[", signature(x = "gpu.matrix.tensorflow", i = "missing", j = "index",
                                value = "ANY"),## double/logical/...
                 function (x, i, j, ..., value) {
                   value <- as.vector(value)
                   if (x@sparse){
                     x@gm <- assignValuesSparse(x, i=c(1:nrow(x)), j, value)
                     res <- x
                   }else{
                     if (typeof(j) == "character") j <- match(j, x@colnames)
                     j <- j - 1
                     indices = lapply(as.integer(j),function(y){list(y)})
                     x <- t(x)
                     res <- tensorflow::tf$tensor_scatter_nd_update(
                       x@gm,
                       indices = indices,
                       updates = matrix(value, ncol = ncol(x))
                     )
                     res <- gpu.matrix.tensorflow(res, dimnames = dimnames(x))
                   }


                   return(t(res))
                 })

setReplaceMethod("[", signature(x = "gpu.matrix.tensorflow", i = "index", j = "index",
                                value = "ANY"),## double/logical/...
                 function (x, i, j, ..., value) {
                   value <- as.vector(value)
                   if (x@sparse) {

                     x@gm <- assignValuesSparse(x, i, j, value)
                     res <- x

                   }else{
                     if (typeof(i) == "character") i <- match(i, x@rownames)
                     if (typeof(j) == "character") j <- match(j, x@colnames)
                     i <- as.integer(i - 1)
                     j <- as.integer(j - 1)
                     if (length(i) < length(j)) {
                       index <- as.matrix(expand.grid(j,i))
                       index <- index[ , c(2,1)]
                     }else{
                       index <- as.matrix(expand.grid(i,j))
                     }
                     if (length(i) == 1) value <- list(value)
                     res <- tensorflow::tf$tensor_scatter_nd_update(
                       x@gm,
                       indices = tensorflow::as_tensor(index),
                       updates = value
                     )

                     res <- gpu.matrix.tensorflow(res, dimnames = dimnames(x))
                   }


                   return(res)
                 })

setReplaceMethod("[[", signature(x = "gpu.matrix.tensorflow", i = "index",
                                value = "ANY"),## double/logical/...
                 function (x, i, ..., value) {
                   value <- as.vector(value)
                   if (length(i) == 1) value <- list(value)

                   index <- do.call(cbind,select_rawIndex(x,i))
                   res <- tensorflow::tf$tensor_scatter_nd_update(
                     x@gm,
                     indices = index,
                     updates = value
                   )

                   res <- gpu.matrix.tensorflow(res, dimnames = dimnames(x))

                   return(res)
                 })
