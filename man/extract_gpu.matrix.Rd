\name{extract_gpu.matrix}
\title{extract_gpu.matrix}
\docType{methods}


\alias{-}
\alias{--methods}
\alias{-,gpu.matrix.tensorflow,missing-method}
\alias{-,gpu.matrix.torch,missing-method}

\alias{[}
\alias{[<-}
\alias{[[}
\alias{[[<-}
\alias{[-methods}
\alias{[<--methods}
\alias{[[-methods}
\alias{[[<--methods}
\alias{[,gpu.matrix.tensorflow,index,index,ANY-method}
\alias{[,gpu.matrix.tensorflow,index,missing,ANY-method}
\alias{[,gpu.matrix.tensorflow,matrix,missing,ANY-method}
\alias{[,gpu.matrix.tensorflow,missing,index,ANY-method}
\alias{[,gpu.matrix.torch,index,index,ANY-method}
\alias{[,gpu.matrix.torch,index,missing,ANY-method}
\alias{[,gpu.matrix.torch,matrix,missing,ANY-method}
\alias{[,gpu.matrix.torch,missing,index,ANY-method}
\alias{[<-,gpu.matrix.tensorflow,index,index,ANY-method}
\alias{[<-,gpu.matrix.tensorflow,index,missing,ANY-method}
\alias{[<-,gpu.matrix.tensorflow,matrix,missing,ANY-method}
\alias{[<-,gpu.matrix.tensorflow,missing,index,ANY-method}
\alias{[<-,gpu.matrix.torch,index,index,ANY-method}
\alias{[<-,gpu.matrix.torch,index,missing,ANY-method}
\alias{[<-,gpu.matrix.torch,matrix,missing,ANY-method}
\alias{[<-,gpu.matrix.torch,missing,index,ANY-method}
\alias{[[,gpu.matrix.tensorflow,index-method}
\alias{[[,gpu.matrix.torch,index-method}
\alias{[[<-,gpu.matrix.tensorflow,index-method}
\alias{[[<-,gpu.matrix.torch,index-method}



\description{
Mimick the \code{[,[<-, [[, and [[<-} base functions. These operators act on GPUmatrix is act on vectors, matrices, arrays and lists.
}


\usage{
\S4method{-}{gpu.matrix.tensorflow,missing}(e1,e2)
\S4method{-}{gpu.matrix.torch,missing}(e1,e2)
\S4method{[}{gpu.matrix.tensorflow,index,index,ANY}(x,i,j)
\S4method{[}{gpu.matrix.tensorflow,index,missing,ANY}(x,i,j,\dots,drop = TRUE)
\S4method{[}{gpu.matrix.tensorflow,matrix,missing,ANY}(x,i,j,\dots,drop = TRUE)
\S4method{[}{gpu.matrix.tensorflow,missing,index,ANY}(x,i,j)
\S4method{[}{gpu.matrix.torch,index,index,ANY}(x,i,j)
\S4method{[}{gpu.matrix.torch,index,missing,ANY}(x,i,j,\dots,drop = TRUE)
\S4method{[}{gpu.matrix.torch,matrix,missing,ANY}(x,i,j,\dots,drop = TRUE)
\S4method{[}{gpu.matrix.torch,missing,index,ANY}(x,i,j)
\S4method{[}{gpu.matrix.tensorflow,index,index,ANY}(x,i,j) <- value
\S4method{[}{gpu.matrix.tensorflow,index,missing,ANY}(x,i,j) <- value
\S4method{[}{gpu.matrix.tensorflow,matrix,missing,ANY}(x,i,j) <- value
\S4method{[}{gpu.matrix.tensorflow,missing,index,ANY}(x,i,j) <- value
\S4method{[}{gpu.matrix.torch,index,index,ANY}(x,i,j) <- value
\S4method{[}{gpu.matrix.torch,index,missing,ANY}(x,i,j) <- value
\S4method{[}{gpu.matrix.torch,matrix,missing,ANY}(x,i,j) <- value
\S4method{[}{gpu.matrix.torch,missing,index,ANY}(x,i,j) <- value
\S4method{[[}{gpu.matrix.tensorflow,index}(x,i,j,\dots)
\S4method{[[}{gpu.matrix.torch,index}(x,i,j,\dots)
\S4method{[[}{gpu.matrix.tensorflow,index}(x,i) <- value
\S4method{[[}{gpu.matrix.torch,index}(x,i) <- value

}


\arguments{

\item{e1}{a gpu.matrix}
\item{e2}{a gpu.matrix}
\item{x}{a gpu.matrix from which extract element(s) or in which to replace element(s)}
\item{i,j,\dots}{indices specifying elements to extract or replace.}
\item{value}{typically an array-like R object of a similar class as x}
\item{drop}{For matrices and arrays. If TRUE the result is coerced to the lowest possible dimension}

}


\seealso{
See Also \code{\link[base]{Extract}}.
}


\examples{

\donttest{
\dontrun{

a <- gpu.matrix(1:9,nrow=3,ncol=3,type = "tensorflow")
rownames(a) <- c("R1","R2","R3")
colnames(a) <- c("C1","C2","C3")

#return
a[3,3] # the element row 3 and column 3
a[6] # the 6th element
a[1,] # the first row
a[c(1,2),] # the first and second row
a[c(1,1),] # the first row twice
a[,1] # the first column
a[,c(1,2)] # the first and second column
a[,c(1,1)] # the first column twice

#replace
a[3,3] <- 100 # replace the element 3,3
a[1,] <- c(1,2,1) # replace the first row
a[,2] <- c(0,0,0) # replace the second column
a[c(1,2),] <- matrix(1:6,nrow = 2) # replace the first and second row



}
}
}


