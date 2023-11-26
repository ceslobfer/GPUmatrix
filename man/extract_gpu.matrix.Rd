\name{extract_gpu.matrix}
\title{extract_gpu.matrix}
\docType{methods}

\description{
These operators mimic the base operators \code{[,[<-, [[, and [[<-} to compute on gpu.matrix-class objects.
}


\alias{[-methods}
\alias{[,gpu.matrix.tensorflow,index,index-method}
\alias{[,gpu.matrix.tensorflow,index,missing-method}
\alias{[,gpu.matrix.tensorflow,matrix,missing-method}
\alias{[,gpu.matrix.tensorflow,missing,index-method}
\alias{[,gpu.matrix.torch,index,index-method}
\alias{[,gpu.matrix.torch,index,missing-method}
\alias{[,gpu.matrix.torch,matrix,missing-method}
\alias{[,gpu.matrix.torch,missing,index-method}
\alias{[<-,gpu.matrix.tensorflow,index,index-method}
\alias{[<-,gpu.matrix.tensorflow,index,missing-method}
\alias{[<-,gpu.matrix.tensorflow,matrix,missing-method}
\alias{[<-,gpu.matrix.tensorflow,missing,index-method}
\alias{[<-,gpu.matrix.torch,index,index-method}
\alias{[<-,gpu.matrix.torch,index,missing-method}
\alias{[<-,gpu.matrix.torch,matrix,missing-method}
\alias{[<-,gpu.matrix.torch,missing,index-method}
\alias{[[,gpu.matrix.tensorflow,index-method}
\alias{[[,gpu.matrix.torch,index-method}
\alias{[[<-,gpu.matrix.tensorflow,index-method}
\alias{[[<-,gpu.matrix.torch,index-method}
\alias{[}
\alias{[<-}
\alias{[[}
\alias{[[<-}
\alias{--methods}
\alias{-}
\alias{-,gpu.matrix.tensorflow,missing-method}
\alias{-,gpu.matrix.torch,missing-method}
\alias{[-methods}
\alias{[<--methods}
\alias{[[-methods}
\alias{[[<--methods}



\usage{
\S4method{-}{gpu.matrix.tensorflow,missing}(e1,e2)
\S4method{-}{gpu.matrix.torch,missing}(e1,e2)
\S4method{[}{gpu.matrix.tensorflow,index,index}(x,i,j)
\S4method{[}{gpu.matrix.tensorflow,index,missing}(x,i,j,\dots,drop = TRUE)
\S4method{[}{gpu.matrix.tensorflow,matrix,missing}(x,i,j,\dots,drop = TRUE)
\S4method{[}{gpu.matrix.tensorflow,missing,index}(x,i,j)
\S4method{[}{gpu.matrix.torch,index,index}(x,i,j)
\S4method{[}{gpu.matrix.torch,index,missing}(x,i,j,\dots,drop = TRUE)
\S4method{[}{gpu.matrix.torch,matrix,missing}(x,i,j,\dots,drop = TRUE)
\S4method{[}{gpu.matrix.torch,missing,index}(x,i,j)
\S4method{[}{gpu.matrix.tensorflow,index,index}(x,i,j) <- value
\S4method{[}{gpu.matrix.tensorflow,index,missing}(x,i,j) <- value
\S4method{[}{gpu.matrix.tensorflow,matrix,missing}(x,i,j) <- value
\S4method{[}{gpu.matrix.tensorflow,missing,index}(x,i,j) <- value
\S4method{[}{gpu.matrix.torch,index,index}(x,i,j) <- value
\S4method{[}{gpu.matrix.torch,index,missing}(x,i,j) <- value
\S4method{[}{gpu.matrix.torch,matrix,missing}(x,i,j) <- value
\S4method{[}{gpu.matrix.torch,missing,index}(x,i,j) <- value
\S4method{[[}{gpu.matrix.tensorflow,index}(x,i,j,\dots)
\S4method{[[}{gpu.matrix.torch,index}(x,i,j,\dots)
\S4method{[[}{gpu.matrix.tensorflow,index}(x,i) <- value
\S4method{[[}{gpu.matrix.torch,index}(x,i) <- value

}


\arguments{

\item{e1}{a \code{\linkS4class{gpu.matrix}}.}
\item{e2}{a \code{\linkS4class{gpu.matrix}}.}
\item{x}{a \code{\linkS4class{gpu.matrix}} object from which extract element(s) or in which to replace element(s)}
\item{i,j,\dots}{indices specifying elements to extract or replace.}
\item{value}{typically an array-like R object of a similar class as \code{x}.}
\item{drop}{For matrices and arrays. If TRUE the result is coerced to the lowest possible dimension}

}

\details{
When replacing a value or values in a gpu.matrix, the gpu.matrix will not change its datatype (corresponding to the parameter \code{dtype} of the function \code{\link{gpu.matrix}}) based on the datatype in value. For example, the code \code{x[1,1] <- value} where the array \code{x} is a gpu.matrix with integer values and \code{value} has 'double' values, only the integer part of \code{value} will be stored in \code{x[1,1]}.
}


\seealso{
See Also \code{\link[base]{Extract}}.
}


\examples{

\donttest{
\dontrun{

a <- gpu.matrix(1:9,nrow=3,ncol=3)
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


