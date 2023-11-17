\name{gpu.matrix-class}
\title{Class 'gpu.matrix' for matrix stored in GPU}
\docType{class}
\alias{gpu.matrix-class}
\description{

GPU computational power is a great resource for computational biology specifically in statistics and linear algebra. the gpu.matrix-class is a class of the GPUmatrix package, that store a matrix in the GPU.

The GPUmatrix package is based on S4 objects in R and we have created a constructor function that acts similarly to the default \code{matrix} constructor in R for CPU matrices. The constructor function is \code{\link{gpu.matrix}} and accepts the same parameters as \code{matrix}.



}
\section{Slots}{ Use the \code{@} operator to access the different slots:
  \describe{
    \item{rownames}{the row names of the gpu.matrix}
    \item{colnames}{the colunm names of the gpu.matrix}
    \item{gm}{the corresponding tensor}
    \item{sparse}{Logical: indicates if the gpu.matrix is sparse or not}
    \item{type}{If it is tensorflow or torch}
  }
}

\seealso{
See Also \code{\link{gpu.matrix}}, \code{\linkS4class{Matrix}}, and \code{\linkS4class{matrix}}..
}
