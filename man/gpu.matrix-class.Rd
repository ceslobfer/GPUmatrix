\name{gpu.matrix-class}
\title{Class 'gpu.matrix' for matrix stored in GPU}
\docType{class}
\alias{gpu.matrix-class}
\description{
The matrix stored in the GPU.
}
\section{Slots}{
  \describe{
    \item{rownames}{the row names of the gpu.matrix}
    \item{colnames}{the colunm names of the gpu.matrix}
    \item{gm}{the corresponding tensor}
    \item{sparse}{Logical: indicates if the gpu.matrix is sparse or not}
  }
}
\examples{
a <- 5
}
