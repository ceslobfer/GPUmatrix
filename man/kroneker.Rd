\name{kroneker}
\title{kroneker}
\docType{methods}



\alias{\%x\%-methods}

\alias{\%x\%,ANY,gpu.matrix.tensorflow-method}

\alias{\%x\%,ANY,gpu.matrix.torch-method}

\alias{\%x\%,gpu.matrix.tensorflow,ANY-method}

\alias{\%x\%,gpu.matrix.torch,ANY-method}




\description{
Kroneker for GPU matrices.
}


\usage{

\S4method{\%x\%}{ANY,gpu.matrix.tensorflow}(X,Y)

\S4method{\%x\%}{ANY,gpu.matrix.torch}(X,Y)

\S4method{\%x\%}{gpu.matrix.tensorflow,ANY}(X,Y)

\S4method{\%x\%}{gpu.matrix.torch,ANY}(X,Y)

}


\arguments{

\item{X}{A gpu.matrix.tensorflow o gpu.matrix.torch object}

\item{Y}{A gpu.matrix.tensorflow o gpu.matrix.torch object or a matrix or a numeric variable}

}
