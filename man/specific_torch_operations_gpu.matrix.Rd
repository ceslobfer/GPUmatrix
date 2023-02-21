\name{specific_torch_operations_gpu.matrix}
\title{specific_torch_operations_gpu.matrix}
\docType{methods}



\alias{chol_solve-methods}

\alias{chol_solve-methods}

\alias{to_dense-methods}

\alias{to_sparse-methods}

\alias{ginv-methods}

\alias{is-methods}

\alias{dtype-methods}

\alias{dtype<--methods}

\alias{chol_solve,ANY,gpu.matrix.torch-method}

\alias{chol_solve,gpu.matrix.torch,ANY-method}

\alias{to_dense,gpu.matrix.torch-method}

\alias{to_sparse,gpu.matrix.torch-method}

\alias{ginv,gpu.matrix.torch-method}

\alias{is,gpu.matrix.torch-method}

\alias{dtype,gpu.matrix.torch-method}

\alias{dtype<-,gpu.matrix.torch-method}




\description{
Specific functions for gpu.matrix.torch object
}


\usage{

\S4method{chol_solve}{ANY,gpu.matrix.torch}(x,y) 

\S4method{chol_solve}{gpu.matrix.torch,ANY}(x,y) 

\S4method{to_dense}{gpu.matrix.torch}(x) 

\S4method{to_sparse}{gpu.matrix.torch}(x) 

\S4method{ginv}{gpu.matrix.torch}(x,tol) 

\S4method{is}{gpu.matrix.torch}(x) 

\S4method{dtype}{gpu.matrix.torch}(x) 

\S4method{dtype<-}{gpu.matrix.torch}(x,value) 

}


\arguments{

\item{x}{A gpu.matrix.torch or a gpu.matrix.tensorflow object.}

\item{y}{a numeric or complex vector or matrix giving the right-hand side(s) of the linear system.}

\item{tol}{A relative tolerance to detect zero singular values.}

\item{value}{type of gpu.matrix object}

}
