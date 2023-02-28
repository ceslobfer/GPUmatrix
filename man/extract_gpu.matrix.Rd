\name{extract_gpu.matrix}
\title{extract_gpu.matrix}
\docType{methods}



\alias{--methods}
\alias{[-methods}
\alias{[<--methods}
\alias{[[-methods}
\alias{[[<--methods}
\alias{-,gpu.matrix.tensorflow,missing-method}
\alias{-,gpu.matrix.torch,missing-method}
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
Extract specific possition of a gpu.matrix object.
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
\item{x}{a gpu.matrix}
\item{i,j,\dots}{indices specifying elements to extract or replace.}
\item{value}{typically an array-like R object of a similar class as x}
\item{drop}{For matrices and arrays. If TRUE the result is coerced to the lowest possible dimension}

}
