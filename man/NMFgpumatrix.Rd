\name{NMFgpumatrix}
\title{Non negative factorization of a matrix}

\description{
The non-negative factorization (NMF) of a matrix is an approximate factorization were an initial matrix \code{V} is approximated by the product of two matrix \code{W} and \code{H} so that,

\eqn{V \approx WH}

This function operates in the same way with the 'base' \code{matrix} objects as with gpu.matrix-class objects, and it does not require any additional changes beyond initializing the input matrix as a gpu.matrix-class object.

}

\alias{NMFgpumatrix}

\usage{
NMFgpumatrix(V, k = 10, Winit = NULL,
             Hinit = NULL, tol = 1e-06,
             niter = 100)
}
\arguments{
  \item{V}{a \code{\linkS4class{gpu.matrix}}. Values in \code{V} must be \eqn{\geq 0}.}
  \item{k}{The inner dimension of the product of the matrices W and H. That is, it corresponds to the number of columns in W and the number of rows in H.}
  \item{Winit}{Initial value for matrix W. Initial values for \code{W} must be \eqn{\geq 0}.}
  \item{Hinit}{Initial value for matrix H. Initial values for \code{H} must be \eqn{\geq 0}.}
  \item{tol}{tolerance to be used for the estimation.}
  \item{niter}{maximum number of iterations.}
}

\details{
We have implemented our own non-negative matrix factorization (NMF) function using Lee
and Seung[1] multiplicative update rule:

\eqn{W_{[i,j]}^{n+1} \leftarrow  W_{[i,j]}^{n} \frac{(V(H^{n+1})^T)_{[i,j]}}{(W^nH^{n+1}(H^{n+1})^T)_{[i,j]}}}

and

\eqn{H_{[i,j]}^{n+1} \leftarrow  H_{[i,j]}^{n} \frac{((W^{n})^TV)_{[i,j]}}{((W^n)^TW^{n}H^{n})_{[i,j]}}}

to update the \eqn{W} and \eqn{H} respectively.

Note that the values of V must be positive. If any value of V is negative, it will be set to 0. If this happens, the following warning message will be displayed: "The values of V must be positive. Negative values in V are set to 0.

If the user decides to initialise the values of W and H, they must also be positive. If there are negative values, they will be set to 0. The following warning message will be displayed: "The Winit values must be positive. Negative values in Winit are set to 0" if Winit has negative values or "The values of Hinit must be positive. Negative values in Hinit are set to 0" if Winit has negative values.

In addition, Winit and Hinit must also have the correct dimensions. Winit must fulfil two conditions: \code{nrow(Winit) == nrow(V)} and  \code{ncol(Winit) == k}. If not, the function will stop with the following error message: "The dimensions of the Winit matrix are incorrect. Please check that \code{nrow(Winit) == nrow(V)} and that \code{ncol(Winit) == k}". On the other hand, Hinit must fulfil two conditions: \code{nrow(Winit) == nrow(V)} and that \code{ncol(Winit) == k}. If not, the function will stop with the following error message: "The dimensions of the Winit matrix are incorrect. Please check that \code{nrow(Winit) == nrow(V)} and that \code{ncol(Winit) == k}".


If the input gpu.matrix-class object is stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.


}

\references{
[1] Lee, D., Seung, H. Learning the parts of objects by non-negative matrix factorization. Nature 401, 788–791 (1999). https://doi.org/10.1038/44565
}

\author{
Angel Rubio and Cesar Lobato.
}


\value{
The function returns a list that contains the corresponding matrix \eqn{W} and \eqn{H}. If the input \code{V} matrix is a gpu.matrix-class object, then both \eqn{W} and \eqn{H} are also gpu.matrix-class objects.
}
\examples{
\donttest{
\dontrun{
library(Matrix)
set.seed(1)
a1 <- gpu.matrix(runif(90),nrow=30,ncol=3)
a2 <- gpu.matrix(runif(30),nrow=3,ncol=10)
V <- a1 \%*\% a2
b <- NMFgpumatrix(V = V, k=3, tol = 1e-6)

#check result:
image(Matrix(as.matrix(V)))
image(Matrix(as.matrix(b$W \%*\% b$H)))

}
}
}

