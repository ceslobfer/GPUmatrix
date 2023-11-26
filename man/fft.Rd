\name{fft}
\title{Fast Discrete Fourier Transform (FFT)}
\description{
This function mimics the function \code{fft} of the library 'stats' to compute on gpu.matrix-class objects: it "Computes the Discrete Fourier Transform (DFT) of an array with a fast algorithm, the “Fast Fourier Transform” (FFT)."

}

\alias{fft}
\alias{fft-methods}
\alias{fft,gpu.matrix.tensorflow-method}
\alias{fft,gpu.matrix.tensorflow,ANY-method}
\alias{fft,gpu.matrix.torch-method}
\alias{fft,gpu.matrix.torch,logical-method}
\alias{fft,gpu.matrix.torch,missing-method}


\usage{
\S4method{fft}{gpu.matrix.tensorflow}(z)
\S4method{fft}{gpu.matrix.torch}(z)
}

\arguments{
\item{z}{a  \code{\linkS4class{gpu.matrix}} object containing the values to be transformed.}
}

\seealso{
For more information see:
\code{\link[stats]{fft}} and \code{\link[torch]{torch_fft_ifft}}.
}

\details{
This function mimics the function \code{\link[stats]{fft}} to operate on gpu.matrix-class objects.

The function \code{fft} internally calls the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.

}

\value{
It returns a gpu.matrix-class object with the transformed values. To access the real and imaginary information, the following code must be executed:
\code{output@gm$real} for the real part and \code{output@gm$imag} for the imaginary part.
}


\examples{
if(installTorch){

  x <- gpu.matrix(1:4)
  output <- fft(x)
  output@gm$real
  output@gm$imag
}


}



