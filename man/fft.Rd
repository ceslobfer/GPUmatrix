\name{fft}
\title{Fast Discrete Fourier Transform (FFT)}
\description{
This function mimics the function \code{fft} of the library stats to compute on gpu.matrix-class objects: it "Computes the Discrete Fourier Transform (DFT) of an array with a fast algorithm, the “Fast Fourier Transform” (FFT)."

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
\item{z}{a real or complex \code{\linkS4class{gpu.matrix}} object containing the values to be transformed.}
}

\seealso{
For more information see:
\code{\link[stats]{fft}} and \code{\link[torch]{torch_fft_ifft}}.
}

\details{
This function mimics the function \code{\link[stats]{fft}} to operate on gpu.matrix-class objects.

Internally, this function calls the \code{tff} function from the 'tensorflow' library or the \code{torch_fft_ifft} function from the torch library (depending on the type of gpu.matrix-class object used). For more details see \code{\link[torch]{torch_fft_ifft}} (there is no help page for the function corresponding to Tensorflow.).

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



