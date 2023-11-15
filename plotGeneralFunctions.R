library(float)
library(Matrix)
library(ggplot2)
library(gridExtra)
library(ggpubr)
library(grid)

creationGPUmatrix_all <- function(refMatrix){
  #float64 cuda
  GPUm_f64_cuda <- gpu.matrix(refMatrix, dtype = "float64",device = "cuda")
  #float32 cuda
  GPUm_f32_cuda <- gpu.matrix(refMatrix, dtype = "float32",device = "cuda")
  #float64 cpu
  GPUm_f64_cpu <- gpu.matrix(refMatrix, dtype = "float64",device = "cpu")
  #float32 cpu
  GPUm_f32_cpu <- gpu.matrix(refMatrix, dtype = "float32",device = "cpu")
  #float64 cuda sparse
  GPUm_f64_cuda_sparse <- gpu.matrix(refMatrix, dtype = "float64",device = "cuda", sparse = T)
  #float32 cuda sparse
  GPUm_f32_cuda_sparse <- gpu.matrix(refMatrix, dtype = "float32",device = "cuda", sparse = T)
  #float64 cpu sparse
  GPUm_f64_cpu_sparse <- gpu.matrix(refMatrix, dtype = "float64",device = "cpu", sparse = T)
  #float32 cpu sparse
  GPUm_f32_cpu_sparse <- gpu.matrix(refMatrix, dtype = "float32",device = "cpu", sparse = T)
  #float32 library
  float32_lib <- float::as.float(refMatrix)
  #Matrix
  Matrix_lib <- Matrix(refMatrix)
  res <- list("GPUm f64 cuda"=GPUm_f64_cuda, "GPUm f32 cuda"=GPUm_f32_cuda,
              "GPUm f64 cpu"=GPUm_f64_cpu, "GPUm f32 cpu"=GPUm_f32_cpu,
              "GPUm f64 cuda sparse"=GPUm_f64_cuda_sparse, "GPUm f32 cuda sparse"=GPUm_f32_cuda_sparse,
              "GPUm f64 cpu sparse"=GPUm_f64_cpu_sparse, "GPUm f32 cpu sparse"=GPUm_f32_cpu_sparse,
              "float32 lib"=float32_lib,"Matrix lib"=Matrix_lib, "MKL-R matrix"=refMatrix, "glm.fit"=refMatrix,
              "speedglm"=refMatrix)
  return(res)
}

SingleFunctionTimeCalculation <- function(listMatrixComparison,f,fgpu, nrows,ncols,Time,typeMatrixPlot){
  resTimes <- c()
  for (A in listMatrixComparison) {
    A <- abs(A)
    if(class(A)[1] == "gpu.matrix.torch"){
      CPU <- system.time({fgpu(A); torch::cuda_synchronize()})
      if (CPU[3]< Time) {
        nTimes <- round(1/(1e-2+CPU[3]))
        CPU <- system.time({for (i in 1:nTimes) {
          fgpu(A); torch::cuda_synchronize()
        }})
        CPU <- CPU / nTimes
      }
      resTimes <- c(resTimes, CPU[[3]])
    }else{
      CPU <- system.time({f(A); A[nrows, ncols]})
      if (CPU[3]< Time) {
        nTimes <- round(1/(1e-2+CPU[3]))
        CPU <- system.time({for (i in 1:nTimes) {
          f(A); A[nrows, ncols]
        }})
        CPU <- CPU / nTimes
      }
      resTimes <- c(resTimes, CPU[[3]])
    }

  }

  return(resTimes)

}

TwoFunctionTimeCalculation <- function(listMatrixComparison1,listMatrixComparison2,
                                       f,fgpu, nrows,ncols, Time, typeMatrixPlot, family=binomial()){
  resTimes <- c()
  for (i in c(1:length(listMatrixComparison1))) {
    A1 <- listMatrixComparison1[[i]]
    A1 <- abs(A1)
    A2 <- listMatrixComparison2[[i]]
    A2 <- abs(A2)
    if(class(A1)[1] == "gpu.matrix.torch"){
      CPU <- system.time({B <- fgpu(A1,A2); torch::cuda_synchronize()})
      if (CPU[3]< Time) {
        nTimes <- round(1/(1e-2+CPU[3]))
        CPU <- system.time({for (i in 1:nTimes) {
          B <- fgpu(A1,A2); torch::cuda_synchronize()
        }})
        CPU <- CPU / nTimes
      }
      resTimes <- c(resTimes, CPU[[3]])
    }else{
      # if(!is.null(OtherFunctionComparison)){
      #     CPU <- system.time({B <- OtherFunctionComparison(A1,A2); A1[nrows,ncols]})
      #     if (CPU[3]< Time) {
      #       nTimes <- round(1/(1e-2+CPU[3]))
      #       CPU <- system.time({for (i in 1:nTimes) {
      #         B <- OtherFunctionComparison(A1,A2); A1[nrows,ncols]
      #       }})
      #       CPU <- CPU / nTimesS
      #     }
      # }S
      if(typeMatrixPlot[i]=="glm.fit"){
        CPU <- system.time({B <- glm.fit(A1,A2, family = family); B})
        if (CPU[3]< Time) {
          nTimes <- round(1/(1e-2+CPU[3]))
          CPU <- system.time({for (i in 1:nTimes) {
            B <- glm.fit(A1,A2, family = family); B
          }})
          CPU <- CPU / nTimes
        }
      }else if(typeMatrixPlot[i]=="speedglm"){
        CPU <- system.time({B <- speedglm(A1,A2, family = family); B})
        if (CPU[3]< Time) {
          nTimes <- round(1/(1e-2+CPU[3]))
          CPU <- system.time({for (i in 1:nTimes) {
            B <- glm.fit(A1,A2, family = family); B
          }})
          CPU <- CPU / nTimes
        }
      }else{
        CPU <- system.time({B <- f(A1,A2); B})
        if (CPU[3]< Time) {
          nTimes <- round(1/(1e-2+CPU[3]))
          CPU <- system.time({for (i in 1:nTimes) {
            B <- f(A1,A2); B
          }})
          CPU <- CPU / nTimes
        }
      }
        # CPU <- system.time({B <- f(A1,A2); B})
        # if (CPU[3]< Time) {
        #   nTimes <- round(1/(1e-2+CPU[3]))
        #   CPU <- system.time({for (i in 1:nTimes) {
        #     B <- f(A1,A2); B
        #   }})
        #   CPU <- CPU / nTimes
        # }

      resTimes <- c(resTimes, CPU[[3]])
    }

  }

  return(resTimes)

}

# typeMatrixPlot = c("GPUm f64 cuda", "GPUm f32 cuda",
                   # "GPUm f64 cpu", "GPUm f32 cpu",
                   # "GPUm f64 cuda sparse", "GPUm f32 cuda sparse",
                   # "GPUm f64 cpu sparse", "GPUm f32 cpu sparse",
                   # "MKL-R matrix")

plotTimeComparison_SingleMatrix<- function(nrowInterval=c(500,700,1000,1400,2000,2800,4000),
                                           ncolInterval=nrowInterval,
                                           typeMatrixPlot = c("MKL-R matrix",
                                                              "GPUm f32 cpu",
                                                              "GPUm f64 cpu",
                                                              "GPUm f32 cuda",
                                                              "GPUm f64 cuda"),
                                           f,fgpu, g = rnorm, Time = .5, namePlot){
  DataFrameTimes <- c()
  sizeMatrixList <- c()
  for (i in c(1:length(nrowInterval))) {
    sizeMatrixList <- c(sizeMatrixList,paste(nrowInterval[i],ncolInterval[i],sep = "x"))
  }

  for(interval in c(1:length(nrowInterval))){
    set.seed(123)
    nrows <- nrowInterval[interval]
    ncols <- ncolInterval[interval]
    data <- g(nrows*ncols)
    A <- matrix(data, nrow = nrows, ncol = ncols)
    listMatrixComparison <- creationGPUmatrix_all(A)
    timeRes <- SingleFunctionTimeCalculation(listMatrixComparison[typeMatrixPlot], f, fgpu,nrows, ncols,Time)
    resTable <- cbind(timeRes,
          rep(nrows,length(typeMatrixPlot)),
          typeMatrixPlot)
    DataFrameTimes <- rbind(DataFrameTimes,resTable)
  }

  return(drawPlotFunction(DataFrameTimes,namePlot))
}
libraries <- c("RcppNumerical", "GPUmatrix", "torch",
               "tensorflow", "Matrix")
trash <- lapply(libraries, require, character.only = TRUE, quietly=T)

plotLRG<- function(nrowInterval=c(10000,15000,20000,25000,30000,40000,50000),
                   ncolInterval=nrowInterval,
                   typeMatrixPlotX = c("MKL-R matrix",
                                      "GPUm f32 cpu",
                                      "GPUm f64 cpu",
                                      "GPUm f32 cuda",
                                      "GPUm f64 cuda",
                                      "glm.fit"),
                   typeMatrixPloty = typeMatrixPlotX,
                   f,fgpu, g = rnorm, Time = .5, namePlot="Conjugate Gradient for Logical Regresion",
                   ylabel="Time in log10(seconds)",
                   xlabel=bquote(bold("Size ") ~ bold(X %in% R^{n * (n/100)})~""),
                   sparsity =NULL){

  if(is.null(sparsity)){
    DataFrameTimes <- c()
    sizeMatrixList <- c()
    for (i in c(1:length(nrowInterval))) {
      sizeMatrixList <- c(sizeMatrixList,paste(nrowInterval[i],ncolInterval[i],sep = "x"))
    }

    for(interval in c(1:length(nrowInterval))){
      set.seed(1234)
      n <- nrowInterval[interval]
      p <- nrowInterval[interval]/100
      X <- cbind(1, matrix(g(n * p), n, p))*.1
      beta_true <- g(p+1)
      linear_preds_true <- X %*% beta_true
      y <- rbinom(n, 1, GPUmatrix:::sigmoid(linear_preds_true))

      # set.seed(123)
      nrows <- nrowInterval[interval]
      ncols <- ncolInterval[interval]

      listMatrixComparison1 <- creationGPUmatrix_all(X)[typeMatrixPlotX]
      listMatrixComparison2 <- creationGPUmatrix_all(y)[typeMatrixPloty]

      timeRes <- TwoFunctionTimeCalculation(listMatrixComparison1,
                                            listMatrixComparison2,
                                            f,fgpu ,nrows, ncols,Time,typeMatrixPlotX)
      resTable <- cbind(timeRes,
                        rep(nrows,length(typeMatrixPlotX)),
                        typeMatrixPlotX)
      DataFrameTimes <- rbind(DataFrameTimes,resTable)
    }
  }else{
    DataFrameTimes <- c()
    sizeMatrixList <- c()
    for (i in c(1:length(nrowInterval))) {
      sizeMatrixList <- c(sizeMatrixList,paste(nrowInterval[i],ncolInterval[i],sep = "x"))
    }

    for(interval in c(1:length(nrowInterval))){
      set.seed(1234)
      sparsity <- 0.9 # Percentage of zero elements
      n <- nrowInterval[interval]
      p <- nrowInterval[interval]/100
      X <- cbind(1, matrix(rnorm(n * p), n, p)*.1)
      X[sample(1:(n * (p+1)), sparsity *  (n * (p+1)))]<- 0
      beta_true <- rnorm(p+1)
      linear_preds_true <- X %*% beta_true
      y <- rbinom(n, 1, GPUmatrix:::sigmoid(linear_preds_true))

      # set.seed(123)
      nrows <- nrowInterval[interval]
      ncols <- ncolInterval[interval]

      listMatrixComparison1 <- creationGPUmatrix_all(X)[typeMatrixPlotX]
      listMatrixComparison2 <- creationGPUmatrix_all(y)[typeMatrixPloty]

      timeRes <- TwoFunctionTimeCalculation(listMatrixComparison1,
                                            listMatrixComparison2,
                                            f,fgpu ,nrows, ncols,Time,typeMatrixPlotX)
      resTable <- cbind(timeRes,
                        rep(nrows,length(typeMatrixPloty)),
                        typeMatrixPlotX)
      DataFrameTimes <- rbind(DataFrameTimes,resTable)
    }
  }


  return(drawPlotFunction(DataFrameTimes,namePlot,xlabel = xlabel))
}


plotGLM<- function(nrowInterval=c(1000,3000,5000,7000,9000,10000,11000),
                   ncolInterval=nrowInterval,
                   typeMatrixPlotX = c("GPUm f32 cpu",
                                       "GPUm f64 cpu",
                                       "GPUm f32 cuda",
                                       "GPUm f64 cuda",
                                       "glm.fit",
                                       "speedglm"),
                   typeMatrixPloty = typeMatrixPlotX,
                   f,fgpu, g = runif, Time = .5, namePlot="General Lineal Model",
                   ylabel="Time in log10(seconds)",
                   xlabel=bquote(bold("Size ") ~ bold(X %in% R^{n * (n/10)})~""),
                   family=poisson()){
  DataFrameTimes <- c()
  sizeMatrixList <- c()
  for (i in c(1:length(nrowInterval))) {
    sizeMatrixList <- c(sizeMatrixList,paste(nrowInterval[i],ncolInterval[i],sep = "x"))
  }

  for(interval in c(1:length(nrowInterval))){
    set.seed(1)

    m <- nrowInterval[interval]
    print(m)

    n <- m/10
    X <- matrix(g(m*n),m,n)
    sol <- rnorm(n)
    y <- rbinom(m, 1, prob = inv.logit(X%*%sol))
    # set.seed(123)
    nrows <- nrowInterval[interval]
    ncols <- ncolInterval[interval]

    A1 <- X
    A2 <- y
    resTimes <- c()

    for (i in c(1:length(typeMatrixPlotX))) {
      print(typeMatrixPlotX[i])
      CPU <- switch (typeMatrixPlotX[i],
                     "glm.fit" = {
                       CPU <- system.time({B <- glm.fit(A1,A2, family = family); B})
                       if (CPU[3]< Time) {
                         nTimes <- round(1/(1e-2+CPU[3]))
                         CPU <- system.time({for (i in 1:nTimes) {
                           B <- glm.fit(A1,A2, family = family); B
                         }})
                         CPU <- CPU / nTimes
                       }
                       CPU
                     } ,
                     "speedglm" = {
                       CPU <- system.time({B <- speedglm.wfit(A2,A1, family = family); B})
                       if (CPU[3]< Time) {
                         nTimes <- round(1/(1e-2+CPU[3]))
                         CPU <- system.time({for (i in 1:nTimes) {
                           B <- speedglm.wfit(A2,A1, family = family); B
                         }})
                         CPU <- CPU / nTimes
                       }
                       CPU
                     } ,
                     "GPUm f32 cpu" = {
                       CPU <- system.time({B <- glm.fit.GPU(A1,A2, family = family, device = "cpu", dtype = "float32"); torch::cuda_synchronize()})
                       if (CPU[3]< Time) {
                         nTimes <- round(1/(1e-2+CPU[3]))
                         CPU <- system.time({for (i in 1:nTimes) {
                           B <- glm.fit.GPU(A1,A2, family = family, device = "cpu", dtype = "float32"); torch::cuda_synchronize()
                         }})
                         CPU <- CPU / nTimes
                       }
                       CPU
                     },
                     "GPUm f32 cuda" = {
                       CPU <- system.time({B <- glm.fit.GPU(A1,A2, family = family, device = "cuda", dtype = "float32"); torch::cuda_synchronize()})
                       if (CPU[3]< Time) {
                         nTimes <- round(1/(1e-2+CPU[3]))
                         CPU <- system.time({for (i in 1:nTimes) {
                           B <- glm.fit.GPU(A1,A2, family = family, device = "cuda", dtype = "float32"); torch::cuda_synchronize()
                         }})
                         CPU <- CPU / nTimes
                       }
                       CPU
                     },
                     "GPUm f64 cpu" = {
                       CPU <- system.time({B <- glm.fit.GPU(A1,A2, family = family, device = "cpu", dtype = "float64"); torch::cuda_synchronize()})
                       if (CPU[3]< Time) {
                         nTimes <- round(1/(1e-2+CPU[3]))
                         CPU <- system.time({for (i in 1:nTimes) {
                           B <- glm.fit.GPU(A1,A2, family = family, device = "cpu", dtype = "float64"); torch::cuda_synchronize()
                         }})
                         CPU <- CPU / nTimes
                       }
                       CPU
                     },
                     "GPUm f64 cuda" = {
                       CPU <- system.time({B <- glm.fit.GPU(A1,A2, family = family, device = "cuda", dtype = "float64"); torch::cuda_synchronize()})
                       if (CPU[3]< Time) {
                         nTimes <- round(1/(1e-2+CPU[3]))
                         CPU <- system.time({for (i in 1:nTimes) {
                           B <- glm.fit.GPU(A1,A2, family = family, device = "cuda", dtype = "float64"); torch::cuda_synchronize()
                         }})
                         CPU <- CPU / nTimes
                       }
                       CPU
                     })
      resTimes <- c(resTimes, CPU[[3]])
    }



    resTable <- cbind(resTimes,
                      rep(nrows,length(typeMatrixPlotX)),
                      typeMatrixPlotX)
    DataFrameTimes <- rbind(DataFrameTimes,resTable)
  }


  return(drawPlotFunction(DataFrameTimes,namePlot,xlabel = xlabel))
}

plotTimeComparison_TwoMatrix<- function(nrowInterval=c(500,700,1000,1400,2000,2800,4000),
                                           ncolInterval=nrowInterval,
                                           typeMatrixPlot = c("MKL-R matrix",
                                                              "GPUm f32 cpu",
                                                              "GPUm f64 cpu",
                                                              "GPUm f32 cuda",
                                                              "GPUm f64 cuda"),
                                           f,fgpu, g = rnorm, Time = .5, namePlot,
                                          ylabel="Time in log10(seconds)",
                                          xlabel="Size matrix n×n"){
  DataFrameTimes <- c()
  sizeMatrixList <- c()
  for (i in c(1:length(nrowInterval))) {
    sizeMatrixList <- c(sizeMatrixList,paste(nrowInterval[i],ncolInterval[i],sep = "x"))
  }

  for(interval in c(1:length(nrowInterval))){
    set.seed(123)
    nrows <- nrowInterval[interval]
    ncols <- ncolInterval[interval]
    # data <- g(nrows*ncols)

    data1 <- g(nrows*ncols)
    data2 <- g(nrows*ncols)
    A1 <- matrix(data1, nrow = nrows, ncol = ncols)
    A2 <- matrix(data2, nrow = nrows, ncol = ncols)
    # A <- matrix(data, nrow = nrows, ncol = ncols)
    listMatrixComparison1 <- creationGPUmatrix_all(A1)[typeMatrixPlot]
    listMatrixComparison2 <- creationGPUmatrix_all(A2)[typeMatrixPlot]
    timeRes <- TwoFunctionTimeCalculation(listMatrixComparison1,
                                          listMatrixComparison2,
                                          f,fgpu ,nrows, ncols,Time)
    resTable <- cbind(timeRes,
                      rep(nrows,length(typeMatrixPlot)),
                      typeMatrixPlot)
    DataFrameTimes <- rbind(DataFrameTimes,resTable)
  }

  return(drawPlotFunction(DataFrameTimes,namePlot,ylabel,xlabel))
}

drawPlotFunction <- function(DataFrameTimes,namePlot,ylabel="Time in log10(seconds)",xlabel="Size matrix n×n"){
  colnames(DataFrameTimes) <- c("Time", "Size", "Type")
  # colorValues <- c("#14690A","#1B06B4", "#1B06B4","#DE044A","#DE044A", "#EABA0A","#EABA0A")

  DataFrameTimes <- as.data.frame(DataFrameTimes)
  DataFrameTimes <-transform(DataFrameTimes,Time=as.numeric(Time),Size=as.numeric(Size))
  DataFrameTimes <- as.data.frame(DataFrameTimes)
  DataFrameTimes$Time[DataFrameTimes$Time==0] <- min(DataFrameTimes$Time[DataFrameTimes$Time!=0])
  colorValues <- getColorsType(DataFrameTimes$Type)
  linetypeValues <- getlineType(DataFrameTimes$Type)
  p<-ggplot(data=DataFrameTimes, aes(x=Size, y=Time, group=Type)) +
    geom_line(aes(color=Type,linetype = Type), linewidth = 1.5, alpha=0.5)+
    geom_point(aes(color=Type,shape=Type), size = 3, alpha=0.5)+
    scale_y_log10(limits=c(min(DataFrameTimes$Time),max(DataFrameTimes$Time)))+
    labs(y = ylabel,x=xlabel, title=namePlot)+
    theme_classic()+
    scale_color_manual(values = colorValues)+
    scale_linetype_manual(values = linetypeValues) +
    scale_shape_manual(values = getShapeManual(DataFrameTimes$Type))+
    theme(axis.text = element_text(face = "bold"),
          axis.title = element_text(face = "bold"),
          plot.title = element_text(face = "bold"),
          legend.text = element_text(face = "bold"),
          legend.position = "bottom")
  return(p)
}

getColorsType <- function(types){
  res <- sapply(sort(unique(types)), FUN = function(type){
    res <- switch (type,
      "MKL-R matrix" = {"#14690A"} ,
      "glm.fit" = {"#9907C7"} ,
      "speedglm" = {"#EA770A"} ,
      "GPUm f32 cpu" = {"#1B06B4"},
      "GPUm f32 cuda" = {"#1B06B4"},
      "GPUm f32 cpu sparse" = {"#EABA0A"},
      "GPUm f32 cuda sparse" = {"#EABA0A"},
      "Matrix lib" = {"#9907C7"} ,
      "GPUm f64 cpu" = {"#C70039"},
      "GPUm f64 cuda" = {"#C70039"},
      "GPUm f64 cpu sparse" = {"#EA770A"},
      "GPUm f64 cuda sparse" = {"#EA770A"}
    )
    return(res)
  })
  return(as.vector(res))
}
getShapeManual <- function(types){
  res <- sapply(sort(unique(types)), FUN = function(type){
    res <- switch (type,
                   "MKL-R matrix" = {16} ,
                   "glm.fit" = {16},
                   "speedglm" = {16},
                   "GPUm f32 cpu" = {16},
                   "GPUm f32 cuda" = {17},
                   "GPUm f32 cpu sparse" = {16},
                   "GPUm f32 cuda sparse" = {17},
                   "Matrix lib" = {16} ,
                   "GPUm f64 cpu" = {16},
                   "GPUm f64 cuda" = {17},
                   "GPUm f64 cpu sparse" = {16},
                   "GPUm f64 cuda sparse" = {17}
    )
    return(res)
  })
  return(as.vector(res))
}
getlineType <- function(types){
  res <- sapply(sort(unique(types)) , FUN = function(type){
    res <- switch (type,
            "MKL-R matrix" ={return("solid")} ,
            "glm.fit" ={return("solid")} ,
            "speedglm" ={return("solid")} ,
            "GPUm f32 cpu" ={return("solid")},
            "GPUm f32 cuda" ={return("dashed")},
            "GPUm f32 cpu sparse" ={return("solid")},
            "GPUm f32 cuda sparse" ={return("dashed")},
            "Matrix lib" ={return("solid")} ,
            "GPUm f64 cpu" ={return("solid")},
            "GPUm f64 cuda" ={return("dashed")},
            "GPUm f64 cpu sparse" ={return("solid")},
            "GPUm f64 cuda sparse" ={return("dashed")}
    )
    return(res)
  })
  return(as.vector(res))
}

# multiplePlotsGPUmatrix <- function(setFunctions, dimPlot, vectorArgNumber,nombresPlot){
#
#   setPlots <- vector(mode = "list", length = length(setFunctions))
#   for (i in c(1:length(setFunctions))) {
#     funTest <- setFunctions[i]
#     if (vectorArgNumber[i]==1) p <- plotTimeComparison_SingleMatrix(f=match.fun(funTest), namePlot=nombresPlot[i])
#     if (vectorArgNumber[i]==2) p <- plotTimeComparison_TwoMatrix(f=match.fun(funTest), namePlot=nombresPlot[i])
#     setPlots[[i]] <- p
#   }
#
#   grid.arrange(setPlots[c(1,2,4)],ncol=dimPlot[1], nrow=dimPlot[2])
# }

#
# plotRowMeans <- plotTimeComparison_SingleMatrix(f=rowMeans,fgpu = GPUmatrix::rowMeans, namePlot="Means of the rows")
# plotSolve <- plotTimeComparison_SingleMatrix(f=match.fun("solve"), namePlot="Inverse of a Matrix")
# # plotFft <- plotTimeComparison_SingleMatrix(f=match.fun("fft"), namePlot="fft")
# plotSvd <- plotTimeComparison_SingleMatrix(f=match.fun("svd"), namePlot="Singular Value Decomposition")
# plotExp <- plotTimeComparison_SingleMatrix(f=match.fun("exp"), namePlot="Exponential of each element")
# plotProduct <- plotTimeComparison_TwoMatrix(f=match.fun("*"), namePlot="Element-wise product")
# # pLengend <- plotTimeComparison_TwoMatrix(f=match.fun("*"), namePlot="Element-wise product")
# plotMatProduct <- plotTimeComparison_TwoMatrix(f=match.fun("%*%"), namePlot="Matrix product")
# legendPlot <- get_legend(plotProduct)
# plotProduct <- plotProduct+ theme(legend.position = "none")+scale_y_log10()+labs(y = "Time in seconds (log10-scale)",x=NULL, title="Element-wise product")
# plotExp <- plotExp+ theme(legend.position = "none")+scale_y_log10()+labs(y = NULL,x=NULL, title="Exponential of each element")
# plotRowMeans <- plotRowMeans+ theme(legend.position = "none")+scale_y_log10()+labs(y = "Time in seconds (log10-scale)",x=NULL, title="Means of the rows")
# plotMatProduct <- plotMatProduct+ theme(legend.position = "none")+scale_y_log10()+labs(y = NULL,x=NULL, title="Matrix product")
# plotSolve <- plotSolve+ theme(legend.position = "none")+scale_y_log10()+labs(y = "Time in seconds (log10-scale)",x="Size matrix n×n", title="Inverse of a matrix")
# plotSvd <- plotSvd+ theme(legend.position = "none")+scale_y_log10()+labs(y = NULL,x="Size matrix n×n", title="FFT")
#
#
# ComputationTimeGraph <- grid.arrange(plotProduct,
#                                      plotExp,
#                                      plotRowMeans,
#                                      plotMatProduct,
#                                      plotSolve,
#                                      plotSvd,
#                                      ncol = 2, nrow=3,
#                                      bottom=legendPlot)
# ComputationTimeGraph
# # torch::cuda_synchronize(torch_device("cuda"))
# # ?cuda_current_device()
# plotQR_solve <- plotTimeComparison_TwoMatrix(f=match.fun("qr.solve"), namePlot="qr.solve", nrowInterval = c(100,400,700,1200,1500,1800,2000))
# plotQR <- plotTimeComparison_SingleMatrix(f=match.fun("qr"), namePlot="qr", nrowInterval = c(100,400,700,1200,1500,1800,2000))
#
# plotNMFgpumatrix <- plotTimeComparison_SingleMatrix(f=match.fun("NMFgpumatrix"), namePlot="Non negative factorization")
# plotLRGC <- plotTimeComparison_TwoMatrix(f=match.fun("qr.solve"), namePlot="qr_solve",typeMatrixPlot =c("MKL-R matrix",
#                                                                                                         "GPUm f32 cpu","GPUm f64 cpu",
#                                                                                                         "GPUm f32 cuda",
#                                                                                                          "GPUm f64 cuda",
#                                                                                                         "Matrix lib",
#                                                                                                          "GPUm f64 cuda sparse",
#                                                                                                          "GPUm f32 cuda sparse","GPUm f64 cpu sparse",
#                                                                                                         "GPUm f32 cpu sparse"), nrowInterval = c(100,400,700,1200,1500,1800,2000))
# plotfft <- plotTimeComparison_SingleMatrix(f=match.fun("fft"), namePlot="fft",typeMatrixPlot =c("MKL-R matrix",
#                                                                                                         "GPUm f32 cpu","GPUm f64 cpu",
#                                                                                                         "GPUm f32 cuda",
#                                                                                                         "GPUm f64 cuda",
#                                                                                                         "GPUm f64 cuda sparse",
#                                                                                                         "GPUm f32 cuda sparse",
#                                                                                                         "GPUm f64 cpu sparse",
#                                                                                                         "GPUm f32 cpu sparse"))
