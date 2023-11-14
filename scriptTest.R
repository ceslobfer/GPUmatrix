# TODO: implement pmax



# Standard glm

counts <- c(18,17,15,20,10,20,25,13,12)
outcome <- gl(3,1,9)
treatment <- gl(3,3)
data.frame(treatment, outcome, counts) # showing data
glm.D93 <- glm(counts ~ outcome + treatment, family = poisson())
summary(glm.D93)


#Speedglm
library(speedglm)
sglm.D93 <- speedglm(counts ~ outcome + treatment, family = poisson())
summary(sglm.D93)

# GPU glm!!
library(GPUmatrix)
source("~/GitHub/GPUmatrix/R/glm.fit.GPU.R")
gpu.glm.D93 <- glm(counts ~ outcome + treatment, family = poisson(), method = "glm.fit.GPU")
fit.glm.D93 <- glm(counts ~ outcome + treatment, family = poisson(), method = "glm.fit")
speed.glm.D93 <- speedglm(counts ~ outcome + treatment)

gpu.glm.D93 <-GPUglmfit(counts ~ outcome + treatment, family = poisson())
# Una vez terminado...
# gpu.glm.D93 <- GPUglm(counts ~ outcome + treatment, family = poisson(), method = "glm.fit.GPU")
# GPUglm <- function(...) {
#   output <- glm(..., method = "glm.fit.GPU")
#   class(gpu.glm.D93) <- c("GPUglm","GPUglm") # --> Fix me!!!
# }
# Devuelve un objeto GPUglm
# summary(gpu.glm.D93) --> IMplementar el metodo correspondiente al summary (copy paste de sppedglm)

class(gpu.glm.D93) <- c("speedglm","speedlm") # --> Fix me!!!
gpu.glm.D932 <- gpu.glm.D93
summary(gpu.glm.D93)

# Let's go with the big guys
library(gtools) # for logit
library(speedglm)
m <- 1000
n <- 100
x <- matrix(runif(m*n),m,n)
sol <- rnorm(n)
y <- rbinom(m, 1, prob = inv.logit(x%*%sol))

system.time(s1 <- glm.fit(x,y,family = binomial())$coefficients)
system.time(s2 <- speedglm.wfit(y,x,family = binomial())$coefficients)
system.time(s3 <- glm.fit.GPU(x,y,family = binomial(),dtype="float32")$coefficients)
system.time(s3 <- glm.fit.GPU(x,y,family = binomial(),dtype="float64")$coefficients)
plot(s1,s2)
plot(s1,s3)
plot(s2-s3)



plotGLM<- function(nrowInterval=c(1000,3000,5000,7000,9000,10000,11000),
                   ncolInterval=nrowInterval,
                   typeMatrixPlotX = c("GPUm f32 cpu",
                                       "GPUm f64 cpu",
                                       "GPUm f32 cuda",
                                       "GPUm f64 cuda",
                                       "glm.fit",
                                       "speedglm"),
                   typeMatrixPloty = typeMatrixPlotX,
                   f,fgpu, g = runif, Time = .5, namePlot="Conjugate Gradient for Logical Regresion",
                   ylabel="Time in log10(seconds)",
                   xlabel=bquote(bold("Size ") ~ bold(X %in% R^{n * (n/20)})~""),
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

    listMatrixComparison1 <- creationGPUmatrix_all(X)[typeMatrixPlotX]
    listMatrixComparison2 <- creationGPUmatrix_all(y)[typeMatrixPloty]

    timeRes <- TwoFunctionTimeCalculation(listMatrixComparison1,
                                          listMatrixComparison2,
                                          f,fgpu ,nrows, ncols,Time,
                                          typeMatrixPlotX, family=family)
    resTable <- cbind(timeRes,
                      rep(nrows,length(typeMatrixPlotX)),
                      typeMatrixPlotX)
    DataFrameTimes <- rbind(DataFrameTimes,resTable)
  }


  return(drawPlotFunction(DataFrameTimes,namePlot,xlabel = xlabel))
}


plotGLMRes <- plotGLM(f=match.fun("glm.fit.GPU"),
                         fgpu=match.fun("glm.fit.GPU"))
glm.fit.GPU(A1,A2,family = poisson())
