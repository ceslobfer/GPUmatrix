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
class(gpu.glm.D93) <- c("speedglm","speedlm") # --> Fix me!!!
gpu.glm.D932 <- gpu.glm.D93
summary(gpu.glm.D93)

# Let's go with the big guys
library(gtools) # for logit
m <- 4000
n <- 1000
x <- matrix(runif(m*n),m,n)
sol <- rnorm(n)
y <- rbinom(m, 1, prob = inv.logit(x%*%sol))

system.time(s1 <- glm.fit(x,y,family = binomial())$coefficients)
system.time(s2 <- speedglm.wfit(y,x,family = binomial())$coefficients)
system.time(s3 <- glm.fit.GPU(x,y,family = binomial())$coefficients)
plot(s1,s2)
plot(s1,s3)
plot(s2-s3)


