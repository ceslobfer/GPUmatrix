
a <- matrix(rnorm(16),4,4)
x <- gpu.matrix(a, type = "tensorflow")
xS <- gpu.matrix(a, type = "tensorflow",sparse = T)
y <- gpu.matrix(a)
yS <- gpu.matrix(a,sparse = T)


determinant(y)
determinant(yS)
determinant(x)
determinant(xS)
determinant(x,logarithm = F)
determinant(xS,logarithm = F)
determinant(y,logarithm = F)
determinant(yS,logarithm = F)

det(x)
det(xS)
det(y)
det(yS)

fft(x)
fft(xS)
fft(y)
fft(yS)

sort(x)
sort(xS)
sort(y)
sort(yS)

sort(x,decreasing = T)
sort(xS,decreasing = T)
sort(y,decreasing = T)
sort(yS,decreasing = T)

round(x)
round(xS)
round(y)
round(yS)

round(x,digits = 1)
round(xS,digits = 1)
round(y,digits = 1)
round(yS,digits = 1)

length(x)
length(xS)
length(y)
length(yS)

dim(x)
dim(xS)
dim(y)
dim(yS)

dim(a)<- c(2,8)
a
dim(x)<- c(2,8)
x
dim(xS)<- c(2,8)
to_dense(xS)
dim(y)<- c(2,8)
y
dim(yS)<- c(2,8)
to_dense(yS)

rownames(x) <- c("a","b")
rownames(xS) <- c("a","b")
rownames(y) <- c("a","b")
rownames(yS) <- c("a","b")

dimnames(x) <- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(a) <- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(xS)<- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(y)<- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(yS)<- list(c("b","b"),c("a","b","c","d","e","f","g","h"))

dimnames(x)
dimnames(xS)
dimnames(y)
dimnames(yS)

x[,1]
xS[,1]
y[,1]
yS[,1]
x[,c(1,2)]
xS[,c(1,2)]
y[,c(1,2)]
yS[,c(1,2)]


