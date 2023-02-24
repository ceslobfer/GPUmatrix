
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
# a
dim(x)<- c(2,8)
# x
dim(xS)<- c(2,8)
# to_dense(xS)
dim(y)<- c(2,8)
# y
dim(yS)<- c(2,8)
# to_dense(yS)

rownames(x) <- c("a","b")
rownames(xS) <- c("a","b")
rownames(y) <- c("a","b")
rownames(yS) <- c("a","b")

dimnames(x) <- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(a) <- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(xS)<- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(y)<- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(yS)<- list(c("b","b"),c("a","b","c","d","e","f","g","h"))

# dimnames(x)
# dimnames(xS)
# dimnames(y)
# dimnames(yS)

x[,1]
xS[,1]
y[,1]
yS[,1]
x[,c(1,2)]
xS[,c(1,2)]
y[,c(1,2)]
yS[,c(1,2)]
x[1,]
xS[1,]
y[1,]
yS[1,]
x[c(1,2),]
xS[c(1,2),]
y[c(1,2),]
yS[c(1,2),]
a[16]
x[16]
xS[16]
y[16]
yS[16]
x[c(1,16)]
xS[c(1,16)]
y[c(1,16)]
yS[c(1,16)]
x[c(1,1),c(1,2)]
to_dense(xS[c(1,1),c(1,2)])
y[c(1,1),c(1,2)]
to_dense(yS[c(1,1),c(1,2)])


rowSums(x)
rowSums(xS)
rowSums(y)
rowSums(yS)

colSums(x)
colSums(xS)
colSums(y)
colSums(yS)


cbind(x,c(1,2))
to_dense(cbind(xS,c(1,2,3,4)))
cbind(y,c(1,2))
to_dense(cbind(yS,c(1,2,3,4)))

rbind(x,c(1,2,3,4))
to_dense(rbind(xS,c(1,2,3,4)))
rbind(y,c(1,2,3,4))
to_dense(rbind(yS,c(1,2,3,4)))

head(x)
head(xS)
head(y)
head(yS)
head(x,1)
head(xS,1)
head(y,1)
head(yS,1)

tail(x)
tail(xS)
tail(y)
tail(yS)
tail(x,1)
tail(xS,1)
tail(y,1)
tail(yS,1)

nrow(x)
nrow(xS)
nrow(y)
nrow(yS)
ncol(x)
ncol(xS)
ncol(y)
ncol(yS)

t(x)
to_dense(t(xS))
t(y)
to_dense(t(yS))

library(Matrix)
M <- Matrix(a)
crossprod(x)
crossprod(xS)
crossprod(y)
crossprod(yS)
crossprod(x,a)
crossprod(xS,a)
crossprod(y,a)
crossprod(yS,a)
crossprod(x,M)
crossprod(xS,M)
crossprod(y,M)
crossprod(yS,M)

tcrossprod(x)
tcrossprod(xS)
tcrossprod(y)
tcrossprod(yS)
tcrossprod(x,a)
tcrossprod(xS,a)
tcrossprod(y,a)
tcrossprod(yS,a)
tcrossprod(x,M)
tcrossprod(xS,M)
tcrossprod(y,M)
tcrossprod(yS,M)

outer(x,a)
outer(xS,a)
outer(y,a)
outer(yS,a)
outer(x,M)
outer(xS,M)
outer(y,M)
outer(yS,M)
