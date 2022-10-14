library(dplyr)

set.seed(124)

setwd("/home/kurt/Documents/Github/CPSC3990-Fall2022/scripts")

# 40,000 maintains the number of points per class. 

# RandomDataGeneration:
point1x = c(runif(40000, min = 0, max = 1980))
point1y = c(runif(40000, min = 0, max = 1980))
point2x = c(point1x + 20)
point2y = c(point1y + 20)

xCategory = c((floor((point1x+10)/500)))
yCategory = c((floor((point1y+10)/500)))

randomData = data.frame(point1x, point1y, point2x, point2y, xCategory, yCategory, Classification = 99, row.names = NULL)

randomData <- randomData %>%
  mutate_if(is.numeric, round, digits = 5)

classifiedRandomData <- randomData %>%
  mutate(Classification = case_when (
    xCategory == 0 & yCategory == 0 ~ 12,
    xCategory == 1 & yCategory == 0 ~ 13,
    xCategory == 2 & yCategory == 0 ~ 14,
    xCategory == 3 & yCategory == 0 ~ 15,
    xCategory == 0 & yCategory == 1 ~ 8,
    xCategory == 1 & yCategory == 1 ~ 9,
    xCategory == 2 & yCategory == 1 ~ 10,
    xCategory == 3 & yCategory == 1 ~ 11,
    xCategory == 0 & yCategory == 2 ~ 4,
    xCategory == 1 & yCategory == 2 ~ 5,
    xCategory == 2 & yCategory == 2 ~ 6,
    xCategory == 3 & yCategory == 2 ~ 7,
    xCategory == 0 & yCategory == 3 ~ 0,
    xCategory == 1 & yCategory == 3 ~ 1,
    xCategory == 2 & yCategory == 3 ~ 2,
    xCategory == 3 & yCategory == 3 ~ 3,
  )
  )

finalDataAll <- subset(classifiedRandomData, select = -c(xCategory, yCategory))

class0Data <- subset(finalDataAll, Classification == 0)
class1Data <- subset(finalDataAll, Classification == 1)
class2Data <- subset(finalDataAll, Classification == 2)
class3Data <- subset(finalDataAll, Classification == 3)
class4Data <- subset(finalDataAll, Classification == 4)
class5Data <- subset(finalDataAll, Classification == 5)
class6Data <- subset(finalDataAll, Classification == 6)
class7Data <- subset(finalDataAll, Classification == 7)
class8Data <- subset(finalDataAll, Classification == 8)
class9Data <- subset(finalDataAll, Classification == 9)
class10Data <- subset(finalDataAll, Classification == 10)
class11Data <- subset(finalDataAll, Classification == 11)
class12Data <- subset(finalDataAll, Classification == 12)
class13Data <- subset(finalDataAll, Classification == 13)
class14Data <- subset(finalDataAll, Classification == 14)
class15Data <- subset(finalDataAll, Classification == 15)

write.csv(finalDataAll, "../data/16Classes/allData.csv", row.names = FALSE)
write.csv(class0Data, "../data/16Classes/class0Data.csv", row.names = FALSE)
write.csv(class1Data, "../data/16Classes/class1Data.csv", row.names = FALSE)
write.csv(class2Data, "../data/16Classes/class2Data.csv", row.names = FALSE)
write.csv(class3Data, "../data/16Classes/class3Data.csv", row.names = FALSE)
write.csv(class4Data, "../data/16Classes/class4Data.csv", row.names = FALSE)
write.csv(class5Data, "../data/16Classes/class5Data.csv", row.names = FALSE)
write.csv(class6Data, "../data/16Classes/class6Data.csv", row.names = FALSE)
write.csv(class7Data, "../data/16Classes/class7Data.csv", row.names = FALSE)
write.csv(class8Data, "../data/16Classes/class8Data.csv", row.names = FALSE)
write.csv(class9Data, "../data/16Classes/class9Data.csv", row.names = FALSE)
write.csv(class10Data, "../data/16Classes/class10Data.csv", row.names = FALSE)
write.csv(class11Data, "../data/16Classes/class11Data.csv", row.names = FALSE)
write.csv(class12Data, "../data/16Classes/class12Data.csv", row.names = FALSE)
write.csv(class13Data, "../data/16Classes/class13Data.csv", row.names = FALSE)
write.csv(class14Data, "../data/16Classes/class14Data.csv", row.names = FALSE)
write.csv(class15Data, "../data/16Classes/class15Data.csv", row.names = FALSE)
