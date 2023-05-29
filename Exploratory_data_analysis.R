metadata <- read.csv(file = "metadata.csv")
library(ggplot2)

cols <- c('darkgoldenrod', 'goldenrod', 'khaki', 'navy', 'royalblue', 'cornflowerblue')
legend_order <- c('BCC', 'MEL', 'SCC', 'ACK', 'NEV', 'SEK')
n <- length(metadata$diagnostic)
skin_dis <- sum(metadata$diagnostic == "ACK") + sum(metadata$diagnostic == "NEV") + sum(metadata$diagnostic == "SEK")
skin_can <- sum(metadata$diagnostic == "BCC") + sum(metadata$diagnostic == "MEL") + sum(metadata$diagnostic == "SCC")

group <- c(rep("Skin diseases" , 3) , rep("Skin cancer" , 3))
diagnostic <- c("ACK" , "NEV" , "SEK", "BCC", "MEL", "SCC")
freq <- c(sum(metadata$diagnostic == "ACK"), sum(metadata$diagnostic == "NEV"), sum(metadata$diagnostic == "SEK"), sum(metadata$diagnostic == "BCC"), sum(metadata$diagnostic == "MEL"), sum(metadata$diagnostic == "SCC"))
perc <- c(sum(metadata$diagnostic == "ACK")/skin_dis, sum(metadata$diagnostic == "NEV")/skin_dis, sum(metadata$diagnostic == "SEK")/skin_dis, sum(metadata$diagnostic == "BCC")/skin_can, sum(metadata$diagnostic == "MEL")/skin_can, sum(metadata$diagnostic == "SCC")/skin_can)
perc <- round(perc, digits = 3)
data <- data.frame(group, diagnostic, freq)

ggplot(data, aes(fill=diagnostic, y=freq, x=group)) + 
  geom_bar(position="dodge", stat="identity") + 
  labs(title='Frequencies of diagnoses in metadata') + 
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab('') + 
  ylab('') + 
  scale_fill_manual(values=cols, breaks=legend_order) +
  #scale_fill_discrete(breaks=legend_order, values=cols) +
  #theme(legend.position="none") + 
  geom_text(position = position_dodge(width= 0.9), aes(y=freq+30, label=perc))

