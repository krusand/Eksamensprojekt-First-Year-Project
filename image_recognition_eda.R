metadata <- read.csv(file = "metadata.csv")
library(ggplot2)

group <- c(rep("Skin diseases" , 3) , rep("Skin cancer" , 3))
diagnostic <- c("ACK" , "NEV" , "SEK", "BCC", "MEL", "SCC")
freq <- c(sum(metadata$diagnostic == "ACK"), sum(metadata$diagnostic == "NEV"), sum(metadata$diagnostic == "SEK"), sum(metadata$diagnostic == "BCC"), sum(metadata$diagnostic == "MEL"), sum(metadata$diagnostic == "SCC"))
data <- data.frame(group, diagnostic, freq)

ggplot(data, aes(fill=condition, y=freq, x=group)) + 
  geom_bar(position="dodge", stat="identity") + 
  labs(title='Frequencies of diagnoses in metadata') + 
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab('') + 
  ylab('') + 
  scale_fill_manual(values = c('navy', 'darkgoldenrod', 'goldenrod', 'royalblue', 'khaki', 'cornflowerblue')) +
  theme(legend.position="none") + 
  geom_text(position = position_dodge(width= 0.9), aes(y=freq+30, label=diagnostic))