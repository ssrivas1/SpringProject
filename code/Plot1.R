library(ggpmisc)
library(ggplot2)
library(ggpubr)
theme_set(
  theme_minimal() +
    theme(legend.position = "top")
)

#Reading the transformed datafile

df = read.csv("C:\\Users\\sriva\\OneDrive\\Documents\\Spring'2022\\Python\\college_data_n.csv")
head(df)

df = df[,c("is_private","pct_phd_faculty","grad_rate")]
df$is_private = as.factor(df$is_private)

#Plot 1
#Scatter plot between Percent of PhD Faculty and Graduation Rate
#Segregating and creating two datasets - One for Private colleges and another for Non-Private Colleges

b <- ggplot(df, aes(x = pct_phd_faculty, y = grad_rate))

b + geom_point(aes(color = is_private, shape = is_private))+
  geom_smooth(aes(color = is_private, fill = is_private), 
              method = "lm", fullrange = TRUE) +
  facet_wrap(~is_private) +
  scale_color_manual(values = c("#00AFBB", "#E7B800" ))+
  scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
  theme_bw()+
  labs(title="Graduation Rate V/S PhD Faculty Percentage based on Type of College",
       x="Percentage of PhD Faculty", y="Graduation Rate")