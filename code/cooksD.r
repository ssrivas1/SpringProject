setwd("C:\\Users\\aghar\\Downloads\\msis-5223-deliverable-1-cobra-kai-main\\msis-5223-deliverable-1-cobra-kai-main\\data")


###########################################
#==============Read in data===============#
#Read in the data for both data sets.	#
###########################################

#==========================================
# Read in the data sample 
#==========================================

reduction_data1 = read.table("C:\\Users\\aghar\\Downloads\\msis-5223-deliverable-1-cobra-kai-main\\msis-5223-deliverable-1-cobra-kai-main\\data\\College_Data_Transformed.csv", quote = "",header=T, sep=",")

#Remove extra columns from the dataset and dividing it into two for PCA and FA
reduction_data = subset(reduction_data, select=-c(college_name,is_private))


reduction_data=as.data.frame(scale(reduction_data[complete.cases(reduction_data),]))
#outlier Analysis
mod <- lm(accept_rate ~ no_of_applications+no_of_apps_accepted+no_of_enrolled+PctTop10HS+PctTop25HS+no_of_FT+no_of_PT+outstate_tuition+room_board_cost+books_cost+personal_spending+pct_phd_faculty+pct_termianl_faculty+sf_ratio+pct_alumni_donors+exp_per_student+grad_rate+total_misc_cost, data = reduction_data)
cooksd <- cooks.distance(mod)
df<-as.data.frame(cooksd)
sample_size <- nrow(reduction_data)
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4/sample_size, col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>40/sample_size, names(cooksd),""), col="red")  # add labels
