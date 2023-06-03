# Maca Interview

## Context
This dashboard uses data open-sourced by the Brazilian e-commerce company Olist. This company attracts independent sellers to sell their products online on Olist's marketplace. Different CSV files were combined to put data from marketing and seller's performance together. That way, we know how the sellers are brought to the platform and how they perform on the platform.

## What I built
I built a RevOps dashboard to the best of my knowledge of what RevOps represents. The dashboard gives information of average closing time, and monthly revenue. We also have a breakdown of percentage of deals closed per origin of contact, and the percentage of deals closed over time (monthly).

Then, the dashboard gives insights, such as what factors maximize the revenue. That way, we can make an informed decision on where to focus our efforts to generate more revenue.

I also included a sales forecasting model so we can see where sales are headed.

## How to run the service

## Decision log
1. Find data suitable for a RevOps dashboard. It was important to have data coming from at least two different domains to respect the project's requirements.
2. Research on RevOps. I wanted to learn what RevOps was about, to know what kind of insights would be interesting.
3. Decide on the tools to use for this project. Here, I decided to go with Streamlit, so that I can focus on the Python code and logic behind the dashboard and not worry about creating a nice UI. It's also easy to deploy and prototype.
4. Decide on what insights to show. Since we want information that ties marketing and revenue, and since I noticed that not all leads eventually turned into sales, I figured that it would interesting to know the ratio of leads that actually turn into sales. Then, I saw that there were many first points of contact, so it would be interesting to know led to more sales, so the company can focus their marketing efforts. Studying the data more, I saw that it took time from the first point of contact to making the sale, and that time must be important because it represents effort by the sale team, and ideally, you close as fast as possible to maximize profit.
5. Buid intelligent recommendations. From the insights above, I determined that machine learning techniques could be used to inform future decisions. I used Shap values with a simple decision tree model to determine what factors can maximize revenue. Then, I used a time series forecasting model to predict sales over the next 6 months. That way, the team can know where sales are headed with a confidence interval. I wanted the horizon to be set by the user, but I didn't have the time.
6. Given more time, I would have added the functionality to set a start and end date on the dashboard to analyze the data. The UI is there, but not the logic. I would have also like to visualize geographical data to see if location of the seller can affect its performance on the platform. 

## How long it took
- 1 day to find the dataset and research into RevOps
- 3 days to decide on what to build and actually build it
- 1 day for documentation and testing