# Big Data Analytics Traffic-Accident PA

Traffic accidents remain a leading cause of injury and mortal-
ity in the United States, presenting a critical challenge for public safety
and health. In response to this pressing issue, our proposed project fo-
cuses on the state of Pennsylvania, aiming to leverage the power of big
data analytics to mitigate the frequency and severity of these incidents.
The goal is to assist authorities in implementing preventive measures for
improved road safety, as well as to explore the integration of autonomous
self-driving technology for accident prevention. This project aims to use a
comprehensive nationwide dataset on car accidents that covers 49 states
of the US from February 2016 to March 2023; however, we will focus on
studying the dataset in PA state. The trained models will be utilized
to forecast accident-prone zones across various counties, utilizing the re-
fined dataset. Visual representations of these predicted hotspots will be
created using geospatial tools to comprehend the spatial distribution of
accidents.

![](figures\pa_traffic_accidents_50.gif)


Dependent variable 
- Severity

Independent variables 
- 'Grid': A geospatial feature representing specific areas or regions.
- 'Zipcode': The postal code of the accident location.
- 'avg_temp': Average temperature, which could influence driving conditions.
- 'avg_visibility': Average visibility, a critical factor in driving safety.
- 'Street', 'Crossing', 'Junction', 'Stop', 'Traffic_Signal'$: These are infrastructure-related features, each encoded using one-hot encoding to convert categorical data into a format suitable for modeling.


## Grid representation

The grid representation

![Drag Racing](figures\pa_traffic_accidents_map_all.png)
![Drag Racing](figures\pa_traffic_accidents_map_2019_10.png)
![Drag Racing](figures\pa_traffic_accidents_map_2020_10.png)


