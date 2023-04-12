##Aron Tobias

SARIMA stands for Seasonal Auto-Regressive Integrated Moving Average.

"Seasonal" means the model can detect and emulate cyclic trends, in this case weekly patterns.
"Auto-Regressive" means the model predicts based on the recent values.
"Integrated" means we take the differences of consecutive terms before analysis.
"Moving Average" means the model predicts based on the recent error.

I thought this would be a good way to predict trends, because the model clearly shows a drop in sales each Sunday when the locals attend church.

It turns out to be really good at maintaining a seasonal trend - so good that it found a hole in the data.
It also predicts the general direction of the data to follow pretty well.

The only loss measurements we have is the residual plot and whatever the stats window spits out - apparently AIC is a good metric.
