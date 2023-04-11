SARIMA stands for Seasonal Auto-Regressive Integrated Moving Average.

"Seasonal" means the model can detect and emulate cyclic trends, in this case weekly patterns.
"Auto-Regressive" means the model predicts based on the recent values.
"Integrated" means we take the differences of consecutive terms before analysis.
"Moving Average" means the model predicts based on the recent error.

I thought this would be a good way to predict trends, because the model clearly shows a drop in sales each Sunday when the locals attend church.

It turns out to be really good at predicting most of the time, but if one of the dips is wider than expected, it will throw off the model's "metronome" and it will start predicting dips [reliably] a day or two before/after they happen.
